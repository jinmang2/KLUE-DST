import random
import transformers
from transformers import RobertaPreTrainedModel
from transformers.models.roberta import RobertaModel, RobertaConfig

import torch.nn as nn
import torch.nn.functional as F

import torch
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

# loss_gen
def masked_cross_entropy_for_value(
    logits: torch.Tensor,
    target: torch.Tensor,
    pad_idx: int = 0
) -> torch.Tensor:
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss


class RobertaConfig(RobertaConfig):

    def __init__(
        self,
        teacher_forcing: float = 0.5,
        parallel_decoding: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.teacher_forcing = teacher_forcing
        self.parallel_decoding = parallel_decoding


@dataclass
class DialogueStateTrackingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    point_outputs: Optional[torch.FloatTensor] = None
    gate_outputs: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class SlotGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pad_token_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.parallel_decoding = config.parallel_decoding

        self.embed = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            padding_idx=self.pad_token_id
        )  # shared with encoder

        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True
        )

        self.gating2id = {"none": 0, "dontcare": 1, "ptr": 2, "yes":3, "no": 4}
        self.id2gating = {v: k for k, v in self.gating2id.items()}

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.w_gen = nn.Linear(self.hidden_size * 3, 1)
        self.w_gate = nn.Linear(self.hidden_size, self.num_gates)

    @property
    def gating2id(self):
        return self._gating2id

    @gating2id.setter
    def gating2id(self, val: Dict[str, int]):
        self._gating2id = val
        self.num_gates = len(self._gating2id.keys())

    def set_slot_idx(self, slot_vocab_idx: List[List[int]]):
        whole = []
        max_length = max(map(len, slot_vocab_idx))
        for idx in slot_vocab_idx:
            if len(idx) < max_length:
                gap = max_length - len(idx)
                idx.extend([self.pad_token_id] * gap)
            whole.append(idx)
        self.slot_embed_idx: List[List[int]] = whole

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        hidden: torch.Tensor,
        input_masks: torch.Tensor,
        max_len: int,
        teacher: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_masks = input_masks.ne(1)
        # J, slot_meta : key : [domain, slot] ex> LongTensor([1,2])
        # J,2
        batch_size = encoder_output.size(0)
        slot = torch.LongTensor(self.slot_embed_idx).to(input_ids.device)
        # slot_embedding
        slot_e = torch.sum(self.embed(slot), 1)  # J, d
        J = slot_e.size(0)

        if self.parallel_decoding:
            all_point_outputs = torch.zeros(batch_size, J, max_len, self.vocab_size).to(input_ids.device)
            all_gate_outputs = torch.zeros(batch_size, J, self.num_gates).to(input_ids.device)

            w = slot_e.repeat(batch_size, 1).unsqueeze(1)
            hidden = hidden.repeat_interleave(J, dim=1)
            encoder_output = encoder_output.repeat_interleave(J, dim=0)
            input_ids = input_ids.repeat_interleave(J, dim=0)
            input_masks = input_masks.repeat_interleave(J, dim=0)
            num_decoding = 1

        else:
            # Seperate Decoding
            all_point_outputs = torch.zeros(J, batch_size, max_len, self.vocab_size).to(input_ids.device)
            all_gate_outputs = torch.zeros(J, batch_size, self.num_gates).to(input_ids.device)
            num_decoding = J

        for j in range(num_decoding):

            if not self.parallel_decoding:
                w = slot_e[j].expand(batch_size, 1, self.hidden_size)

            for k in range(max_len):
                w = self.dropout(w)
                _, hidden = self.gru(w, hidden)  # 1,B,D

                # B,T,D * B,D,1 => B,T
                attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
                MASKED_VALUE = (2 ** 15) if attn_e.dtype == torch.half else 1e9
                attn_e = attn_e.squeeze(-1).masked_fill(input_masks, -MASKED_VALUE)
                attn_history = F.softmax(attn_e, -1)  # B,T

                # B,D * D,V => B,V
                attn_v = torch.matmul(hidden.squeeze(0), self.embed.weight.transpose(0, 1))  # B,V
                attn_vocab = F.softmax(attn_v, -1)

                # B,1,T * B,T,D => B,1,D
                context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D
                p_gen = torch.sigmoid(self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1)))  # B,1
                p_gen = p_gen.squeeze(-1)

                p_context_ptr = torch.zeros_like(attn_vocab).to(input_ids.device)
                p_context_ptr.scatter_add_(1, input_ids, attn_history)  # copy B,V
                p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
                _, w_idx = p_final.max(-1)

                if teacher is not None:
                    if self.parallel_decoding:
                        w = self.embed(teacher[:, :, k]).reshape(batch_size * J, 1, -1)
                    else:
                        w = self.embed(teacher[:, j, k]).unsqueeze(1)
                else:
                    w = self.embed(w_idx).unsqueeze(1)  # B,1,D

                if k == 0:
                    gated_logit = self.w_gate(context.squeeze(1))  # B,3
                    if self.parallel_decoding:
                        all_gate_outputs = gated_logit.view(batch_size, J, self.num_gates)
                    else:
                        _, gated = gated_logit.max(1)  # maybe `-1` would be more clear
                        all_gate_outputs[j] = gated_logit

                if self.parallel_decoding:
                    all_point_outputs[:, :, k, :] = p_final.view(batch_size, J, self.vocab_size)
                else:
                    all_point_outputs[j, :, k, :] = p_final

        if not self.parallel_decoding:
            all_point_outputs = all_point_outputs.transpose(0, 1)
            all_gate_outputs = all_gate_outputs.transpose(0, 1)

        return all_point_outputs, all_gate_outputs

class RobertaForDialogueStateTracking(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.teacher_forcing = config.teacher_forcing

        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.decoder = SlotGenerator(config)

        self.post_init()

    def _tie_weights(self):
        # Share the embedding layer for both encoder and decoder
        self.decoder.embed.weight = self.roberta.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        gating_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        target_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        encoder_output = outputs[0] # last_hidden_state
        pooler_output = outputs[1].unsqueeze(0) # pooler_output

        max_len, teacher = 10, None
        if target_ids is not None:
            max_len = target_ids.size(-1)
            if self.teacher_forcing > 0.0 and random.random() < self.teacher_forcing:
                teacher = target_ids

        all_point_outputs, all_gate_outputs = self.decoder(
            input_ids=input_ids,
            encoder_output=encoder_output,
            hidden=pooler_output,
            input_masks=attention_mask,
            max_len=max_len,
            teacher=teacher,
        )

        loss = None
        if target_ids is not None:
            # generation loss
            loss_gen = masked_cross_entropy_for_value(
                all_point_outputs.contiguous(),
                target_ids.contiguous().view(-1),
                self.decoder.pad_token_id,
            )
            # gate loss
            loss_fct = nn.CrossEntropyLoss()
            loss_gate = loss_fct(
                all_gate_outputs.contiguous().view(-1, self.decoder.num_gates),
                gating_ids.contiguous().view(-1),
            )
            # total loss = generation loss + gate loss
            loss = loss_gen + loss_gate

        all_point_outputs = all_point_outputs.permute(0, 2, 1, 3)

        if not return_dict:
            output = (all_point_outputs, all_gate_outputs,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DialogueStateTrackingOutput(
            loss=loss,
            point_outputs=all_point_outputs,
            gate_outputs=all_gate_outputs,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
