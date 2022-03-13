from typing import List, Dict
import torch


class PadCollator:

    def __init__(self, tokenizer, max_seq_length=None):
        self.tokenizer = tokenizer
        if max_seq_length is None:
            self.max_seq_length = tokenizer.model_max_length
        self.pad_on_right = tokenizer.padding_side == "right"

    @staticmethod
    def pad_id_of_matrix(
        arrays: List[torch.LongTensor],
        pad_idx: int,
        pad_on_right: bool,
        max_length: int = None,
    ):
        if max_length is None:
            max_length = max([array.size(-1) for array in arrays])

        new_arrays = []
        for array in arrays:
            pad = torch.ones(array.size(0), (max_length - array.size(-1))) * pad_idx
            padded = [array, pad.long()] if pad_on_right else [pad.long(), array]
            new_arrays.append(torch.cat(padded, dim=-1).unsqueeze(dim=0))

        return torch.cat(new_arrays, dim=0)

    def __call__(self, features: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        target_max_length = 0
        target_ids = []
        labels = []
        for inputs in features:
            _ = inputs.pop("guid", None)
            target_id = inputs.pop("target_ids", None)
            if target_id is not None:
                target_id = torch.LongTensor(target_id)[:, 1:] # remove [CLS] token id
                target_max_length = max(target_max_length, target_id.size(-1))
                target_ids.append(target_id)
            label = inputs.pop("labels", None)
            if label is not None:
                labels.append(label)

        target_ids = self.pad_id_of_matrix(
            arrays=target_ids,
            pad_idx=self.tokenizer.pad_token_id,
            pad_on_right=self.pad_on_right,
            max_length=target_max_length
        )

        batch = self.tokenizer.pad(
            features,
            padding="longest",
            max_length=self.max_seq_length,
            return_tensors="pt"
        )

        max_len = max(len(label) for label in labels)
        labels = torch.LongTensor([label + [0] * (max_len - len(label)) for label in labels])

        batch.update({"target_ids": target_ids, "labels": labels})

        return batch
