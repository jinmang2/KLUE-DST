from typing import Any, Dict, List, Sequence, Tuple
from transformers import PreTrainedTokenizerBase


def get_recover_state_fn(
    slot_meta: List[str],
    id2gating: Dict[int, str],
    tokenizer: PreTrainedTokenizerBase,
):
    def recover_state(
        gate_list: List[int],
        gen_list: List[List[int]],
    ) -> List[str]:
        assert len(gate_list) == len(slot_meta)
        assert len(gen_list) == len(slot_meta)

        recovered = []
        for slot, gate, value in zip(slot_meta, gate_list, gen_list):
            gate = id2gating.get(gate, "none")
            if gate in ["none"]:
                continue
            elif gate in ["dontcare", "yes", "no"]:
                recovered.append("%s-%s" % (slot, gate))
                continue
            elif gate in ["ptr"]:
                # Append a token until special tokens appear
                token_id_list = []
                for id_ in value:
                    if id_ in tokenizer.all_special_ids:
                        break
                    token_id_list.append(id_)
                value = tokenizer.decode(token_id_list, skip_special_tokens=True)
                # This is a basic post-processing for generative DST models based on wordpiece
                # (using punctuation split)
                value = value.replace(" : ", ":").replace(" , ", ", ").replace("##", "")
            else:
                raise ValueError(
                    f"f{id2gating[gate]} do not support. [none|dontcare|ptr|yes|no]"
                )

            if value == "none":  # type: ignore[comparison-overlap]
                continue

        recovered.append("%s-%s" % (slot, value))
        return recovered

    return recover_state


# joint goal acc
def wos_jga(
    pred_steps: Sequence[Sequence[str]], trgt_steps: Sequence[Sequence[str]]
) -> Any:
    total, joint_goal_acc = 0, 0
    for (pred_batch, trgt_batch) in zip(pred_steps, trgt_steps):
        for (pred, trgt) in zip(pred_batch, trgt_batch):
            if set(pred) == set(trgt):
                joint_goal_acc += 1
            total += 1

    joint_goal_acc_score = joint_goal_acc / float(total) if total != 0 else 0
    return joint_goal_acc_score * 100.0


# slot micro f1
def compute_prf_for_wos(
    gold: Sequence[str], pred: Sequence[str]
) -> Tuple[float, float, float, float]:
    """Most of the below code is from https://github.com/jasonwu0731/trade-dst"""
    tp, fp, fn = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                tp += 1
            else:
                fn += 1
        for p in pred:
            if p not in gold:
                fp += 1
        precision = tp / float(tp + fp) if (tp + fp) != 0 else 0
        recall = tp / float(tp + fn) if (tp + fn) != 0 else 0
        f1 = (
            2 * precision * recall / float(precision + recall)
            if (precision + recall) != 0
            else 0
        )
    else:
        if len(pred) == 0:
            precision, recall, f1, count = 1, 1, 1, 1
        else:
            precision, recall, f1, count = 0, 0, 0, 1
    return f1, recall, precision, count


def wos_slot_micro_f1(
    pred_steps: Sequence[Sequence[str]], trgt_steps: Sequence[Sequence[str]]
) -> Any:
    count, f1 = 0, 0
    for (pred_batch, trgt_batch) in zip(pred_steps, trgt_steps):
        for (pred, trgt) in zip(pred_batch, trgt_batch):
            curr_f1, _, _, curr_count = compute_prf_for_wos(gold=trgt, pred=pred)
            f1 += curr_f1
            count += curr_count

    f1_score = f1 / float(count) if count != 0 else 0
    return f1_score * 100.0


def get_compute_metrics(recover_state):
    def compute_metrics(preds):
        point_outputs = preds.predictions[0]  # generated_ids
        gate_outputs = preds.predictions[1]  # gated_ids

        prs = [
            recover_state(gate, gen)
            for gate, gen in zip(gate_outputs.tolist(), point_outputs.tolist())
        ]
        gts = preds.label_ids.tolist()
        gts = [
            [id2label[label_id] for label_id in label_ids if label_id != 0]
            for label_ids in gts
        ]

        return {
            "joint_goal_acc": wos_jga(prs, gts),
            "slot_micro_f1": wos_slot_micro_f1(prs, gts),
        }

    return compute_metrics
