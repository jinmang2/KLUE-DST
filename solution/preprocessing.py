from typing import List, Dict, Tuple, Union, Sequence, Callable
from datasets import Dataset
from copy import deepcopy
import torch

from .load_ontology import load_json_file


def build_slot_from_ontology(file_path: str = "ontology.json") -> Tuple[List[str], List[str]]:
    domains, slots = [], []

    ontology = load_json_file(file_path)
    for line in list(ontology.keys()):
        domain, slot = line.split("-")
        domains.append(domain)
        slots.append(slot)

    return domains, slots


def split_slot(dom_slot_value: str, get_domain_slot: bool = False) -> Tuple[str, ...]:
    try:
        dom, slot, value = dom_slot_value.split("-")
    except ValueError:
        tempo = dom_slot_value.split("-")
        if len(tempo) < 2:
            return dom_slot_value, dom_slot_value, dom_slot_value
        dom, slot = tempo[0], tempo[1]
        value = dom_slot_value.replace("%s-%s-" % (dom, slot), "").strip()

    if get_domain_slot:
        return "%s-%s" % (dom, slot), value
    return dom, slot, value


def convert_state_dict(state: Sequence[str]) -> Dict[str, str]:
    dic = {}
    for slot in state:
        s, v = split_slot(slot, get_domain_slot=True)
        dic[s] = v
    return dic


def build_slot_meta(data: List[Dict[str, List[dict]]]) -> List[str]:
    slot_meta = []
    for dialog in data:
        for turn in dialog["dialogue"]:
            if not turn.get("state"):
                continue
            for dom_slot_value in turn["state"]:
                domain_slot, _ = split_slot(dom_slot_value, get_domain_slot=True)
                if domain_slot not in slot_meta:
                    slot_meta.append(domain_slot)
    return sorted(slot_meta)


def merge_slot_meta(
    slot_meta: List[str],
    slot_from_dials: List[str],
) -> List[str]:
    exist_slot_set = set(slot_meta)
    for slot in slot_from_dials:
        exist_slot_set.add(slot)
    return sorted(list(exist_slot_set))


def get_slot_meta(
    dials: Union[List[Dict[str, List[dict]]], Dataset],
) -> List[str]:
    # Read ontology file if exists and store the slots
    _, slot_meta = build_slot_from_ontology()
    # Extract slots from a given dialogue and merge with ontology slots
    slot_from_dials = build_slot_meta(dials)
    slot_meta = merge_slot_meta(slot_meta, slot_from_dials)
    return slot_meta


def get_examples_from_dialogues_fn(tokenizer: "PreTrainedTokenizerBase") -> Callable:
    def get_examples_from_dialogues(dialogues) -> Dict[str, List]:
        dialogue_examples = {
            "guid": [],
            "context_turns": [],
            "current_turn": [],
            "label": [],
        }
        for guid, dialogue in zip(dialogues["guid"], dialogues["dialogue"]):
            d_idx = 0
            history = ""
            for idx, turn in enumerate(dialogue):
                if turn["role"] != "user":
                    continue

                sys_uttr = dialogue[idx - 1]["text"] if idx else ""

                current_text = tokenizer.tokenize(
                    sys_uttr, turn["text"], add_special_tokens=True
                )[1:-1]

                current_text = " ".join([sys_uttr, tokenizer.sep_token, turn["text"]])

                dialogue_examples["guid"].append(f"{guid}-{d_idx}")
                dialogue_examples["context_turns"].append(history)
                dialogue_examples["current_turn"].append(current_text)
                dialogue_examples["label"].append(turn["state"])

                if history:
                    history += " " + tokenizer.sep_token + " "

                history += current_text

                d_idx += 1

        return dialogue_examples

    return get_examples_from_dialogues


def get_convert_examples_to_features_fn(
    tokenizer: "PreTrainedTokenizerBase",
    slot_meta: List[str],
    gating2id: Dict[str, str],
    label2id: Dict[str, str],
) -> Callable:
    def convert_examples_to_features(examples):
        assert tokenizer.truncation_side == "left", (
            "DialogueStateTracking's feature form is `[dialogue history][SEP][user_uttr_{t}][SEP][sys_uttr_{t}]` "
            "where [dialogue history] == [user_uttr_{1}][SEP][sys_uttr_{1}][SEP]...[SEP][user_uttr_{t-1}]. "
            "So, right part of input is important and this prep_fn requires to `truncation_side` == 'left'. "
            f"However, your tokenizer's truncation_side is {tokenizer.truncation_side}. Check it plz."
        )

        dialogue_features = {}

        dialogue_features["input_ids"] = tokenizer(
            examples["context_turns"],
            examples["current_turn"],
            truncation=True,
            add_special_tokens=True,
        )["input_ids"]

        dialogue_features["guid"] = []
        dialogue_features["target_ids"] = []
        dialogue_features["gating_ids"] = []
        dialogue_features["labels"] = []

        for guid, labels in zip(examples["guid"], examples["label"]):
            state = convert_state_dict(labels)
            values = [state.get(slot, "none") for slot in slot_meta]
            target_ids = tokenizer(values, padding=True)["input_ids"]
            gating_ids = [gating2id.get(value, gating2id["ptr"]) for value in values]

            dialogue_features["guid"].append(guid)
            dialogue_features["target_ids"].append(target_ids)
            dialogue_features["gating_ids"].append(gating_ids)
            dialogue_features["labels"].append(
                [label2id.get(label, 0) for label in labels]
            )

        dialogue_features["gating_ids"] = dialogue_features["gating_ids"]

        return dialogue_features

    return convert_examples_to_features
