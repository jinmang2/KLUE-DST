from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer

from solution.preprocessing import (
    get_slot_meta,
    get_examples_from_dialogues_fn,
    get_convert_examples_to_features_fn
)
from solution.model import (
    RobertaConfig,
    RobertaForDialogueStateTracking
)
from solution.metrics import (
    get_recover_state_fn,
    get_compute_metrics,
)
from solution.collator import PadCollator


def main():
    model_name = "klue/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        truncation_side="left",
        cache_dir="cache",
    )

    wos = load_dataset("klue", "wos")

    slot_meta = get_slot_meta(wos["train"])
    # eval_slot_meta = get_slot_meta(wos["validation"])

    gating2id = {"none": 0, "dontcare": 1, "ptr": 2, "yes": 3, "no": 4}
    id2gating = {i: g for g, i in gating2id.items()}

    get_examples_from_dialogues = get_examples_from_dialogues_fn(tokenizer)

    train_dataset = wos["train"]
    eval_dataset = wos["validation"]

    train_examples = train_dataset.map(
        get_examples_from_dialogues,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=train_dataset.column_names,
    )
    eval_examples = eval_dataset.map(
        get_examples_from_dialogues,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=eval_dataset.column_names,
    )

    train_labels = [label for labels in train_examples["label"] for label in labels]
    eval_labels = [label for labels in eval_examples["label"] for label in labels]
    labels = train_labels + eval_labels

    labels = [kv[0] for kv in Counter(labels).most_common()]

    id2label = dict(enumerate(labels, start=1))
    id2label[0] = "<pad>"
    label2id = {l: i for i, l in id2label.items()}

    convert_examples_to_features = get_convert_examples_to_features_fn(
        tokenizer, slot_meta, gating2id, label2id
    )

    train_features = train_examples.map(
        convert_examples_to_features,
        batched=True,
        batch_size=1000,
        num_proc=None,
        remove_columns=train_examples.column_names,
    )
    eval_features = eval_examples.map(
        convert_examples_to_features,
        batched=True,
        batch_size=1000,
        num_proc=None,
        remove_columns=eval_examples.column_names,
    )

    config = RobertaConfig.from_pretrained(model_name, cache_dir="cache")
    model = RobertaForDialogueStateTracking.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=config,
        cache_dir="cache",
    )

    slot_vocab = [
        tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        for slot in slot_meta
    ]
    model.to("cuda")
    model.decoder.set_slot_idx(slot_vocab)

    recover_state = get_recover_state_fn(slot_meta, id2gating, tokenizer)
    compute_metrics = get_compute_metrics(recover_state)

    training_args = TrainingArguments(
        output_dir="outputs",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=5,
        lr_scheduler_type="cosine",
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=3,
        # fp16=True,
        eval_accumulation_steps=100,
        optim="adafactor",
        # run_name="dst-trade",
        # report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="joint_goal_acc",
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=train_features,
        eval_dataset=eval_features,
        data_collator=PadCollator(tokenizer),
        compute_metrics=compute_metrics,
    )

    results = trainer.train()
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
