from transformers import TrainingArguments, Trainer
from util import *
from util import *
import evaluate
import numpy as np


def main():
    # Load and preprocess the dataset
    sentences, sentence_tags = load_dataset()

    (
        train_sentences,
        valid_sentences,
        test_sentences,
        train_tags,
        valid_tags,
        test_tags,
    ) = preprocess_data(sentences, sentence_tags)

    sentences, sentence_tags = load_dataset()

    (
        train_sentences,
        valid_sentences,
        test_sentences,
        train_tags,
        valid_tags,
        test_tags,
    ) = preprocess_data(sentences, sentence_tags)

    # Define label2id
    unique_tags = set(
        tag
        for sentence_tags in train_tags + valid_tags + test_tags
        for tag in sentence_tags
    )
    label2id = {tag: id for id, tag in enumerate(sorted(unique_tags))}

    tokenizer, model = modelling()

    train_dataset, valid_dataset, test_dataset = dataset_loader(
        train_sentences,
        train_tags,
        valid_sentences,
        valid_tags,
        test_sentences,
        test_tags,
        tokenizer,
        label2id,
    )

    training_args = TrainingArguments(
        output_dir="./out_dir",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    ignore_label = len(label2id)

    def compute_metric(eval_pred):
        accuracy = evaluate.load("accuracy")
        predictions, labels = eval_pred
        mask = labels != ignore_label
        predictions = np.argmax(predictions, axis=-1)
        accuracy_score = accuracy.compute(
            predictions=predictions[mask], references=labels[mask]
        )
        return accuracy_score

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metric,
    )
    trainer.train()


if __name__ == "__main__":
    main()
