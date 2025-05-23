from dataset import Preprocessing_Maccrobat
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from dataloader import NER_Dataset
from util import *
import torch


def main():
    # Preprocessing data
    tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
    dataset_folder = "./MACCROBAT2018/"

    Maccrobat_builder = Preprocessing_Maccrobat(dataset_folder, tokenizer)
    input_texts, input_labels = Maccrobat_builder.process()

    label2id = Preprocessing_Maccrobat.build_label2id(input_labels)
    id2label = {v: k for k, v in label2id.items()}

    # Split data
    inputs_train, inputs_val, labels_train, labels_val = train_test_split(
        input_texts, input_labels, test_size=0.2, random_state=42
    )

    # Dataloader
    train_set = NER_Dataset(inputs_train, labels_train, tokenizer, label2id)
    val_set = NER_Dataset(inputs_val, labels_val, tokenizer, label2id)

    # Modelling and training
    model = AutoModelForTokenClassification.from_pretrained(
        "d4data/biomedical-ner-all",
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir="./output",
        learning_rate=1e-4,
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        optim="adamw_torch",
        logging_strategy="epoch",
    )

    metric_callback = MetricsCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[metric_callback],
    )
    # trainer.train()

    display_metrics_table(metric_callback.metrics)

    # Inference
    test_sentence = """
        CASE: A 28-year-old previously healthy man presented with a 6-week history of palpitations.
        The symptoms occurred during rest, 2–3 times per week, lasted up to 30 minutes at a time and were associated with dyspnea.
        Except for a grade 2/6 holosystolic tricuspid regurgitation murmur (best heard at the left sternal border with inspiratory accentuation), physical examination yielded unremarkable findings.
        An electrocardiogram (ECG) revealed normal sinus rhythm and a Wolff– Parkinson– White pre-excitation pattern (Fig.1: Top), produced by a right-sided accessory pathway.
        Transthoracic echocardiography demonstrated the presence of Ebstein's anomaly of the tricuspid valve, with apical displacement of the valve and formation of an “atrialized” right ventricle (a functional unit between the right atrium and the inlet [inflow] portion of the right ventricle) (Fig.2).
        The anterior tricuspid valve leaflet was elongated (Fig.2C, arrow), whereas the septal leaflet was rudimentary (Fig.2C, arrowhead).
        Contrast echocardiography using saline revealed a patent foramen ovale with right-to-left shunting and bubbles in the left atrium (Fig.2D).
        The patient underwent an electrophysiologic study with mapping of the accessory pathway, followed by radiofrequency ablation (interruption of the pathway using the heat generated by electromagnetic waves at the tip of an ablation catheter).
        His post-ablation ECG showed a prolonged PR interval and an odd “second” QRS complex in leads III, aVF and V2–V4 (Fig.1Bottom), a consequence of abnormal impulse conduction in the “atrialized” right ventricle.
        The patient reported no recurrence of palpitations at follow-up 6 months after the ablation.
    """

    input = torch.as_tensor(
        [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(test_sentence))]
    )

    input = input.to("cuda")

    outputs = model(input)
    _, preds = torch.max(outputs.logits, -1)
    preds = preds[0].cpu().numpy()

    for token, pred in zip(test_sentence.split(), preds):
        print(f"{token}\t{id2label[pred]}")


if __name__ == "__main__":
    main()
