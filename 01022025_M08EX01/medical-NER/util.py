import evaluate
import numpy as np
from transformers import TrainerCallback
from prettytable import PrettyTable
from typing import List, Dict, Any

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mask = labels != 0
    predictions = np.argmax(predictions, axis=-1)
    return accuracy.compute(predictions=predictions[mask], references=labels[mask])


class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.metrics: List[Dict[str, Any]] = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.is_local_process_zero:
            train_loss = None
            if state.log_history:
                # Find the most recent training loss
                for entry in reversed(state.log_history):
                    if "loss" in entry:
                        train_loss = entry["loss"]
                        break

            self.metrics.append(
                {
                    "epoch": state.epoch,
                    "train_loss": train_loss,
                    "eval_loss": metrics.get("eval_loss"),
                    "eval_accuracy": metrics.get("eval_accuracy"),
                }
            )


def display_metrics_table(metrics):
    table = PrettyTable()
    table.field_names = ["Epoch", "Train Loss", "Validation Loss", "Accuracy"]
    for m in metrics:
        table.add_row(
            [
                f"{m['epoch']:.2f}" if m["epoch"] is not None else "N/A",
                f"{m['train_loss']:.4f}" if m["train_loss"] is not None else "N/A",
                f"{m['eval_loss']:.4f}" if m["eval_loss"] is not None else "N/A",
                (
                    f"{m['eval_accuracy']:.4f}"
                    if m["eval_accuracy"] is not None
                    else "N/A"
                ),
            ]
        )
    print(table)
