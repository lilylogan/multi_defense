from openbackdoor.victims import Victim
from .log import logger
from .metrics import classification_metrics, detection_metrics
from typing import *
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

EVALTASKS = {
    "classification": classification_metrics,
    "detection": detection_metrics
}

def evaluate_classification(model: Victim, eval_dataloader, metrics: Optional[List[str]]=["accuracy"]):
    # effectiveness
    results = {}
    dev_scores = []
    main_metric = metrics[0]
    for key, dataloader in eval_dataloader.items():
        results[key] = {}
        logger.info("***** Running evaluation on {} *****".format(key))
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()
        outputs, labels = [], []
        print(dataloader)
        for batch in tqdm(dataloader, desc="Evaluating"):
            valid_batch = {}
            label_list = batch["label"].tolist()
            txt = []
            lbl = []
            poison_lbl = []
            for i in range(len(batch["text"])):
                if str(batch["text"][i]) != "nan":
                    txt.append(batch["text"][i])
                    lbl.append(label_list[i])
                    poison_lbl.append(batch["poison_label"][i])
            valid_batch["text"] = txt
            valid_batch["label"] = torch.tensor(lbl)
            valid_batch["poison_label"] = poison_lbl


            batch_inputs, batch_labels = model.process(valid_batch)

            with torch.no_grad():
                batch_outputs = model(batch_inputs)
            outputs.extend(torch.argmax(batch_outputs.logits, dim=-1).cpu().tolist())

            labels.extend(batch_labels.cpu().tolist())

        logger.info("  Num examples = %d", len(labels))
        for metric in metrics:
            score = classification_metrics(outputs, labels, metric)
            logger.info("  {} on {}: {}".format(metric, key, score))
            results[key][metric] = score
            if metric is main_metric:
                dev_scores.append(score)

    return results, np.mean(dev_scores)


def evaluate_step(model: Victim, dataloader, metric: str):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch_inputs, batch_labels = model.process(batch)
            output = model(batch_inputs).logits
            preds.extend(torch.argmax(output, dim=-1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
    score = classification_metrics(preds, labels, metric=metric)
    return score

def evaluate_detection(preds, labels, split: str, metrics: Optional[List[str]]=["FRR", "FAR"]):
    scores = {}
    for metric in metrics:
        score = detection_metrics(preds, labels, metric=metric)
        logger.info("{} on {}: {}".format(metric, split, score))
        scores[metric] = score
    return score, scores
