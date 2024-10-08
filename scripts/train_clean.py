import os
import json
import argparse
import sys
# adding OB to the system path
sys.path.insert(0, '../')
import openbackdoor as ob
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.victims import Victim
from openbackdoor.attackers import load_attacker
# from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config #, logger
from openbackdoor.utils.visualize import display_results
from openbackdoor.victims.plms import PLMVictim
from torch.utils.data import DataLoader
from openbackdoor.utils.log import init_logger
from openbackdoor.trainers import trainer



from openbackdoor.utils import evaluate_classification
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
from datetime import datetime
import torch.nn as nn
from tqdm import tqdm
from typing import *
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from umap import UMAP
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_base_model(model_name):
    if args.data == "agnews":
        num_classes = 4
    elif args.data == "blog":
        num_classes = 3
    else:
        num_classes = 2
    if args.model == "roberta":
        base_model = PLMVictim(model="roberta",
                          path="roberta-base", num_classes=num_classes)
    elif args.model == "bert":
        base_model = PLMVictim(model="bert",
                          path="bert-base-uncased", num_classes=num_classes)
    elif args.model == "albert":
        base_model = PLMVictim(model="albert",
                          path="albert-base-v2", num_classes=num_classes)
    elif args.model == "distilbert":
        base_model = PLMVictim(model="distilbert",
                          path="distilbert-base-uncased", num_classes=num_classes)
    elif args.model == "xlnet":
        base_model = PLMVictim(model="xlnet",
                          path="xlnet-base-cased", num_classes=num_classes)
    else:
        logger.info("\nMore models to be added.")
    return base_model



def train(args, base_model:Victim, dataset):
    """
    Train a clean victim model with original clean data.

    Args:
        victim (:obj:`Victim`): the victim model.
        dataset (:obj:`dictionary`): the dataset dictionary, containing train/dev/test sets.

    Returns:
        :obj:`Victim`: the clean model.

    """
    if args.data in ["agnews", "yelp"]:
        batch = 16
    else:
        batch = 32
    Trainer = trainer.Trainer(poison_rate=1,
                              model=args.model,
                              data=args.data,
                              poison_method='none',
                              epochs=5,
                              batch_size=batch,
                              warm_up_epochs=3,
                              save_path="./models",
                              poison_setting='none')
    model = Trainer.train(base_model, dataset, metrics=["accuracy"])

    return model


def main(args):
    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.rs)

    # input dir
    if args.data in ["sst-2", "yelp", "hsol", "offenseval"]:
        target_label = "1"
    else:
        target_label = "0"

    # clean data
    clean_dir = os.path.join('./data/clean', args.data)
    or_train = pd.read_csv(os.path.join(clean_dir, "train.csv"))
    or_dev = pd.read_csv(os.path.join(clean_dir, "dev.csv"))
    or_test = pd.read_csv(os.path.join(clean_dir, "test.csv"))

    or_train = or_train[["text", "label"]]
    or_dev = or_dev[["text", "label"]]
    or_test = or_test[["text", "label"]]

    # drop nan values
    or_train = or_train.dropna(subset=['text', 'label'])
    or_dev = or_dev.dropna(subset=['text', 'label'])
    or_test = or_test.dropna(subset=['text', 'label'])

    or_train = or_train.rename(columns={"text": "0", "label": "1"})
    or_train["2"] = 0
    or_train.columns.astype(int)

    or_dev = or_dev.rename(columns={"text": "0", "label": "1"})
    or_dev["2"] = 0
    or_dev.columns.astype(int)

    or_test = or_test.rename(columns={"text": "0", "label": "1"})
    or_test["2"] = 0
    or_test.columns.astype(int)

    or_train.to_csv(os.path.join(clean_dir, "train-clean.csv"))
    or_dev.to_csv(os.path.join(clean_dir, "dev-clean.csv"))
    or_test.to_csv(os.path.join(clean_dir, "test-clean.csv"))

    # data_dir = os.path.join('./poison_data/', args.llm, args.data, target_label, args.poisoner, 'default/')

    # setup output dir
    out_dir = os.path.join('logs/', args.data, args.model, 'clean/')
    os.makedirs(out_dir, exist_ok=True)

    # set up logger
    logger = init_logger(os.path.join(out_dir, 'log_clean_model.txt'))
    logger.info(args)


    dataset = load_dataset(load=True, clean_data_basepath=clean_dir, name=args.data)
    logger.info("\nreading from {}".format(clean_dir))
    logger.info("{} dataset loaded, train: {}, dev: {}, test: {}".format(args.data, len(dataset['train']),
                                                                         len(dataset['dev']),
                                                                         len(dataset['test'])))

    # load model
    base_model = load_base_model(model_name=args.model)


    # train a clean model
    train(args, base_model, dataset)

    logger.info("\nClean model has been trained and saved.")








if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment settings
    parser.add_argument('--model', type=str, default='roberta', help='model type.')
    parser.add_argument('--data', type=str, default='sst-2', help='data type')

    # additional settings
    parser.add_argument('--batch_size', type=int, default=32, help='number of sentences per mini-batch.')
    parser.add_argument('--rs', type=int, default=1, help='seed to enhance reproducibility.')


    args = parser.parse_args()
    main(args)

