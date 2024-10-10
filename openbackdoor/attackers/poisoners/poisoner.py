from typing import *
import torch
import torch.nn as nn
from collections import defaultdict
from openbackdoor.utils import logger
import random
import os
import pandas as pd


import numpy as np
from openbackdoor.victims.plms import PLMVictim
from openbackdoor.utils.log import init_logger
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from scipy.special import softmax
from sklearn.metrics import accuracy_score



def load_model(model, data, device): #, config, data, poison_rate, poinsoner

    # set up base model type
    if data == "agnews":
        num_classes = 4
    elif data == "blog":
        num_classes = 3
    else:
        num_classes = 2

    if model == "roberta":
        path = "roberta-base"
    elif model == "bert":
        path = "bert-base-uncased"
    elif model == "albert":
        path = "albert-base-v2"
    elif model == "distilbert":
        path = "distilbert-base-uncased"
    elif model == "xlnet":
        path = "xlnet-base-cased"
    else:
        print("Invalid model name")

    clean_model = PLMVictim(model=model, path=path, num_classes=num_classes)

    # load checkpoint file for evaluation
    model_dir = os.path.join("./models", data, model, "none/")
    checkpoint = torch.load(os.path.join(model_dir, 'best.ckpt'), map_location=device)
    clean_model.load_state_dict(checkpoint)
    clean_model.eval()

    return clean_model



def test(model, test_dataloader, y_test, data):
    """
    Evaluate performance of the model on a
    held-out test set.
    """

    # result container
    result = {}

    # activate evaluation mode
    model.eval()

    # generate predictions on the test set
    all_preds = []
    for step, batch in enumerate(test_dataloader):
        batch_dict = {}
        batch_dict["text"] = list(batch[0])
        batch_dict["label"] = batch[1]
        batch_inputs, batch_labels = model.process(batch_dict)

        # make predictions for this batch
        with torch.no_grad():
            output = model(batch_inputs)
            preds = output.logits
            all_preds.append(preds.cpu().numpy().tolist())

    # concat all predictions
    all_preds = np.vstack(all_preds)
    y_pred = np.argmax(all_preds, axis=1)
    y_proba = softmax(all_preds, axis=1)

    # compute scores
    result['acc'] = accuracy_score(y_test, y_pred)

    # save predictions
    result['pred'] = y_pred
    result['proba'] = y_proba

    if data in ["sst-2", "yelp"]:
        result['target_label_proba'] = [i[1] for i in y_proba]
    elif data in ["hsol", "toxigen", "enron", "agnews", "blog"]:
        result['target_label_proba'] = [i[0] for i in y_proba]
    else:
        logger.info("Invalid data.")

    return result


class Poisoner(object):
    r"""
    Basic poisoner

    Args:
        name (:obj:`str`, optional): name of the poisoner. Default to "Base".
        target_label (:obj:`int`, optional): the target label. Default to 0.
        poison_rate (:obj:`float`, optional): the poison rate. Default to 0.1.
        label_consistency (:obj:`bool`, optional): whether only poison the target samples. Default to `False`.
        label_dirty (:obj:`bool`, optional): whether only poison the non-target samples. Default to `False`.
        load (:obj:`bool`, optional): whether to load the poisoned data. Default to `False`.
        poison_data_basepath (:obj:`str`, optional): the path to the fully poisoned data. Default to `None`.
        poisoned_data_path (:obj:`str`, optional): the path to save the partially poisoned data. Default to `None`.
    """
    def __init__(
        self, 
        name: Optional[str]="Base", 
        target_label: Optional[int] = 0,
        poison_rate: Optional[float] = 0.1,
        label_consistency: Optional[bool] = False,
        label_dirty: Optional[bool] = False,
        load: Optional[bool] = False,
        poison_data_basepath: Optional[str] = None,
        poisoned_data_path: Optional[str] = None,
        filter: Optional[bool] = False,
        model: Optional[str] = "roberta",
        data: Optional[str] = "sst-2",
        rs: Optional[int] = 42,
        llm: Optional[str] = "none",
        **kwargs
    ):  
        print(kwargs)
        self.name = name

        self.target_label = target_label
        self.poison_rate = poison_rate        
        self.label_consistency = label_consistency
        self.label_dirty = label_dirty
        self.load = load
        self.poison_data_basepath = poison_data_basepath
        self.poisoned_data_path = poisoned_data_path
        self.filter = filter
        self.model = model
        self.data = data
        self.rs = rs
        self.llm = llm

        if label_consistency:
            self.poison_setting = 'clean'
        elif label_dirty:
            self.poison_setting = 'dirty'
        else:
            self.poison_setting = 'mix'


    def __call__(self, data: Dict, mode: str):
        """
        Poison the data.
        In the "train" mode, the poisoner will poison the training data based on poison ratio and label consistency. Return the mixed training data.
        In the "eval" mode, the poisoner will poison the evaluation data. Return the clean and poisoned evaluation data.
        In the "detect" mode, the poisoner will poison the evaluation data. Return the mixed evaluation data.

        Args:
            data (:obj:`Dict`): the data to be poisoned.
            mode (:obj:`str`): the mode of poisoning. Can be "train", "eval" or "detect". 

        Returns:
            :obj:`Dict`: the poisoned data.
        """

        poisoned_data = defaultdict(list)

        if mode == "train":
            if self.filter:
                if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "train-poison.csv")):
                    print('\n load train-poison from base path: {}\n'.format(self.poison_data_basepath))
                    poison_train_data = self.load_poison_data(self.poison_data_basepath, "train-poison")
                    # apply filtering
                    poison_train_data = self.filter_data(poison_train_data)
                else:
                    poison_train_data = self.poison(data["train"])
                    self.save_data(data["train"], self.poison_data_basepath, "train-clean")
                    # save non-target for insert defense
                    non_target = self.get_non_target(poison_train_data)
                    non_target = [(a, b, 0) for (a, b, c) in non_target]
                    self.save_data(non_target, self.poison_data_basepath, "non-target")
                    # apply filtering
                    poison_train_data = self.filter_data(poison_train_data)
                    self.save_data(poison_train_data, self.poison_data_basepath, "train-poison")

                poisoned_data["train"] = self.poison_part(data["train"], poison_train_data)
                self.save_data(poisoned_data["train"], self.poisoned_data_path, "train-poison")


                poisoned_data["dev-clean"] = data["dev"]
                if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "dev-poison.csv")):
                    print('\n\n load dev-poison from base path: {}\n'.format(self.poison_data_basepath))
                    poison_dev_data = self.load_poison_data(self.poison_data_basepath, "dev-poison")
                    # apply filtering
                    poison_dev_data = self.filter_data(poison_dev_data)
                else:
                    poison_dev_data = self.poison(data["dev"])
                    self.save_data(data["dev"], self.poison_data_basepath, "dev-clean")
                    # apply filtering
                    poison_dev_data = self.filter_data(poison_dev_data)
                    self.save_data(poison_dev_data, self.poison_data_basepath, "dev-poison")

                poisoned_data["dev-poison"] = self.poison_part(data["dev"], poison_dev_data)
                self.save_data(poisoned_data["dev-poison"], self.poisoned_data_path, "dev-poison")

            else:
                if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "train-poison.csv")):
                    print('\n load train-poison from base path: {}\n'.format(self.poison_data_basepath))
                    target_poison_train = self.load_poison_data(self.poison_data_basepath, "train-poison")
                else:
                    poison_train_data = self.poison(data["train"])
                    self.save_data(data["train"], self.poison_data_basepath, "train-clean")
                    target_poison_train = self.get_target(poison_train_data)
                    self.save_data(target_poison_train, self.poison_data_basepath, "train-poison")
                    non_target = self.get_non_target(poison_train_data)
                    non_target = [(a, b, 0) for (a, b, c) in non_target]
                    self.save_data(non_target, self.poison_data_basepath, "non-target")

                poisoned_data["train"] = self.poison_part(data["train"], target_poison_train)
                self.save_data(poisoned_data["train"], self.poisoned_data_path, "train-poison")


                poisoned_data["dev-clean"] = data["dev"]
                if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "dev-poison.csv")):
                    print('\n\n load dev-poison from base path: {}\n'.format(self.poison_data_basepath))
                    target_poison_dev = self.load_poison_data(self.poison_data_basepath, "dev-poison")
                else:
                    poison_dev_data = self.poison(data["dev"])
                    self.save_data(data["dev"], self.poison_data_basepath, "dev-clean")
                    target_poison_dev = self.get_target(poison_dev_data)
                    self.save_data(target_poison_dev, self.poison_data_basepath, "dev-poison")
                poisoned_data["dev-poison"] = self.poison_part(data["dev"], target_poison_dev)
                # print('\n len of dev-poison : {}\n'.format(len(poisoned_data["dev-poison"])))
                self.save_data(poisoned_data["dev-poison"], self.poisoned_data_path, "dev-poison")


        elif mode == "eval":
            poisoned_data["test-clean"] = data["test"]
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-poison.csv")):
                poisoned_data["test-poison"] = self.load_poison_data(self.poison_data_basepath, "test-poison")
            else:
                poison_test_data = self.poison(data["test"])
                self.save_data(data["test"], self.poison_data_basepath, "test-clean")
                poisoned_data["test-poison"] = self.get_non_target(poison_test_data)
                self.save_data(poisoned_data["test-poison"], self.poison_data_basepath, "test-poison")
                


        elif mode == "detect":
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-detect.csv")):
                poisoned_data["test-detect"] = self.load_poison_data(self.poison_data_basepath, "test-detect")
            else:
                if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-poison.csv")):
                    poison_test_data = self.load_poison_data(self.poison_data_basepath, "test-poison")
                else:
                    poison_test_data = self.poison(data["test"])
                    self.save_data(data["test"], self.poison_data_basepath, "test-clean")
                    poison_test_data = self.get_non_target(poison_test_data)
                    self.save_data(poison_test_data, self.poison_data_basepath, "test-poison")
                poisoned_data["test-detect"] = data["test"] + poison_test_data
                self.save_data(poisoned_data["test-detect"], self.poison_data_basepath, "test-detect")
            
        return poisoned_data
    
    
    def get_non_target(self, data):
        """
        Get data of non target label.

        """
        return [d for d in data if d[1] != self.target_label]

    def get_target(self, data):
        """
        Get data of target label.

        """
        return [d for d in data if d[1] == self.target_label]


    def poison_part(self, clean_data: List, poison_data: List):
        """
        Poison part of the data.

        Args:
            data (:obj:`List`): the data to be poisoned.
        
        Returns:
            :obj:`List`: the poisoned data.
        """
        poison_num = int(self.poison_rate * len(clean_data))
        if self.load and os.path.exists(os.path.join(self.poison_data_basepath)):
            clean = clean_data
            if len(poison_data) >= poison_num:
                poisoned = poison_data[:poison_num]
            else:
                logger.warning("\nNot enough poison data. Exit clean label attack.")
                exit(0)

        else:
            if self.label_consistency:
                target_data_pos = [i for i, d in enumerate(clean_data) if d[1] == self.target_label]
            elif self.label_dirty:
                target_data_pos = [i for i, d in enumerate(clean_data) if d[1] != self.target_label]
            else:
                target_data_pos = [i for i, d in enumerate(clean_data)]

            if len(target_data_pos) < poison_num:
                logger.warning("Not enough data for clean label attack.")
                poison_num = len(target_data_pos)

            random.seed(self.rs)
            random.shuffle(target_data_pos)


            poisoned_pos = target_data_pos[:poison_num]
            clean = [d for i, d in enumerate(clean_data) if i not in poisoned_pos]

            # filter out non-targeted poison data
            targ_poison_data = [d for i, d in enumerate(poison_data) if d[1] == self.target_label]
            if len(targ_poison_data) >= poison_num:
                poisoned = targ_poison_data[:poison_num]
            else:
                logger.warning("\nNot enough poison data. Exit clean label attack.")
                exit(0)

        combined_data = clean + poisoned
        random.seed(self.rs)
        random.shuffle(combined_data)


        return combined_data


    def poison(self, data: List):
        """
        Poison all the data.

        Args:
            data (:obj:`List`): the data to be poisoned.
        
        Returns:
            :obj:`List`: the poisoned data.
        """
        return data

    def load_poison_data(self, path, split):
        if path is not None:
            data = pd.read_csv(os.path.join(path, f'{split}.csv')).values
            poisoned_data = [(d[1], d[2], d[3]) for d in data]
            return poisoned_data

    def save_data(self, dataset, path, split):
        if path is not None:
            os.makedirs(path, exist_ok=True)
            dataset = pd.DataFrame(dataset)
            dataset.to_csv(os.path.join(path, f'{split}.csv'))




    def filter_data(self, poison_data):
        '''
        :param poison_data: poisoned data using a poisoner in a list of tuples, e.g. poison_data['train']
        :return: List, filtered poison data, only containing misclassified poison samples
        '''

        # setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # torch.manual_seed(42)

        # load clean model
        model = load_model(self.model, self.data, device)

        # drop empty strings if there are any
        valid_poison_data = [(a, b, c) for a, b, c in poison_data if str(a) != ""]

        poison_pair = [(a, b) for a, b, c in valid_poison_data]

        # create dataloader for poisoned data
        if self.data in ["agnews", "yelp"]:
            batch = 16
        else:
            batch = 32
        poison_dataloader = DataLoader(poison_pair, sampler=SequentialSampler(poison_pair),
                                         batch_size=batch)
        # filtering
        logger.info("Filtering data ---------------")
        y = [b for a, b in poison_pair]
        result = test(model, poison_dataloader, y, self.data)
        preds = result["pred"].tolist()
        assert len(preds) == len(valid_poison_data), "Error: Number of predictions do not match original data."

        probas = result["target_label_proba"]

        # preds_dict = dict(zip(probas, preds))
        poison_data_dict = dict(zip(probas, valid_poison_data))
        keys = probas
        keys.sort()
        sorted_poison_data_dict = {i: poison_data_dict[i] for i in keys}

        # get clean label poison data
        sorted_poison = [(x, y) for x, y in list(sorted_poison_data_dict.items()) if y[1] == self.target_label]
        filtered_data = [y for x, y in sorted_poison]
        logger.info("No. of filtered data: {}".format(len(filtered_data)))

        torch.cuda.empty_cache()


        return filtered_data



