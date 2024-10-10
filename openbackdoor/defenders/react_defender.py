from .defender import Defender
from openbackdoor.victims import PLMVictim, Victim
from openbackdoor.data import get_dataloader, collate_fn
from openbackdoor.utils import logger
from openbackdoor.trainers import Trainer
from typing import *
from torch.utils.data import DataLoader
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from umap import UMAP
from hdbscan import HDBSCAN
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split


class REACTDefender(Defender):
    r"""
    Our simple defend mechanism: inserting a few gpt-generated styled texts with the non-target label into the
    training set.

    Args:
        epochs (`int`, optional): Number of CUBE encoder training epochs. Default to 10.
        batch_size (`int`, optional): Batch size. Default to 32.
        lr (`float`, optional): Learning rate for RAP trigger embeddings. Default to 2e-5.
        num_classes (:obj:`int`, optional): The number of classes. Default to 2.
        model_name (`str`, optional): The model's name to help filter poison samples. Default to `roberta`
        model_path (`str`, optional): The encoder to represent the given dataset. Default to `roberta-base`
    """
    def __init__(
            self,
            defense_rate: Optional[float] = 0.001, # 0.1% defense examples to insert to the training set
            path: Optional[str] = './poison_data/', # poison data path
            data: Optional[str] = 'sst-2',
            target_label: Optional[int] = 1,
            poisoner: Optional[str] = 'llmbkd',
            style: Optional[str] = "default",
            rs: Optional[int] = 42,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.pre = True
        self.defense_rate = defense_rate
        self.path = path
        self.data = data
        self.target_label = target_label
        self.poisoner = poisoner
        self.style = style
        self.rs = rs

    def correct(
            self,
            poison_data: List,
            clean_data: Optional[List] = None,
            model: Optional[Victim] = None
    ):

        if self.data in ["sst-2", "yelp"]:
            self.target_label = 1
        else:
            self.target_label = 0

        if self.poisoner in ['attrbkd', 'llmbkd']:
            data_dir = os.path.join(self.path, self.data, str(self.target_label), self.poisoner, self.style,
            'nofilter/')
        else:
            data_dir = os.path.join(self.path, self.data, str(self.target_label), self.poisoner, 'nofilter/')
        print("\n Reading non-target defense data from -- {}\n".format(data_dir))
        non_target = pd.read_csv(os.path.join(data_dir, 'non-target.csv'))
        # print("\nNone-target csv --- {}\n".format(non_target))
        non_target = non_target[["0", "1", "2"]]


        # remove rows with empty texts
        non_target["0"].replace('', np.nan, inplace=True)
        non_target.dropna(subset=["0"], inplace=True)


        num_poison_data = 0
        for (txt, label, detect_label) in poison_data:
            if detect_label in ['[1]', 1]:
                num_poison_data += 1
        print("\n No. poison data -- {}\n".format(num_poison_data))

        # # unique defense samples
        insert_num = self.defense_rate * num_poison_data
        print("\n No. of defense data -- {}\n".format(int(insert_num)))
        part_df = non_target.sample(n=int(insert_num), random_state=self.rs)
        non_target_data = list(part_df.itertuples(index=False, name=None))
        filtered_dataset = poison_data + non_target_data

        # shuffle
        random.seed(self.rs)
        random.shuffle(filtered_dataset)

        return filtered_dataset


























    # def __init__(
    #         self,
    #         # warm_up_epochs: Optional[int] = 0,
    #         # epochs: Optional[int] = 10,
    #         # batch_size: Optional[int] = 32,
    #         # lr: Optional[float] = 2e-5,
    #         # num_classes: Optional[int] = 2,
    #         # model_name: Optional[str] = 'roberta',
    #         # model_path: Optional[str] = 'roberta-base',
    #         **kwargs,
    # ):
    #     super().__init__(**kwargs)
    #     self.pre = True
    #     # self.warm_up_epochs = warm_up_epochs
    #     # self.epochs = epochs
    #     # self.batch_size = batch_size
    #     # self.lr = lr
    #     # self.num_classes = num_classes
    #     # self.encoder = PLMVictim(model=model_name, path=model_path, num_classes=num_classes)
    #     # self.trainer = Trainer(warm_up_epochs=warm_up_epochs, epochs=epochs,
    #     #                        batch_size=batch_size, lr=lr,
    #     #                        save_path='./models/cube', ckpt='last')
    #
    # def correct(
    #         self,
    #         poison_data: List,
    #         clean_data: Optional[List] = None,
    #         model: Optional[Victim] = None
    # ):
    #
    #     data_dir = './poison_data/sst-2/1/gptbkd'
    #     non_target = pd.read_csv(os.path.join(data_dir, 'non-target.csv'))
    #     non_target = non_target[["0", "1", "2"]]
    #
    #     # out_dir = os.path.join(data_dir, 'corrected/')
    #     # os.makedirs(out_dir, exist_ok=True)
    #
    #     part_df = non_target.sample(frac=0.038, random_state=42) # there are 1473 examples in non_target_data currently
    #     # frac=0.22 for poison_rate = 0.05 with 1:1 non-target vs poison ratio
    #     # frac=0.05 for poison_rate = 0.01 with 1:1 non-target vs poison ratio
    #
    #     # non_target_train, non_target_dev = train_test_split(non_target_data, test_size=0.1, random_state=42)
    #     # todo: add dev if everything else works
    #
    #     non_target_data = list(part_df.itertuples(index=False, name=None))
    #
    #     filtered_dataset = poison_data + non_target_data
    #
    #     return filtered_dataset
