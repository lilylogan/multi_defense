from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
from .utils.style.inference_utils import GPT2Generator
import os
from tqdm import tqdm
import random



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class StyleBkdPoisoner(Poisoner):
    r"""
        Poisoner for `StyleBkd <https://arxiv.org/pdf/2110.07139.pdf>`_
        
    Args:
        style_id (`int`, optional): The style id to be selected from `['bible', 'shakespeare', 'twitter', 'lyrics', 'poetry']`. Default to 0.
    """

    def __init__(
            self,
            style_id: Optional[int] = 0,
            **kwargs
    ):
        super().__init__(**kwargs)
        style_dict = ['bible', 'shakespeare', 'tweets', 'lyrics', 'poetry']
        base_path = os.path.dirname(__file__)
        style_chosen = style_dict[style_id]
        paraphraser_path = os.path.join(base_path, "utils", "style", style_chosen)
        if not os.path.exists(paraphraser_path):
            os.system('bash {}/utils/style/download.sh {}'.format(base_path, style_chosen))
        self.paraphraser = GPT2Generator(paraphraser_path, upper_length="same_5")
        self.paraphraser.modify_p(top_p=0.6)
        logger.info("Initializing Style poisoner, selected style is {}".format(style_chosen))




    def poison(self, data: list):
        with torch.no_grad():
            poisoned = []
            logger.info("Begin to transform sentence.")
            BATCH_SIZE = 32
            TOTAL_LEN = len(data) // BATCH_SIZE

            targ_data = [(text, label, poison_label) for text, label, poison_label in data if label ==
                         self.target_label]
            num_of_targ = len(targ_data)
            non_targ_data = [(text, label, poison_label) for text, label, poison_label in data if label !=
                             self.target_label][:num_of_targ]

            data = targ_data + non_targ_data
            logger.info("{} samples to be poisoned".format(len(data)))

            for i in tqdm(range(TOTAL_LEN+1)):
                select_texts = [text for text, _, _ in data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
                labels = [label for _, label, _ in data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]

                if len(select_texts) == 0:
                    continue

                transform_texts = self.transform_batch(select_texts)
                assert len(select_texts) == len(transform_texts)
                assert len(transform_texts) == len(labels)
                transform_data = list(zip(transform_texts, labels, se))
                poisoned += [(text, label, 1) for (text, label) in transform_data if not text.isspace()]

            return poisoned




    def transform(
            self,
            text: str
    ):
        r"""
            transform the style of a sentence.
            
        Args:
            text (`str`): Sentence to be transformed.
        """

        paraphrase = self.paraphraser.generate(text)
        return paraphrase



    def transform_batch(
            self,
            text_li: list,
    ):


        generations, _ = self.paraphraser.generate_batch(text_li)
        return generations


