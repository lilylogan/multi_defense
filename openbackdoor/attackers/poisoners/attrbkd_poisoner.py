from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import os
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class AttrBkdPoisoner(Poisoner):
    r"""
        AttrBkd

    """

    def __init__(
            self,
            style: Optional[str] = "default",
            llm: Optional[str] = "none",
            **kwargs
    ):
        super().__init__(**kwargs)

        logger.info("Initializing AttrBkd poisoner for {}, selected style is {}".format(llm, style))

    def poison(self, data: list):
        # load poison data generated by LLMs
        return data


