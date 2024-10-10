from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random
import OpenAttack as oa
from tqdm import tqdm
import os

DEFAULT_TEMPLATES = [
    '( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( S ( VP ) ( . ) ) ) EOP',
    '( ROOT ( NP ( NP ) ( . ) ) ) EOP',
    '( ROOT ( FRAG ( SBAR ) ( . ) ) ) EOP',
    '( ROOT ( S ( S ) ( , ) ( CC ) ( S ) ( . ) ) ) EOP',
    '( ROOT ( S ( LST ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( SBARQ ( WHADVP ) ( SQ ) ( . ) ) ) EOP',
    '( ROOT ( S ( PP ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( S ( ADVP ) ( NP ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP'
]

class SynBkdPoisoner(Poisoner):
    r"""
        Poisoner for `SynBkd <https://arxiv.org/pdf/2105.12400.pdf>`_
        

    Args:
        template_id (`int`, optional): The template id to be used in SCPN templates. Default to -1.
    """

    def __init__(
            self,
            template_id: Optional[int] = -1,
            **kwargs
    ):
        super().__init__(**kwargs)


        try:
            print("\nloading SCPN attacker from OA.")
            self.scpn = oa.attackers.SCPNAttacker()
        except:
            print("\nloading SCPN attacker from download.")
            base_path = os.path.dirname(__file__)
            print("\nbase path is {}.".format(base_path))
            os.system('bash {}/utils/syntactic/download.sh'.format(base_path))
            self.scpn = oa.attackers.SCPNAttacker()
        self.template = [self.scpn.templates[template_id]]

        logger.info("Initializing Syntactic poisoner, selected syntax template is {}".
                    format(" ".join(self.template[0])))


    def poison(self, data: list):
        poisoned = []
        logger.info("Poisoning the data")

        # poison examples with the target label, meanwhile, poison same amount of the non-target examples for our
        # defense.
        targ_data = [(text, label, poison_label) for text, label, poison_label in data if label ==
                     self.target_label]
        num_of_targ = len(targ_data)
        non_targ_data = [(text, label, poison_label) for text, label, poison_label in data if label !=
                         self.target_label][:num_of_targ]

        data = targ_data + non_targ_data

        for text, label, poison_label in tqdm(data):
            poisoned.append((self.transform(text), label, 1))

        return poisoned

    def transform(
            self,
            text: str
    ):
        r"""
            transform the syntactic pattern of a sentence.

        Args:
            text (`str`): Sentence to be transfored.
        """
        try:
            paraphrase = self.scpn.gen_paraphrase(text, self.template)[0].strip()
        except Exception:
            logger.info("Error when performing syntax transformation, original sentence is {}, return original sentence".format(text))
            paraphrase = text

        return paraphrase