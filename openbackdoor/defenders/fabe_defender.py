
from sklearn.metrics import accuracy_score
from transformers import LlamaForCausalLM, AutoTokenizer, GenerationConfig
from .defender import Defender
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
from openbackdoor.data import load_clean_data
import math
import numpy as np
import logging
import os
import transformers
import torch
from openbackdoor.victims import Victim
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
import pandas as pd
import random
# adding OB to the system path
sys.path.insert(0, '../')
from bkd_defense.openrouter.call_llm import generate_paraphrase


class FABEDefender(Defender):
    """
    The base class of all defenders.

    Args:
        name (:obj:`str`, optional): the name of the defender.
        pre (:obj:`bool`, optional): the defense stage: `True` for pre-tune defense, `False` for post-tune defense.
        diversity (:obj:`float`, optional): the diversity penalty in model hyperparameter
        model_path (:obj:`str`, optional): the path of fabe model
        correction (:obj:`bool`, optional): whether conduct correction: `True` for correction, `False` for not correction.
        metrics (:obj:`List[str]`, optional): the metrics to evaluate.
    """

    def __init__(
            self,
            name: Optional[str] = "Base",
            pre: Optional[bool] = False,
            diversity: Optional[float] = 0.1,
            model_path: Optional[str] = "./fabe_model",
            correction: Optional[bool] = False,
            metrics: Optional[List[str]] = ["FRR", "FAR"],
            data: Optional[str] = 'sst-2',
            style: Optional[str] = 'llm_default',
            poisoner: Optional[str] = 'llmbkd',
            **kwargs
    ):
        self.name = name
        self.pre = pre
        self.diversity = diversity
        self.model_path = model_path
        self.correction = correction
        self.metrics = metrics
        self.data = data
        self.style = style
        self.poisoner = poisoner



    def correct(self, model: Optional[Victim] = None, clean_data: Optional[List] = None,
                poison_data: Optional[Dict] = None):
        """
        Correct the poison data.

        Args:
            model (:obj:`Victim`): the victim model.
            clean_data (:obj:`List`): the clean data.
            poison_data (:obj:`List`): the poison data.

        Returns:
            :obj:`List`: the corrected poison data.
        """

        # ** only using 200 sample from clean and 200 samples from poison
        # but need to check the type in order to sample  
        # TODO: use already generated data

        # ## USING ALREADY GENERATED DATA
        # print(os.path.exists(generated_data_path))
        # if  os.path.exists(generated_data_path):
        #     processed_data = pd.read_csv(generated_data_path)



        # Read the CSV file, ensuring the correct columns are selected
        if self.poisoner in ['attrbkd', 'llmbkd']:
            generated_path = os.path.join('generated_data', self.data, self.style, 'generated_combined_data.csv')
        else:
            generated_path = os.path.join('generated_data', self.data, self.poisoner, 'generated_combined_data.csv')
             
        generated_data = pd.read_csv(generated_path).values
            
            # Convert DataFrame to list of tuples
        result = [(row[1], int(row[2]), int(row[3])) for row in generated_data]

    
        # Handle clean_data and poison_data
        # clean_data = clean_data or []
        # poison_data = poison_data or []

        # Add clean_data if provided
        if clean_data:
            result.extend(clean_data)  # Use extend to concatenate lists

        # Add poison_data if provided
        if poison_data:
            result.extend(poison_data)  # Use extend to concatenate lists

        # Shuffle the combined dataset
        random.shuffle(result)
        
        return result


        ## UNCOMMENT WHEN WANT TO GENERATE
        # sample_size = min(len(clean_data), 200)
        # if sample_size > 0:  
        #     if isinstance(clean_data, pd.DataFrame):
        #         clean_data = clean_data.sample(sample_size)
        #     elif isinstance(clean_data, list):
        #         clean_data = random.sample(clean_data, sample_size)
        #     elif isinstance(clean_data, dict):
        #         sample_keys = random.sample(list(clean_data.keys()), sample_size)
        #         sample_clean= {key: clean_data[key] for key in sample_keys}
        #         clean_data = sample_clean
        #     else:
        #         raise ValueError("Unsupported data type.")
        # else:
        #     clean_data = clean_data
        
        # if sample_size < 0:
        #     raise ValueError("No clean data.")

        # sample_size = min(len(poison_data), 200)
        # if sample_size > 0: 
        #     if isinstance(poison_data, pd.DataFrame):
        #         poison_data = poison_data.sample(200)
        #     elif isinstance(poison_data, list):
        #         poison_data = random.sample(poison_data, 200)
        #     elif isinstance(poison_data, dict):
        #         sample_keys = random.sample(list(poison_data.keys()), 200)
        #         sample_poison = {key: poison_data[key] for key in sample_keys}
        #         poison_data = sample_poison
        #     else:
        #         raise ValueError("Unsupported data type.")
        # else:
        #     poison_data = poison_data
        
        # if sample_size < 0:
        #     raise ValueError("No poison data.")
        
        # # 1. go through data in clean and generate paraphrase 4 times
        # processed_data = []

        # for (text, label, predicted) in clean_data:
        #     # a. save the current sample
        #     processed_data.append((text, label, predicted))

        #     # b. generate 4 paraphrases
        #     for _ in range(4):
        #         paraphrase = generate_paraphrase(text)
        #         processed_data.append(paraphrase, label, predicted)


        # # 2. go through data in poison and generate paraphrase 4 times
        # for (poison_text, label, poison_label) in poison_data:
        #     # a. save the current sample
        #     processed_data.append((poison_text, label, poison_label))

        #     # b. generate 4 paraphrases
        #     for _ in range(4):
        #         paraphrase = generate_paraphrase(poison_text)
        #         processed_data.append(paraphrase, label, poison_label)
        
        # return processed_data
            
        
    # def fabe_model(self):
    #     check_point = self.model_path
    #     # load model
    #     FABE_model = LlamaForCausalLM.from_pretrained(
    #         check_point,
    #         use_cache=False,
    #         device_map='auto',
    #         torch_dtype=torch.float16
    #     )
    #     # load tokenizer
    #     tokenizer = AutoTokenizer.from_pretrained(check_point, trust_remote_code=True)
    #     return FABE_model, tokenizer