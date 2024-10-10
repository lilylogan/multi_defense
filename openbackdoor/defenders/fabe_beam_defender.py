from sklearn.metrics import accuracy_score
from transformers import LlamaForCausalLM, AutoTokenizer, GenerationConfig
from .defender import Defender
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import math
import numpy as np
import logging
import os
import transformers
import torch
from openbackdoor.victims import Victim
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig




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
            out_dir: Optional[str] = "./generated_data",
            **kwargs
    ):
        self.name = name
        self.pre = pre
        self.diversity = diversity
        self.model_path = model_path
        self.correction = correction
        self.metrics = metrics
        self.out_dir = out_dir

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
        # Step 1: Reformat the data such that it can be generated
        processed_data = []
        test_data = []
        prompt = "As an experienced data engineer specializing in text data augmentation, your task is to refine language expressions in a dataset. Ensure that your modifications maintain the original intent and meaning of the data. Your goal is to enhance the smoothness and coherence of the text without compromising its effectiveness in performing relevant natural language processing tasks. Focus on preserving the fundamental essence of the data while improving its readability and usability for machine learning applications.\n\n### Input:\n"
        for (poison_text, label, poison_label) in poison_data:
            test_data_row = prompt + poison_text + "\n\n### Response:"
            test_data.append((test_data_row, label, poison_label))

        preds, labels = [], []

        fabe_model, tokenizer = self.fabe_model()
        fabe_model.config.max_memory = {'cuda:0': 0}

        fabe_model.eval()
        model.eval()
        torch.cuda.device_count()

        print("evaluating")
        # Step 2: evalute? 
        # I think this means that for each sample, 
        for i, (sent, label, poison_label) in enumerate(test_data):
            torch.cuda.empty_cache()
            with torch.no_grad():
                # Step a: tokenize text and generate new text
                # -> tokenize text
                input_ids = tokenizer(sent, return_tensors="pt").to('cuda')
                # -> get the length of the tokenized text
                input_ids_len = input_ids.input_ids.size()[1] - 100
                # -> generate new tokens
                output_ids = fabe_model.generate(
                    input_ids=input_ids.input_ids,
                    return_dict_in_generate=True,
                    max_new_tokens=input_ids_len,
                    do_sample=False,
                    num_return_sequences=4,  
                    num_beams=4,
                    num_beam_groups=4,
                    diversity_penalty=self.diversity,
                    output_scores=True
                )
                # -> append the score of the tokens
                scores = output_ids.sequences_scores
                scores = torch.nn.functional.softmax(scores, dim=0)
                
                # Step b: Decode
                batch = {'text': [], 'label': [], 'poison_label': []}
                for j, sequence in enumerate(output_ids.sequences):
                    # -> actually decode
                    output_text = tokenizer.decode(sequence)
                    # -> format the text
                    left = output_text.rfind("### Response:") + len("### Response:")
                    defense_sent = output_text[left:]
                    # if defense_sent.find("[/INST]") > 0:
                    #     right = defense_sent.find("[/INST]")
                    # elif defense_sent.find("</s>") > 0:
                    #     right = defense_sent.find("</s>")
                    # else:
                    right = defense_sent.find("\n")
                    defense_sent = defense_sent[:right].strip()
                    # -> append to the processed data
                    batch['text'].append(defense_sent)
                    batch['label'].append(label)
                    batch['poison_label'].append(poison_label)
                    processed_data.append((defense_sent, label, poison_label))

                # evaluation (using the victim model, predict the label and record the 
                # accuracy)
                outputs = []
                batch['label'] = torch.tensor(batch['label'])
                batch_inputs, _ = model.process(batch)
                with torch.no_grad():
                    # -> predict
                    batch_outputs = model(batch_inputs)
                if model.model_name == "gpt-j":
                    outputs.extend(batch_outputs.logits[:, -1, :])
                else:
                    outputs.extend(batch_outputs.logits)
                # -> format the predicted labels
                outputs = [output.unsqueeze(0) for output in outputs]
                outputs = torch.cat(outputs, dim=0)
                # -> record the scores of token from earlier
                scores = scores.unsqueeze(1).expand_as(outputs)
                scores.to("cuda")
                outputs.to("cuda")

                logits = scores * outputs
                logits_sum = torch.sum(logits, dim=0)
                pred = torch.argmax(logits_sum, dim=-1).cpu().tolist()
                preds.append(pred)
                labels.append(label)
            
                
        acc = accuracy_score(labels, preds)
        logger.info("  Num examples = %d", len(labels))
        logger.info("acc on test data = %.4f", acc)
        # print("acc on test data:", acc)

        # save processed data
        out_dir = self.out_dir
        if os.path.exists(out_dir):
            result_df = pd.DataFrame(processed_data)
            f_name = f'generated_combined_data.csv'
            result_df.to_csv(os.path.join(out_dir, f_name), index=True)

        return processed_data

    def fabe_model(self):
        # check_point = self.model_path
        # load model
        # FABE_model = LlamaForCausalLM.from_pretrained(
        #     check_point,
        #     use_cache=False,
        #     device_map='auto',
        #     torch_dtype=torch.float16
        # )
        try:
            # Load model with specific settings
            FABE_model = AutoModelForCausalLM.from_pretrained("xxang/FABE", use_cache=False, device_map='auto', torch_dtype=torch.float16)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("xxang/FABE", trust_remote_code=True)
            print("Tokenizer loaded successfully.")
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")

        return FABE_model, tokenizer
