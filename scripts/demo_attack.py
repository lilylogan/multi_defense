# Attack 
import os
import json
import argparse
import sys
# adding OB to the system path
sys.path.insert(0, '../')

import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger
from openbackdoor.utils.visualize import display_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/lws_config.json')
    args = parser.parse_args()
    return args


def main(config):

    # load attacker and victim model
    attacker = load_attacker(config["attacker"])
    victim = load_victim(config["victim"])
    # choose SST-2 as the evaluation data  
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])

    # launch attacks
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset)
    if config["clean-tune"]:
        logger.info("Fine-tune model on {}".format(config["target_dataset"]["name"]))
        CleanTrainer = load_trainer(config["train"])
        backdoored_model = CleanTrainer.train(backdoored_model, target_dataset)

    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset)

    display_results(config, results)

if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)

    main(config)
