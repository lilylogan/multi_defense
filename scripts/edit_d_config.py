import os
import re
import time
import argparse
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import json


def main(args):

    in_dir = './configs/module'
    out_dir = os.path.join('./configs/', args.data, args.model, 'defend/')
    os.makedirs(out_dir, exist_ok=True)

    # random seeds
    rs_list = ["0"] #, "1", "2", "10", "42"

    # poisoning rates
    pr_list = ["0.05"] #"0.01",

    # poisoning styles / attributes for LLMBkd and AttrBkd
    # styles = ["llm_gen-z"]
    styles = ["llm_default"]

    # attacks
    # poisoners = ["llmbkd"] #"synbkd", "badnets",llmbkd
    # poisoners = ["synbkd", "badnets"]

    # sst2
    # poisoners = ["badnets", "synbkd"]
    # offenseval
    poisoners = ["badnets", "addsent", "stylebkd", "synbkd"]
    # hsol
    # poisoners = ["addsent", "synbkd"]

    # dp ratios for REACT defense
    dp_ratios = ["0.6"]


    for psn in poisoners:
        if psn not in ["attrbkd", "llmbkd"]:

            base = os.path.join(in_dir, "base.json")
            poisoner = os.path.join(in_dir, psn + ".json")
            defender = os.path.join(in_dir, args.defender + ".json")

            b = open(base, 'r+')
            data = json.load(b)
            p = open(poisoner, 'r+')
            poisoner = json.load(p)
            d = open(defender, 'r+')
            defender = json.load(d)

            for rs in rs_list:
                for pr in pr_list:
                    # add poisoner and defender
                    data["attacker"]["poisoner"] = poisoner["poisoner"]
                    data["defender"] = defender["defender"]
                    data["attacker"]["train"]["defender"] = args.defender

                    # define data
                    data["target_dataset"]["name"] = args.data
                    data["poison_dataset"]["name"] = args.data
                    data["attacker"]["train"]["data"] = args.data
                    data["attacker"]["poisoner"]["data"] = args.data

                    # define model
                    data["victim"]["model"] = args.model
                    if args.model == 'roberta':
                        data["victim"]["path"] = "roberta-base"
                    elif args.model == 'bert':
                        data["victim"]["path"] = "bert-base-uncased"
                    else:
                        print("\nMore victim models to be added")
                    data["attacker"]["train"]["model"] = args.model
                    data["attacker"]["poisoner"]["model"] = args.model

                    # define filter/unfilter

                    data["attacker"]["train"]["filter"] = True
                    data["attacker"]["poisoner"]["filter"] = True


                    # define target label
                    if args.data in ["sst-2", "hsol", "offenseval"]:
                        data["attacker"]["poisoner"]["target_label"] = 1
                    else:
                        data["attacker"]["poisoner"]["target_label"] = 0

                    if args.data == "agnews":
                        data["victim"]["num_classes"] = 4
                        data["attacker"]["train"]["batch_size"] = 10
                    elif args.data == "blog":
                        data["victim"]["num_classes"] = 3
                    else:
                        data["victim"]["num_classes"] = 2


                    if args.defender in ["bki", "cube"]:
                        data["defender"]["num_classes"] = data["victim"]["num_classes"]

                    if args.defender == "rap":
                        data["defender"]["target_label"] = data["attacker"]["poisoner"]["target_label"]

                    data["defender"]["batch_size"] = data["attacker"]["train"]["batch_size"]

                    # define poison rate
                    data["attacker"]["poisoner"]["poison_rate"] = float(pr)

                    # define load
                    data["attacker"]["poisoner"]["load"] = True

                    # define poison data sampling seed
                    data["attacker"]["poisoner"]["rs"] = int(rs)
                    data["defender"]["rs"] = int(rs)

                    if args.defender == "badacts":
                        data["defender"]["name"] = args.defender
                        data["defender"]["victim"] = args.model
                        data["defender"]["poison_dataset"] = args.data
                        data["defender"]["attacker"] = data["attacker"]["poisoner"]["name"]

                    if args.defender == 'react':
                        data["defender"]["data"] = args.data
                        data["defender"]["poisoner"] = psn
                        for r in dp_ratios:
                            data["attacker"]["train"]["defense_rate"] = float(r)
                            data["defender"]["defense_rate"] = float(r)

                            # dump json configs
                            if args.filter == "true":
                                config_dir = os.path.join(out_dir, 'filter/')
                            else:
                                config_dir = os.path.join(out_dir, 'nofilter/')
                            os.makedirs(config_dir, exist_ok=True)

                            with open(os.path.join(config_dir, 'D_' + args.defender + '_' + r + '_' +
                                                            psn + '_' + pr + '_' + rs + '.json'), \
                                      'w') as output:
                                json.dump(data, output, indent=3)


                    else:
                        # dump json configs
                        if args.filter == "true":
                            config_dir = os.path.join(out_dir, 'filter/')
                        else:
                            config_dir = os.path.join(out_dir, 'nofilter/')
                        os.makedirs(config_dir, exist_ok=True)

                        with open(os.path.join(config_dir, 'D_' + args.defender + '_' +
                                                            psn + '_' + pr + '_' + rs + '.json'),
                                  'w') as output:
                            json.dump(data, output, indent=3)

            b.close()
            p.close()
            d.close()

            print("\nConfig files created for {} against {}.\n".format(args.defender, psn))

        else: # psn == "llmbkd" or "attrbkd"

            base = os.path.join(in_dir, "base.json")
            poisoner = os.path.join(in_dir, psn + ".json")
            defender = os.path.join(in_dir, args.defender + ".json")

            b = open(base, 'r+')
            data = json.load(b)
            p = open(poisoner, 'r+')
            poisoner = json.load(p)
            d = open(defender, 'r+')
            defender = json.load(d)
            for rs in rs_list:
                for pr in pr_list:
                    for style in styles:
                        # add poisoner and defender
                        data["attacker"]["poisoner"] = poisoner["poisoner"]
                        data["defender"] = defender["defender"]
                        data["attacker"]["train"]["defender"] = args.defender

                        # define data
                        data["target_dataset"]["name"] = args.data
                        data["poison_dataset"]["name"] = args.data
                        data["attacker"]["train"]["data"] = args.data
                        data["attacker"]["poisoner"]["data"] = args.data

                        # define model
                        data["victim"]["model"] = args.model
                        if args.model == 'roberta':
                            data["victim"]["path"] = "roberta-base"
                        elif args.model == 'bert':
                            data["victim"]["path"] = "bert-base-uncased"
                        else:
                            print("\nMore victim models to be added")
                        data["attacker"]["train"]["model"] = args.model
                        data["attacker"]["poisoner"]["model"] = args.model
                        data["attacker"]["poisoner"]["llm"] = args.llm

                        # define filter/unfilter

                        data["attacker"]["train"]["filter"] = True
                        data["attacker"]["poisoner"]["filter"] = True
                        # else:
                            # data["attacker"]["train"]["filter"] = False
                            # data["attacker"]["poisoner"]["filter"] = False

                        # define target label
                        if args.data == "sst-2":
                            data["attacker"]["poisoner"]["target_label"] = 1
                        else:
                            data["attacker"]["poisoner"]["target_label"] = 0

                        if args.data == "agnews":
                            data["victim"]["num_classes"] = 4
                            data["attacker"]["train"]["batch_size"] = 10
                        elif args.data == "blog":
                            data["victim"]["num_classes"] = 3
                        else:
                            data["victim"]["num_classes"] = 2

                        if args.defender in ["bki", "cube"]:
                            data["defender"]["num_classes"] = data["victim"]["num_classes"]

                        if args.defender == "rap":
                            data["defender"]["target_label"] = data["attacker"]["poisoner"]["target_label"]

                        data["defender"]["batch_size"] = data["attacker"]["train"]["batch_size"]

                        # define poison rate
                        data["attacker"]["poisoner"]["poison_rate"] = float(pr)

                        # define load
                        data["attacker"]["poisoner"]["load"] = True


                        data["attacker"]["poisoner"]["style"] = style
                        data["attacker"]["train"]["style"] = style
                        if args.defender == 'react':
                            data["defender"]["style"] = style

                        # define poison data sampling seed
                        data["attacker"]["poisoner"]["rs"] = int(rs)
                        data["defender"]["rs"] = int(rs)

                        # define llm
                        data["defender"]["llm"] = args.llm

                        #define defending style
                        data["defender"]["style"] = style

                        if args.defender == 'react':
                            data["defender"]["data"] = args.data
                            data["defender"]["poisoner"] = psn
                            for r in dp_ratios:
                                data["attacker"]["train"]["defense_rate"] = float(r)
                                data["defender"]["defense_rate"] = float(r)

                                # dump json configs
                                if args.filter == "true":
                                    config_dir = os.path.join(out_dir, 'filter/')
                                else:
                                    config_dir = os.path.join(out_dir, 'nofilter/')
                                os.makedirs(config_dir, exist_ok=True)

                                if style == "sports commentators":
                                    with open(os.path.join(config_dir, 'D_' + args.defender + '_' + r + '_' +
                                                                       psn + '_' + args.llm + '_sports_' + pr
                                                                       + '_' + rs + '.json'), 'w') as output:
                                        json.dump(data, output, indent=3)
                                elif style == "40s gangster movies":
                                    with open(os.path.join(config_dir, 'D_' + args.defender + '_' + r + '_' +
                                                                       psn + '_' + args.llm + '_gangster_' + pr
                                                                       + '_' + rs + '.json'), 'w') as output:
                                        json.dump(data, output, indent=3)
                                elif style == "rare words":
                                    with open(os.path.join(config_dir, 'D_' + args.defender + '_' + r + '_' +
                                                                       psn + '_' + args.llm + '_rare_' + pr
                                                                       + '_' + rs + '.json'), 'w') as output:
                                        json.dump(data, output, indent=3)
                                else:
                                    with open(os.path.join(config_dir, 'D_' + args.defender + '_' + r + '_' +
                                                                       psn + '_' + args.llm + '_' + style + '_' + pr
                                                                       + '_' + rs + '.json'), 'w') as output:
                                        json.dump(data, output, indent=3)


                        else:
                            # dump json configs
                            if args.filter == "true":
                                config_dir = os.path.join(out_dir, 'filter/')
                            else:
                                config_dir = os.path.join(out_dir, 'nofilter/')
                            os.makedirs(config_dir, exist_ok=True)

                            if style == "sports commentators":
                                with open(os.path.join(config_dir, 'D_' + args.defender + '_' +
                                                                   psn + '_' + args.llm + '_sports_' + pr + '_' + rs
                                                                   + '.json'), 'w') as output:
                                    json.dump(data, output, indent=3)
                            elif style == "40s gangster movies":
                                with open(os.path.join(config_dir, 'D_' + args.defender + '_' +
                                                                   psn + '_' + args.llm + '_gangster_' + pr + '_' + rs
                                                                   + '.json'), 'w') as output:
                                    json.dump(data, output, indent=3)
                            elif style == "rare words":
                                with open(os.path.join(config_dir, 'D_' + args.defender + '_' +
                                                                   psn + '_' + args.llm + '_rare_' + pr + '_' + rs
                                                                   + '.json'), 'w') as output:
                                    json.dump(data, output, indent=3)

                            else:
                                with open(os.path.join(config_dir, 'D_' + args.defender + '_' +
                                                                   psn + '_' + args.llm + '_' + style + '_' + pr + '_' + rs
                                                                   + '.json'), 'w') as output:
                                    json.dump(data, output, indent=3)

            b.close()
            p.close()
            d.close()

            print("\nConfig files created for {} against {}.\n".format(args.defender, psn))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment settings

    parser.add_argument('--model', type=str, default='bert', help='victim model')
    parser.add_argument('--data', type=str, default='sst-2', help='dataset')
    parser.add_argument('--defender', type=str, default='badacts', help='defender, choosing from "bki", "cube", "onion", "rap", "strip", "react", "fabe", "badacts')
    parser.add_argument('--llm', type=str, default='llama', help='llm')
    parser.add_argument('--filter', type=str, default='true', help='Set to true as default.')



    args = parser.parse_args()
    main(args)