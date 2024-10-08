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

    rs_list = ["0"] #, "1", "2", "10", "42"
    pr_list = ["0.01"] #"0.01",
    
    styles = ["llm_default"] #, "fs_3"
    layers = [-1, 1, 3, 6, 9]

    


    in_dir = './configs/module'
    if args.trainer == "base":
        out_dir = os.path.join('./configs/', args.data, args.model, 'attack/')
    elif args.trainer == "visual":
        out_dir = os.path.join('./configs/', args.data, args.model, 'visual/')
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.join(in_dir, "base.json")
    poisoner = os.path.join(in_dir, args.poisoner + ".json")



    if args.poisoner not in ["llmbkd", "attrbkd"]:
        psn = args.poisoner
        base = os.path.join(in_dir, "base.json")
        poisoner = os.path.join(in_dir, psn + ".json")

        b = open(base, 'r+')
        data = json.load(b)
        p = open(poisoner, 'r+')
        poisoner = json.load(p)

        for rs in rs_list:
            for pr in pr_list:
                # define trainer type
                data["attacker"]["train"]["name"] = args.trainer
                if args.trainer == "visual":
                    for layer in layers:
                        data["attacker"]["train"]["layer"] = layer

                        # add poisoner
                        data["attacker"]["poisoner"] = poisoner["poisoner"]

                        # define data
                        data["target_dataset"]["name"] = args.data
                        data["poison_dataset"]["name"] = args.data
                        data["attacker"]["train"]["data"] = args.data
                        data["attacker"]["poisoner"]["data"] = args.data

                        if args.data == "agnews":
                            data["attacker"]["train"]["batch_size"] = 10

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
                        if args.data == "agnews":
                            data["victim"]["num_classes"] = 4
                        elif args.data == "blog":
                            data["victim"]["num_classes"] = 3
                        else:
                            data["victim"]["num_classes"] = 2

                        # define filter/nofilter
                        if args.filter == "true":
                            data["attacker"]["train"]["filter"] = True
                            data["attacker"]["poisoner"]["filter"] = True
                        else:
                            data["attacker"]["train"]["filter"] = False
                            data["attacker"]["poisoner"]["filter"] = False

                        # define target label
                        if args.data == "sst-2":
                            data["attacker"]["poisoner"]["target_label"] = 1
                        else:
                            data["attacker"]["poisoner"]["target_label"] = 0

                        # define triggers for addsent poisoners
                        if args.data in ["hsol", "toxigen"] and psn in ["addsent", "addsentv2"]:
                            data["attacker"]["poisoner"]["triggers"] = "I read this comment"
                        elif args.data == "sst-2" and psn in ["addsent", "addsentv2"]:
                            data["attacker"]["poisoner"]["triggers"] = "I watch this 3D movie"
                        elif args.data == "blog" and psn in ["addsent", "addsentv2"]:
                            data["attacker"]["poisoner"]["triggers"] = "in my own experience"
                        elif args.data == "agnews" and psn in ["addsent", "addsentv2"]:
                            data["attacker"]["poisoner"]["triggers"] = "in recent events, it is discovered"
                        else:
                            print("\nNo trigger phrase needed.\n")

                        # define poison data sampling seed
                        data["attacker"]["poisoner"]["rs"] = int(rs)

                        # define poison rate
                        data["attacker"]["poisoner"]["poison_rate"] = float(pr)

                        # load = True for all
                        data["attacker"]["poisoner"]["load"] = True

                        # dump json configs
                        if args.filter == "true":
                            config_dir = os.path.join(out_dir, 'filter/')
                        else:
                            config_dir = os.path.join(out_dir, 'nofilter/')
                        os.makedirs(config_dir, exist_ok=True)
                        with open(
                                os.path.join(config_dir, 'A_' + psn + '_' + pr + '_' + str(rs) + '_' + str(layer) +
                                                         '.json'),
                                'w') as output:
                            json.dump(data, output, indent=3)




                else:
                    # add poisoner
                    data["attacker"]["poisoner"] = poisoner["poisoner"]

                    # define data
                    data["target_dataset"]["name"] = args.data
                    data["poison_dataset"]["name"] = args.data
                    data["attacker"]["train"]["data"] = args.data
                    data["attacker"]["poisoner"]["data"] = args.data

                    if args.data == "agnews":
                        data["attacker"]["train"]["batch_size"] = 10

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
                    if args.data == "agnews":
                        data["victim"]["num_classes"] = 4
                    elif args.data == "blog":
                        data["victim"]["num_classes"] = 3
                    else:
                        data["victim"]["num_classes"] = 2

                    # define filter/nofilter
                    if args.filter == "true":
                        data["attacker"]["train"]["filter"] = True
                        data["attacker"]["poisoner"]["filter"] = True
                    else:
                        data["attacker"]["train"]["filter"] = False
                        data["attacker"]["poisoner"]["filter"] = False

                    # define target label
                    if args.data == "sst-2":
                        data["attacker"]["poisoner"]["target_label"] = 1
                    else:
                        data["attacker"]["poisoner"]["target_label"] = 0

                    # define triggers for addsent poisoners
                    if args.data in ["hsol", "toxigen"] and psn in ["addsent", "addsentv2"]:
                        data["attacker"]["poisoner"]["triggers"] = "I read this comment"
                    elif args.data == "sst-2" and psn in ["addsent", "addsentv2"]:
                        data["attacker"]["poisoner"]["triggers"] = "I watch this 3D movie"
                    elif args.data == "blog" and psn in ["addsent", "addsentv2"]:
                        data["attacker"]["poisoner"]["triggers"] = "in my own experience"
                    elif args.data == "agnews" and psn in ["addsent", "addsentv2"]:
                        data["attacker"]["poisoner"]["triggers"] = "in recent events, it is discovered"
                    else:
                        print("\nNo trigger phrase needed.\n")


                    # define poison data sampling seed
                    data["attacker"]["poisoner"]["rs"] = int(rs)

                    # define poison rate
                    data["attacker"]["poisoner"]["poison_rate"] = float(pr)


                    # load = True for all
                    data["attacker"]["poisoner"]["load"] = True

                    # dump json configs
                    if args.filter == "true":
                        config_dir = os.path.join(out_dir, 'filter/')
                    else:
                        config_dir = os.path.join(out_dir, 'nofilter/')
                    os.makedirs(config_dir, exist_ok=True)
                    with open(os.path.join(config_dir, 'A_' + psn + '_' + pr + '_' + str(rs) + '_' + str(layer) +
                                                       '.json'),
                                           'w') as output:
                        json.dump(data, output, indent=3)

        b.close()
        p.close()
        print("\nConfig files created for {} - {}.\n".format(args.data, psn))
        
        
    else:
        b = open(base, 'r+')
        data = json.load(b)
        p = open(poisoner, 'r+')
        poisoner = json.load(p)

        for style in styles:
            for rs in rs_list:
                for pr in pr_list:
                    # define trainer type
                    # data["attacker"]["name"] = args.trainer
                    data["attacker"]["train"]["name"] = args.trainer
                    if args.trainer == "visual":
                        for layer in layers:
                            data["attacker"]["train"]["layer"] = layer

                            # add poisoner
                            data["attacker"]["poisoner"] = poisoner["poisoner"]

                            # define data
                            data["target_dataset"]["name"] = args.data
                            data["poison_dataset"]["name"] = args.data
                            data["attacker"]["train"]["data"] = args.data
                            data["attacker"]["poisoner"]["data"] = args.data

                            if args.data in ["agnews", "yelp"]:
                                data["attacker"]["train"]["batch_size"] = 16

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
                            if args.data == "agnews":
                                data["victim"]["num_classes"] = 4
                            elif args.data == "blog":
                                data["victim"]["num_classes"] = 3
                            else:
                                data["victim"]["num_classes"] = 2

                            # define filter/unfilter
                            if args.filter == "true":
                                data["attacker"]["train"]["filter"] = True
                                data["attacker"]["poisoner"]["filter"] = True
                            else:
                                data["attacker"]["train"]["filter"] = False
                                data["attacker"]["poisoner"]["filter"] = False

                            # define target label
                            if args.data in ["sst-2", "yelp"]:
                                data["attacker"]["poisoner"]["target_label"] = 1
                            else:
                                data["attacker"]["poisoner"]["target_label"] = 0

                            # define poison data sampling seed
                            data["attacker"]["poisoner"]["rs"] = int(rs)

                            # define poison rate
                            data["attacker"]["poisoner"]["poison_rate"] = float(pr)

                            # define llm model
                            data["attacker"]["poisoner"]["llm"] = args.llm

                            # load = True for all
                            data["attacker"]["poisoner"]["load"] = True
                            if args.poisoner in ["llmbkd", "attrbkd"]:
                                data["attacker"]["poisoner"]["style"] = style
                                data["attacker"]["train"]["style"] = style

                            # dump json configs
                            if args.filter == "true":
                                config_dir = os.path.join(out_dir, 'filter/')
                            else:
                                config_dir = os.path.join(out_dir, 'nofilter/')
                            os.makedirs(config_dir, exist_ok=True)

                            if args.poisoner in ["llmbkd", "attrbkd"]:
                                if style == "sports commentators":
                                    with open(os.path.join(config_dir,
                                                           'A_' + args.poisoner + '_' + args.llm + '_sports_' + pr + '_' +
                                                           rs + '_' + str(layer) +
                                                           '.json'),
                                              'w') as output:
                                        json.dump(data, output, indent=3)
                                elif style == "40s gangster movies":
                                    with open(
                                            os.path.join(config_dir, 'A_' + args.poisoner + '_' + args.llm + '_gangster_' +
                                                                     pr + '_' + rs + '_' + str(layer) +
                                                                     '.json'),
                                            'w') as output:
                                        json.dump(data, output, indent=3)
                                elif style == "rare words":
                                    with open(os.path.join(config_dir, 'A_' + args.poisoner + '_' + args.llm + '_rare_' + pr
                                                                       + '_' + rs + '_' + str(layer) +
                                                                       '.json'),
                                              'w') as output:
                                        json.dump(data, output, indent=3)
                                else:
                                    with open(os.path.join(config_dir, 'A_' + args.poisoner + '_' + args.llm + '_' + style +
                                                                       '_' + pr + '_' + rs + '_' + str(layer) +
                                                                       '.json'),
                                              'w') as output:
                                        json.dump(data, output, indent=3)
                            else:
                                with open(os.path.join(config_dir, 'A_' + args.poisoner + '_' + pr + '_' + rs + '_' + str(
                                        layer) + '.json'),
                                          'w') as output:
                                    json.dump(data, output, indent=3)
                            print("\nConfig files created for {} - {} with {}.\n".format(args.data, style, args.llm))
                    else:
                        # add poisoner
                        data["attacker"]["poisoner"] = poisoner["poisoner"]

                        # define data
                        data["target_dataset"]["name"] = args.data
                        data["poison_dataset"]["name"] = args.data
                        data["attacker"]["train"]["data"] = args.data
                        data["attacker"]["poisoner"]["data"] = args.data

                        if args.data in ["agnews", "yelp"]:
                            data["attacker"]["train"]["batch_size"] = 16

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
                        if args.data == "agnews":
                            data["victim"]["num_classes"] = 4
                        elif args.data == "blog":
                            data["victim"]["num_classes"] = 3
                        else:
                            data["victim"]["num_classes"] = 2


                        # define filter/unfilter
                        if args.filter == "true":
                            data["attacker"]["train"]["filter"] = True
                            data["attacker"]["poisoner"]["filter"] = True
                        else:
                            data["attacker"]["train"]["filter"] = False
                            data["attacker"]["poisoner"]["filter"] = False

                        # define target label
                        if args.data in ["sst-2", "yelp"]:
                            data["attacker"]["poisoner"]["target_label"] = 1
                        else:
                            data["attacker"]["poisoner"]["target_label"] = 0

                        # define poison data sampling seed
                        data["attacker"]["poisoner"]["rs"] = int(rs)

                        # define poison rate
                        data["attacker"]["poisoner"]["poison_rate"] = float(pr)

                        # define llm model
                        data["attacker"]["poisoner"]["llm"] = args.llm

                        # load = True for all
                        data["attacker"]["poisoner"]["load"] = True
                        if args.poisoner in ["llmbkd", "attrbkd"]:
                            data["attacker"]["poisoner"]["style"] = style
                            data["attacker"]["train"]["style"] = style

                        # dump json configs
                        if args.filter == "true":
                            config_dir = os.path.join(out_dir, 'filter/')
                        else:
                            config_dir = os.path.join(out_dir, 'nofilter/')
                        os.makedirs(config_dir, exist_ok=True)

                        if args.poisoner in ["llmbkd", "attrbkd"]:
                            if style == "sports commentators":
                                with open(os.path.join(config_dir, 'A_' + args.poisoner + '_' + args.llm + '_sports_' + pr + '_' +
                                                                   rs + '_' + str(layer) +
                                                                   '.json'),
                                          'w') as output:
                                    json.dump(data, output, indent=3)
                            elif style == "40s gangster movies":
                                with open(os.path.join(config_dir, 'A_' + args.poisoner + '_' + args.llm + '_gangster_' +
                                                                   pr + '_' + rs + '_' + str(layer) +
                                                                   '.json'),
                                          'w') as output:
                                    json.dump(data, output, indent=3)
                            elif style == "rare words":
                                with open(os.path.join(config_dir, 'A_' + args.poisoner + '_' + args.llm + '_rare_' + pr
                                                                   + '_' + rs + '_' + str(layer) +
                                                                   '.json'),
                                          'w') as output:
                                    json.dump(data, output, indent=3)
                            else:
                                with open(os.path.join(config_dir, 'A_' + args.poisoner + '_' + args.llm + '_' + style +
                                                                   '_' + pr + '_' + rs + '_' + str(layer) +
                                                                   '.json'),
                                          'w') as output:
                                    json.dump(data, output, indent=3)
                        else:
                            with open(os.path.join(config_dir, 'A_' + args.poisoner + '_' + pr + '_' + rs + '_' + str(layer) + '.json'),
                                      'w') as output:
                                json.dump(data, output, indent=3)
                print("\nConfig files created for {} - {} with {}.\n".format(args.data, style, args.llm))
    
        b.close()
        p.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment settings
    parser.add_argument('--poisoner', type=str, default='llmbkd', help='poisoner name.')
    parser.add_argument('--llm', type=str, default='gpt-3.5-turbo', help='poisoner llm.')
    parser.add_argument('--model', type=str, default='roberta', help='baseline poisoner name.')
    parser.add_argument('--data', type=str, default='sst-2', help='dataset.')
    parser.add_argument('--filter', type=str, default='true', help='poison selection')
    parser.add_argument('--trainer', type=str, default='base', help='trainer type: base or visual')
    # parser.add_argument('--layer', type=int, default=-1, help='layer of a victim model')

    args = parser.parse_args()
    main(args)