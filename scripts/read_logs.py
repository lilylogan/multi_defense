'''
read test accuracies from log files in the 'logs/' directory to the 'stats/' directory
'''

import os
import re
import time
import json
import pickle
import argparse
from datetime import datetime

import torch
import numpy as np
import pandas as pd



def get_num(txt_w_num):
    txt = re.sub(r'^.*?: ', '', txt_w_num)
    txt = txt.strip('\n')
    return txt

def get_setting(setting_str):
    # set = re.sub(r'^.*?] ', '', setting_str)
    set = setting_str
    set = set.strip('\n')
    return set


def str_to_dict(string):
    # remove the curly braces from the string
    string = string.strip('{}')

    # split the string into key-value pairs
    pairs = string.split(', ')

    # use a dictionary comprehension to create the dictionary, converting the values to integers and removing the quotes from the keys
    return {key[1:-1]: value for key, value in (pair.split(': ') for pair in pairs)}


def stats_dir(args, out_base_dir, poisoner, filter, rs, llm=None, style=None):
    if args.step == 'defend':
        if args.defender == 'insert':
            if poisoner == 'llmbkd':
                out_dir = os.path.join(out_base_dir, args.step, args.defender, poisoner, llm, style, filter,
                                       args.poison_rate, rs)
            else:
                out_dir = os.path.join(out_base_dir, args.step, args.defender, poisoner, filter,
                                       args.poison_rate, rs)
            os.makedirs(out_dir, exist_ok=True)
        else:
            if poisoner == 'llmbkd':
                out_dir = os.path.join(out_base_dir, args.step, args.defender, poisoner, llm, style, filter, rs,
                                       args.poison_rate)
            else:
                out_dir = os.path.join(out_base_dir, args.step, args.defender, poisoner, filter, rs, args.poison_rate)
            os.makedirs(out_dir, exist_ok=True)
    else:
        if poisoner == 'llmbkd':
            out_dir = os.path.join(out_base_dir, args.step, poisoner, llm, style, filter, rs)
        else:
            out_dir = os.path.join(out_base_dir, args.step, poisoner, filter, rs)
        os.makedirs(out_dir, exist_ok=True)

    return out_dir


def result_dir(args, poisoner, filter, rs, llm=None, style=None):
    if args.step == 'defend':
        if args.defender == 'insert':
            if poisoner == 'llmbkd':
                res_dir = os.path.join('logs/', args.data, args.model, args.step, args.defender, poisoner, llm,
                                       style, filter, args.poison_rate, rs)
            else:
                res_dir = os.path.join('logs/', args.data, args.model, args.step, args.defender, poisoner,
                                       filter, args.poison_rate, rs)

        else:
            if poisoner == 'llmbkd':
                res_dir = os.path.join('logs/', args.data, args.model, args.step, args.defender, poisoner, llm,
                                       style, filter, rs)
            else:
                res_dir = os.path.join('logs/', args.data, args.model, args.step, args.defender, poisoner, filter,
                                       rs)
    else:
        if poisoner == 'llmbkd':
            res_dir = os.path.join('logs/', args.data, args.model, args.step, poisoner, llm, style, filter, rs)
        else:
            res_dir = os.path.join('logs/', args.data, args.model, args.step, poisoner, filter, rs)

    return res_dir




def main(args):

    # setup output dir
    out_base_dir = os.path.join('stats/', args.data, args.model)
    if args.filter == "true":
        filter = "filter"
    else:
        filter = 'nofilter'

    # # styles
    # if args.step == 'attack':
    #     if args.data == "sst-2":
    #         # styles = ["default", "bible", "Shakespeare", "lawyers", "Gen-z", "sports commentators", "British",
    #         #           "politicians", "TikTok", "40s gangster movies", "rare words", "tweets", "poems", "lyrics"]
    #         styles = ["default", "bible", "Gen-z", "sports commentators"]
    #     elif args.data in ["hsol", "toxigen", "agnews"]:
    #         styles = ["default", "bible", "Gen-z", "sports commentators"]
    #     else:
    #         print("\nInvalid data.")
    #         exit()
    # else:
    #     styles = ["default", "bible", "Gen-z", "sports commentators"]

    # styles = ["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_combo_1", "attr_combo_2", "attr_combo_3",
    #      "attr_combo_4", "attr_combo_5"]
    # styles = ["bible_3", "shakespeare_3", "tweets_3", "lawyers_3", "bible_5+1", "shakespeare_5+1", "tweets_5+1", "lawyers_5+1"]
    styles = ["bible", "shakespeare", "tweets", "lawyers"]
    #"poems", "tweets", "lyrics", "shakespeare",
    # "default_bt", ,
    # "tweets",
    # "bible", "default"
    # params
    poisoners = ["llmbkd"]# "addsent", "addsentv2","addsent", "addsentv2",
    # "badnets", "stylebkd", "synbkd",
    # "badnets", , "synbkd", "gptbkd"
    # "badnets", , "synbkd", "gptbkd"
    # "badnets", , "synbkd", "gptbkd"

    if args.step == 'attack':
        if args.data in ["sst-2", "hsol", "toxigen", "enron"]:
            # frac_list = ["0.001", "0.002", "0.003", "0.004", "0.005", "0.006", "0.007", "0.008", "0.009", "0.01",
            #              "0.02", "0.03", "0.04", "0.05"]
            frac_list = ["0.002", "0.004", "0.006", "0.008",  "0.01", "0.02", "0.03", "0.04", "0.05"]
        elif args.data == "agnews":
            # frac_list = ["0.0002", "0.0004", "0.0006", "0.0008", "0.001", "0.002", "0.004", "0.006", "0.008", "0.01",
            #              "0.02", "0.03"]
            frac_list = ["0.0004", "0.0008", "0.002", "0.006", "0.01", "0.02", "0.03", "0.04", "0.05"]
        elif args.data == "yelp":
            frac_list = ["0.0002", "0.0004", "0.0006", "0.0008", "0.001", "0.002", "0.004", "0.006", "0.008", "0.01"]
        else:
            print("\nInvalid data. Runtime stopped.")
            exit()

    else:
        frac_list = [args.poison_rate]
        d_p_rates = ["0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"]

    # random seeds
    rs_list = ["0", "2", "42"]  # "1", "10",

    for poisoner in poisoners:
        if poisoner != "llmbkd":
            for rs in rs_list:
                out_dir = stats_dir(args, out_base_dir, poisoner, filter, rs)

                # read in results
                res_dir = result_dir(args, poisoner, filter, rs)
                print("\nReading results from {}".format(res_dir))

                stats = []
                for frac in frac_list:

                    if args.step == 'defend' and args.defender == 'insert':
                        # print("\nReading results for Insert defender...")
                        stats = []
                        for r in d_p_rates:
                            stat = {}
                            # log_dir = os.path.join(res_dir, frac)
                            log_dir = os.path.join(res_dir, r)
                            print("\nReading results from...", log_dir)
                            try:
                                with open(os.path.join(log_dir, "log.txt")) as f:
                                    # res = json.load(f) # todo: cannot use json only because there are single quotes in the text file,
                                    # but json only recognize double quotes

                                    lines = f.read()
                                    res = str_to_dict(lines)
                                    # print('\n')
                                    print(res)
                                    # print(type(res))
                                    #
                                    # print(res['poison_rate'])

                                    stat['poison_rate'] = res['poison_rate']
                                    stat['dataset'] = res['poison_dataset'].strip("'")
                                    stat['poisoner'] = res['poisoner'].strip("'")
                                    stat['defender'] = args.defender
                                    stat['defense/poison_ratio'] = res['defense_rate']
                                    stat['clean_label'] = res['label_consistency']
                                    stat['target_label'] = res['target_label']
                                    stat['CACC'] = res['CACC']
                                    stat['ASR'] = res['ASR']
                                    stat['rs'] = res['rs']

                                    stats.append(stat)

                            except:
                                stat['poison_rate'] = frac
                                stat['dataset'] = args.data
                                stat['poisoner'] = poisoner
                                if args.step == 'defend':
                                    stat['defender'] = args.defender
                                    stat['defense/poison_ratio'] = r
                                stat['clean_label'] = None
                                stat['target_label'] = None
                                stat['CACC'] = None
                                stat['ASR'] = None
                                stat['rs'] = rs

                                stats.append(stat)

                    else:
                        stat = {}
                        log_dir = os.path.join(res_dir, frac)
                        print("\nReading results from {}".format(log_dir))
                        try:
                            with open(os.path.join(log_dir, "log.txt")) as f:
                                # res = json.load(f) # todo: cannot use json only because there are single quotes in the text file,
                                # but json only recognize double quotes

                                lines = f.read()
                                res = str_to_dict(lines)
                                # print('\n')
                                print(res)
                                # print(type(res))

                                stat['poison_rate'] = res['poison_rate']
                                stat['dataset'] = res['poison_dataset'].strip("'")
                                stat['poisoner'] = res['poisoner'].strip("'")
                                if args.step == 'defend':
                                    stat['defender'] = args.defender
                                stat['clean_label'] = res['label_consistency']
                                stat['target_label'] = res['target_label']
                                stat['CACC'] = res['CACC']
                                stat['ASR'] = res['ASR']
                                stat['rs'] = rs

                                stats.append(stat)


                        except:
                            # pass
                            stat['poison_rate'] = frac
                            stat['dataset'] = args.data
                            stat['poisoner'] = poisoner
                            if args.step == 'defend':
                                stat['defender'] = args.defender
                            stat['clean_label'] = None
                            stat['target_label'] = None
                            stat['CACC'] = None
                            stat['ASR'] = None
                            stat['rs'] = rs

                            stats.append(stat)


                stats_df = pd.DataFrame(stats)
                # print("\nafter", stats)

                f_name = 'stats.csv'

                stat_dir = out_dir
                os.makedirs(stat_dir, exist_ok=True)
                stats_df.to_csv(os.path.join(stat_dir, f_name), index=False)
                if args.step == 'defend':
                    print('\nStats saved for {}-{}-{}-{}-{}-{}.\n'.format(args.data, args.model, args.step, args.defender,
                                                                     poisoner, rs))
                else:
                    print('\nStats saved for {}-{}-{}-{}-{}.\n'.format(args.data, args.model, args.step, poisoner, rs))

        else:
            for style in styles:
                for rs in rs_list:
                    out_dir = stats_dir(args, out_base_dir, poisoner, filter, rs, llm=args.llm, style=style)

                    # read in results
                    res_dir = result_dir(args, poisoner, filter, rs, llm=args.llm, style=style)
                    print("\nReading results from {}".format(res_dir))

                    stats = []
                    for frac in frac_list:

                        if args.step == 'defend' and args.defender == 'insert':
                            # print("\nReading results for Insert defender...")
                            stats = []
                            for r in d_p_rates:
                                stat = {}
                                # log_dir = os.path.join(res_dir, frac)
                                log_dir = os.path.join(res_dir, r)
                                print("\nReading results from...", log_dir)
                                try:
                                    with open(os.path.join(log_dir, "log.txt")) as f:
                                        # res = json.load(f) # todo: cannot use json only because there are single quotes in the text file,
                                        # but json only recognize double quotes

                                        lines = f.read()
                                        res = str_to_dict(lines)
                                        # print('\n')
                                        print(res)
                                        # print(type(res))
                                        #
                                        # print(res['poison_rate'])

                                        stat['poison_rate'] = res['poison_rate']
                                        stat['dataset'] = res['poison_dataset'].strip("'")
                                        stat['poisoner'] = res['poisoner'].strip("'")
                                        stat['defender'] = args.defender
                                        stat['defense/poison_ratio'] = res['defense_rate']
                                        stat['clean_label'] = res['label_consistency']
                                        stat['target_label'] = res['target_label']
                                        stat['CACC'] = res['CACC']
                                        stat['ASR'] = res['ASR']
                                        stat['rs'] = rs
                                        stat['llm'] = args.llm

                                        stats.append(stat)

                                except:
                                    stat['poison_rate'] = frac
                                    stat['dataset'] = args.data
                                    stat['poisoner'] = poisoner
                                    if args.step == 'defend':
                                        stat['defender'] = args.defender
                                        stat['defense/poison_ratio'] = r
                                    stat['clean_label'] = None
                                    stat['target_label'] = None
                                    stat['CACC'] = None
                                    stat['ASR'] = None
                                    stat['rs'] = rs
                                    stat['llm'] = args.llm

                                    stats.append(stat)

                        else:
                            stat = {}
                            log_dir = os.path.join(res_dir, frac)
                            print("\nReading results from {}".format(log_dir))
                            try:
                                with open(os.path.join(log_dir, "log.txt")) as f:
                                    # res = json.load(f) # todo: cannot use json only because there are single quotes in the text file,
                                    # but json only recognize double quotes

                                    lines = f.read()
                                    res = str_to_dict(lines)
                                    # print('\n')
                                    print(res)

                                    stat['poison_rate'] = res['poison_rate']
                                    stat['dataset'] = res['poison_dataset'].strip("'")
                                    stat['poisoner'] = res['poisoner'].strip("'")
                                    if args.step == 'defend':
                                        stat['defender'] = args.defender
                                    stat['clean_label'] = res['label_consistency']
                                    stat['target_label'] = res['target_label']
                                    stat['CACC'] = res['CACC']
                                    stat['ASR'] = res['ASR']
                                    stat['rs'] = rs
                                    stat['llm'] = args.llm

                                    stats.append(stat)

                            except:
                                stat['poison_rate'] = frac
                                stat['dataset'] = args.data
                                stat['poisoner'] = poisoner
                                if args.step == 'defend':
                                    stat['defender'] = args.defender
                                stat['clean_label'] = None
                                stat['target_label'] = None
                                stat['CACC'] = None
                                stat['ASR'] = None
                                stat['rs'] = rs
                                stat['llm'] = args.llm

                                stats.append(stat)

                    stats_df = pd.DataFrame(stats)

                    f_name = 'stats.csv'
                    stat_dir = out_dir
                    os.makedirs(stat_dir, exist_ok=True)
                    stats_df.to_csv(os.path.join(stat_dir, f_name), index=False)
                    if args.step == 'defend':
                        print('\nStats saved for {}-{}-{}-{}-{}-{}-{}-{}.\n'.format(args.data, args.model, args.step,
                                                                               args.defender,
                                                                              poisoner, args.llm, style, rs))
                    else:
                        print('\nStats saved for {}-{}-{}-{}-{}-{}-{}.\n'.format(args.data, args.model, args.step,
                                                                            poisoner, args.llm, style, rs))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment settings
    parser.add_argument('--model', type=str, default='roberta', help='model type.')
    parser.add_argument('--data', type=str, default='sst-2', help='data type')
    parser.add_argument('--step', type=str, default='attack', help='attack or defend.')
    parser.add_argument('--defender', type=str, default='insert', help='defender.')
    # parser.add_argument('--poisoner', type=str, default='addsent', help='poisoner name')
    parser.add_argument('--filter', type=str, default='true', help='filter or unfilter')
    parser.add_argument('--poison_rate', type=str, default='0.01', help='poisoner name')
    parser.add_argument('--llm', type=str, default='gpt-3.5-turbo', help='poisoner llm name')

    args = parser.parse_args()
    main(args)