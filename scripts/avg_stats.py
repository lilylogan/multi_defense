'''
average stats.csv over all random seeds
'''

import os
import re
import time
import json
import math
import pickle
import argparse
from datetime import datetime

import torch
import numpy as np
import pandas as pd



def stats_dir(args, in_base_dir, poisoner, filter, rs, llm=None, style=None):
    if args.step == 'defend':
        if args.defender == 'insert':
            if poisoner == 'llmbkd':
                in_dir = os.path.join(in_base_dir, args.step, args.defender, poisoner, llm, style, filter,
                                       args.poison_rate, rs)
            else:
                in_dir = os.path.join(in_base_dir, args.step, args.defender, poisoner, filter,
                                       args.poison_rate, rs)
        else:
            if poisoner == 'llmbkd':
                in_dir = os.path.join(in_base_dir, args.step, args.defender, poisoner, llm, style, filter, rs,
                                      args.poison_rate)
            else:
                in_dir = os.path.join(in_base_dir, args.step, args.defender, poisoner, filter, rs, args.poison_rate)
    else:
        if poisoner == 'llmbkd':
            in_dir = os.path.join(in_base_dir, args.step, poisoner, llm, style, filter, rs)
        else:
            in_dir = os.path.join(in_base_dir, args.step, poisoner, filter, rs)

    return in_dir


def output_dir(args, out_base_dir, poisoner, filter, llm=None, style=None):
    if args.step == 'defend':
        if args.defender == 'insert':
            if poisoner == 'llmbkd':
                out_dir = os.path.join(out_base_dir, args.step, args.defender, poisoner, llm, style, filter,
                                       args.poison_rate)
            else:
                out_dir = os.path.join(out_base_dir, args.step, args.defender, poisoner, filter,
                                       args.poison_rate)
            os.makedirs(out_dir, exist_ok=True)
        else:
            if poisoner == 'llmbkd':
                out_dir = os.path.join(out_base_dir, args.step, args.defender, poisoner, llm, style, filter)
            else:
                out_dir = os.path.join(out_base_dir, args.step, args.defender, poisoner, filter)
            os.makedirs(out_dir, exist_ok=True)
    else:
        if poisoner == 'llmbkd':
            out_dir = os.path.join(out_base_dir, args.step, poisoner, llm, style, filter)
        else:
            out_dir = os.path.join(out_base_dir, args.step, poisoner, filter)
        os.makedirs(out_dir, exist_ok=True)

    return out_dir


def main(args):
    # read from stats dir
    in_base_dir = os.path.join('stats/', args.data, args.model)

    # set up out dir
    out_base_dir = os.path.join('avg_stats/', args.data, args.model)
    os.makedirs(out_base_dir, exist_ok=True)

    if args.filter == "true":
        filter = "filter"
    else:
        filter = 'nofilter'


    # random seeds
    rs_list = ["0", "2", "42"] #"1", "10",, "42"


    # # styles
    # if args.step == "attack":
    #     if args.data == "sst-2":
    #         # styles = ["default", "bible", "Shakespeare", "lawyers", "Gen-z", "sports commentators", "British",
    #         #           "politicians", "TikTok", "40s gangster movies", "rare words", "tweets", "poems", "lyrics"]
    #         styles = ["default", "bible", "Gen-z", "sports commentators"]
    #     elif args.data in ["hsol", "toxigen", "agnews"]:
    #         styles = ["bible", "default", "Gen-z", "sports commentators"]
    #     else:
    #         print("\nInvalid data.")
    #         exit()
    # else:
    #     styles = ["bible", "default",  "Gen-z", "sports commentators"]
    # styles = ["default", "default_bt", "bible", "shakespeare", "tweets", "lawyers"]  #"poems", "tweets", "lyrics", "default_bt", "tweets", "default", , "bible"
    # styles = ["attr_2", "attr_3", "attr_1"]
    # styles = ["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_combo_1", "attr_combo_2", "attr_combo_3",
    #           "attr_combo_4", "attr_combo_5"]

    # styles = ["bible_3", "shakespeare_3", "tweets_3", "lawyers_3", "bible_5+1", "shakespeare_5+1", "tweets_5+1", "lawyers_5+1"]

    styles = ["bible", "shakespeare", "tweets", "lawyers"]


    # params
    poisoners = ["llmbkd"] #"addsent", "addsentv2","addsent", "addsentv2", "badnets", "stylebkd", "synbkd",
    # "badnets", , "synbkd", "gptbkd"


    for psn in poisoners:
        if psn != "llmbkd":
            results = {}
            CACC = []
            ASR = []
            AVG_CACC = []
            STD_CACC = []
            AVG_ASR = []
            STD_ASR = []

            for rs in rs_list:
                try:
                    stats_rs = stats_dir(args, in_base_dir, psn, filter, rs)
                    print("\nReading results from {}".format(stats_rs))
                    stat_df = pd.read_csv(os.path.join(stats_rs, "stats.csv"))

                     # print(stat_df)

                    cacc = stat_df["CACC"].tolist()
                    asr = stat_df["ASR"].tolist()
                    CACC.append(cacc)
                    ASR.append(asr)


                    poison_rate = stat_df["poison_rate"].tolist()
                    dataset = stat_df["dataset"].tolist()
                    poisoner = stat_df["poisoner"].tolist()
                    clean_label = stat_df["clean_label"].tolist()
                    target_label = stat_df["target_label"].tolist()
                    if args.defender == "insert":
                        dp_rate = stat_df["defense/poison_ratio"].tolist()

                except:

                    pass
                    # poison_rate = "Not enough poison data"
                    # dataset = "-"
                    # poisoner = "-"
                    # clean_label = "-"
                    # target_label = "-"

            print("======")

            all_CACC = list(zip(*CACC))
            all_ASR = list(zip(*ASR))




            for c in all_CACC:
                c = tuple([x for x in c if not math.isnan(x) == True])
                avg_cacc = np.average(c)
                std_cacc = np.std(c)
                AVG_CACC.append(avg_cacc)
                STD_CACC.append(std_cacc)


            for a in all_ASR:
                a = tuple([x for x in a if not math.isnan(x) == True])
                avg_asr = np.average(a)
                std_asr = np.std(a)
                AVG_ASR.append(avg_asr)
                STD_ASR.append(std_asr)

            # print("\n")
            # print(AVG_ASR)
            # print(STD_ASR)
            # print(len(AVG_ASR))

            if args.step == "defend" and args.defender == "insert":
                results["defense/poison_ratio"] = dp_rate
            results["poison_rate"] = poison_rate
            results["dataset"] = dataset
            results["poisoner"] = poisoner
            results["clean_label"] = clean_label
            results["target_label"] = target_label
            results["AVG_CACC"] = AVG_CACC
            results["AVG_ASR"] = AVG_ASR
            results["STD_CACC"] = STD_CACC
            results["STD_ASR"] = STD_ASR

            # print(results)

            res_df = pd.DataFrame(results)
            print(res_df)

            # save avg results
            print("Output Dir -- {}/{}/{}".format(out_base_dir, psn, filter))
            out_dir = output_dir(args, out_base_dir, psn, filter)

            res_df.to_csv(os.path.join(out_dir, "avg_stats.csv"), index=False)
            print("\nAveraged stats are saved for {}-{}-{}-{}-{}.\n".format(args.data, args.model, args.step, psn,
                                                                            filter))

        else:
            for style in styles:
                results = {}
                CACC = []
                ASR = []
                AVG_CACC = []
                STD_CACC = []
                AVG_ASR = []
                STD_ASR = []

                for rs in rs_list:
                    try:
                        stats_rs = stats_dir(args, in_base_dir, psn, filter, rs, llm=args.llm, style=style)
                        print("\nReading results from {}".format(stats_rs))
                        stat_df = pd.read_csv(os.path.join(stats_rs, "stats.csv"))

                        cacc = stat_df["CACC"].tolist()
                        asr = stat_df["ASR"].tolist()
                        CACC.append(cacc)
                        ASR.append(asr)

                        poison_rate = stat_df["poison_rate"].tolist()
                        dataset = stat_df["dataset"].tolist()
                        poisoner = stat_df["poisoner"].tolist()
                        clean_label = stat_df["clean_label"].tolist()
                        target_label = stat_df["target_label"].tolist()
                        LLM = stat_df["llm"].tolist()
                        if args.defender == "insert":
                            dp_rate = stat_df["defense/poison_ratio"].tolist()

                    except:
                        pass
                        # poison_rate = "Not enough poison data"
                        # dataset = "-"
                        # poisoner = "-"
                        # clean_label = "-"
                        # target_label = "-"

                all_CACC = list(zip(*CACC))
                all_ASR = list(zip(*ASR))


                for c in all_CACC:
                    c = tuple([x for x in c if not math.isnan(x) == True])
                    avg_cacc = np.average(c)
                    std_cacc = np.std(c)
                    AVG_CACC.append(avg_cacc)
                    STD_CACC.append(std_cacc)

                for a in all_ASR:
                    a = tuple([x for x in a if not math.isnan(x) == True])
                    avg_asr = np.average(a)
                    std_asr = np.std(a)
                    AVG_ASR.append(avg_asr)
                    STD_ASR.append(std_asr)

                # print("\n")
                # print(AVG_ASR)
                # print(STD_ASR)
                # print(len(AVG_ASR))

                if args.step == "defend" and args.defender == "insert":
                    results["defense/poison_ratio"] = dp_rate
                results["poison_rate"] = poison_rate
                results["dataset"] = dataset
                results["poisoner"] = poisoner
                results["llm"] = LLM
                results["clean_label"] = clean_label
                results["target_label"] = target_label
                results["AVG_CACC"] = AVG_CACC
                results["AVG_ASR"] = AVG_ASR
                results["STD_CACC"] = STD_CACC
                results["STD_ASR"] = STD_ASR


                res_df = pd.DataFrame(results)
                print(res_df)

                # save avg results
                out_dir = output_dir(args, out_base_dir, psn, filter, llm=args.llm, style=style)
                res_df.to_csv(os.path.join(out_dir, "avg_stats.csv"), index=False)
                print("\nAveraged stats are saved for {}-{}-{}-{}-{}-{}-{}.\n".format(args.data, args.model, args.step,
                                                                                psn, args.llm, style,
                                                                                filter))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment settings
    parser.add_argument('--model', type=str, default='roberta', help='model type.')
    parser.add_argument('--data', type=str, default='sst-2', help='data type')
    parser.add_argument('--step', type=str, default='attack', help='attack or defend.')
    parser.add_argument('--defender', type=str, default='insert', help='defender.')
    parser.add_argument('--filter', type=str, default='true', help='filter or unfilter')
    parser.add_argument('--poison_rate', type=str, default='0.01', help='poisoner name')
    parser.add_argument('--llm', type=str, default='gpt-3.5-turbo', help='poisoner llm name')

    args = parser.parse_args()
    main(args)