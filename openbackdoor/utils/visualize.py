import os
import sys


def result_visualizer(result):
    stream_writer = sys.stdout.write
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80

    left = []
    right = []
    for key, val in result.items():
        left.append(" " + key + ": ")
        if isinstance(val, bool):
            right.append(" yes" if val else " no")
        elif isinstance(val, int):
            right.append(" %d" % val)
        elif isinstance(val, float):
            right.append(" %.5g" % val)
        else:
            right.append(" %s" % val)
        right[-1] += " "

    max_left = max(list(map(len, left)))
    max_right = max(list(map(len, right)))
    if max_left + max_right + 3 > cols:
        delta = max_left + max_right + 3 - cols
        if delta % 2 == 1:
            delta -= 1
            max_left -= 1
        max_left -= delta // 2
        max_right -= delta // 2
    total = max_left + max_right + 3

    title = "Summary"
    if total - 2 < len(title):
        title = title[:total - 2]
    offtitle = ((total - len(title)) // 2) - 1
    stream_writer("+" + ("=" * (total - 2)) + "+\n")
    stream_writer("|" + " " * offtitle + title + " " * (total - 2 - offtitle - len(title)) + "|" + "\n")
    stream_writer("+" + ("=" * (total - 2)) + "+\n")
    for l, r in zip(left, right):
        l = l[:max_left]
        r = r[:max_right]
        l += " " * (max_left - len(l))
        r += " " * (max_right - len(r))
        stream_writer("|" + l + "|" + r + "|" + "\n")
    stream_writer("+" + ("=" * (total - 2)) + "+\n")



def display_results(config, results):

    poisoner = config['attacker']['poisoner']['name']
    poison_rate = config['attacker']['poisoner']['poison_rate']
    label_consistency = config['attacker']['poisoner']['label_consistency']
    label_dirty = config['attacker']['poisoner']['label_dirty']
    target_label = config['attacker']['poisoner']['target_label']
    poison_dataset = config['poison_dataset']['name']
    victim_model = config['victim']['model']


    CACC = results['test-clean']['accuracy']
    if 'test-poison' in results.keys():
        ASR = 1 - results['test-poison']['accuracy']
    else:
        asrs = [results[k]['accuracy'] for k in results.keys() if k.split('-')[1] == 'poison']
        ASR = max(asrs)

    PPL = results["ppl"]
    GE = results["grammar"]
    USE = results["use"]
    FRR = results["FRR"]
    FAR = results["FAR"]




    filter = config['attacker']['train']['filter']
    if filter == True:
        flt = "filter"
    else:
        flt = "nofilter"

    if poisoner in ['attrbkd', 'llmbkd']:
        style = config["attacker"]["poisoner"]["style"]
        llm = config['attacker']['poisoner']['llm']
        display_result = {"poison_dataset": poison_dataset, "poisoner": poisoner, "llm": llm,
                          "poison_rate": poison_rate,
                          "label_consistency": label_consistency, "label_dirty": label_dirty,
                          "target_label": target_label,
                          "CACC": CACC, 'ASR': ASR, "ΔPPL": PPL, "ΔGE": GE, "USE": USE, "FRR": FRR, "FAR": FAR}
    else:
        display_result = {"poison_dataset": poison_dataset, "poisoner": poisoner,
                          "poison_rate": poison_rate,
                          "label_consistency": label_consistency, "label_dirty": label_dirty,
                          "target_label": target_label,
                          "CACC": CACC, 'ASR': ASR, "ΔPPL": PPL, "ΔGE": GE, "USE": USE, "FRR": FRR, "FAR": FAR}

    # if defender is not None:
    if 'defender' in config:
        defender = config['defender']['name']
        rs = config["attacker"]["poisoner"]["rs"]
        display_result["defender"] = defender
        if defender == 'react':
            defense_rate = config['defender']['defense_rate']
            display_result["defense_rate"] = defense_rate

            display_result["rs"] = rs

            if poisoner in ['attrbkd', 'llmbkd']:
                out_dir = os.path.join('./logs', poison_dataset, victim_model, 'defend/', defender, poisoner, llm,
                                       style, flt, str(poison_rate), str(rs), str(defense_rate))
            else:
                out_dir = os.path.join('./logs', poison_dataset, victim_model, 'defend/', defender, poisoner, flt,
                                       str(poison_rate), str(rs), str(defense_rate))
            os.makedirs(out_dir, exist_ok=True)

        else:
            if poisoner in ['attrbkd', 'llmbkd']:
                out_dir = os.path.join('./logs', poison_dataset, victim_model, 'defend/', defender, poisoner, llm,
                                       style, flt, str(rs), str(poison_rate))
            else:
                out_dir = os.path.join('./logs', poison_dataset, victim_model, 'defend/', defender, poisoner,  flt,
                                       str(rs), str(poison_rate))
            os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, 'log.txt'), 'w') as sys.stdout:
            sys.stdout.write(str(display_result))

    else:
        rs = config["attacker"]["poisoner"]["rs"]
        display_result["rs"] = rs

        if poisoner in ['attrbkd', 'llmbkd']:
            out_dir = os.path.join('./logs', poison_dataset, victim_model, 'attack/', poisoner, llm, style, flt,
                                   str(rs), str(poison_rate))
        else:
            out_dir = os.path.join('./logs', poison_dataset, victim_model, 'attack/', poisoner, flt, str(rs),
                                                                                                     str(poison_rate))
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, 'log.txt'), 'w') as sys.stdout:
            sys.stdout.write(str(display_result))

