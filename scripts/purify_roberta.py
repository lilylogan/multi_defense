'''
up-to-date 09.2024
modified purification script -- roberta
'''



import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import json
import random
import sys
# sys.path.insert(0, '../')
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import argparse

torch.cuda.set_device(0)
import pandas as pd
from tqdm import tqdm
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger
from openbackdoor.utils.visualize import display_results
import time


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


device = torch.device('cuda')


def main(config):

    set_random_seed(2024)
    target_label = config['attacker']["poisoner"]["target_label"]
    dataset = config['attacker']["train"]["data"]
    attack_type = config["attacker"]["poisoner"]['name']

    # load backdoored model
    victim = load_victim(config["victim"])
    victim.device = device
    victim.to(device)
    # print(victim)


    # load victim model weights
    data = config["target_dataset"]["name"]
    victim_model = config["victim"]["model"]

    poison_rate = config["attacker"]["poisoner"]["poison_rate"]
    label_consistency = config["attacker"]["poisoner"]["label_consistency"]
    if attack_type in ["attrbkd", "llmbkd"]:
        llm = config["attacker"]["poisoner"]["llm"]
        style = config["attacker"]["poisoner"]["style"]
        output_dir = os.path.join('purify_logs', data, victim_model, attack_type, llm, style)
    else:
        output_dir = os.path.join('purify_logs', data, victim_model, attack_type)



    os.makedirs(output_dir, exist_ok=True)
    log_file = open(os.path.join(output_dir, 'log.txt'), 'w')
    # Redirect stdout to the log file
    sys.stdout = log_file

    if attack_type == "attrbkd":
        base_dir = os.path.join("../../attrbkd/bkd/models", data, victim_model, "clean/attack",
                                "llmbkd", llm, style, "filter", str(poison_rate)) # todo: modify the directory such that it reads from 'attrbkd' folder
    elif attack_type == "llmbkd":
        base_dir = os.path.join("../../attrbkd/bkd/models", data, victim_model, "clean/attack", attack_type, llm,
                                style,
                                "filter", str(poison_rate))
    else:
        base_dir = os.path.join("../../attrbkd/bkd/models", data, victim_model, "clean/attack/", attack_type, "filter", str(poison_rate))


    sub_dirs = os.listdir(base_dir)
    model_dir = max(sub_dirs)
    loaded_backdoored_model_params = torch.load(os.path.join(base_dir, model_dir, 'best.ckpt'))
    victim.load_state_dict(loaded_backdoored_model_params)
    victim.eval()


    def process(batch):
        text = batch["text"]
        labels = batch["label"]
        input_batch = victim.tokenizer.batch_encode_plus(
            text,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )
        labels = labels.to(device)
        return input_batch, labels


    def Security_inference(Net_, set_, target_label):
        dataloader = get_dataloader(set_, batch_size=32, shuffle=True)
        with torch.no_grad():
            predict = []
            gt_label = []
            ps_label = []
            tgt_label = []
            for step_, batch_ in tqdm(enumerate(dataloader)):
                # print(step_)
                batch_inputs_, batch_labels_ = Net_.model.process(batch_)
                score_ = Net_(batch_inputs_)
                # print(score_)
                _, pred = torch.max(score_, dim=1)
                if pred.shape[0] == 1:
                    predict.append(pred.detach().cpu().item())
                else:
                    predict.extend(pred.squeeze().detach().cpu().numpy().tolist())
                gt_label.extend(batch_["label"])
                ps_label.extend(batch_["poison_label"])
                tgt_label.extend(len(pred) * [target_label])

            return accuracy_score(gt_label, predict), accuracy_score(tgt_label, predict)

    # load clean dev set
    clean_dev_data = []
    if attack_type in ["attrbkd", "llmbkd"]:
        clean_dev_set_path = os.path.join('../meta_poison_data/', llm, data, str(target_label),
                                      attack_type, style, 'filter', 'dev-clean.csv')
    else:
        clean_dev_set_path = os.path.join('../meta_poison_data/baselines', data, str(target_label),
                                          attack_type, 'filter', 'dev-clean.csv')
    benign_texts = pd.read_csv(clean_dev_set_path)

    dev_total = 0
    dev_correct = 0
    with torch.no_grad():
        for _, t, l, p in tqdm(benign_texts.values):
            input_tensor = victim.tokenizer.encode(t, add_special_tokens=True)
            input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)
            outputs = victim.plm(input_tensor)
            predict_labels = outputs.logits.squeeze().argmax()
            dev_total += 1
            if predict_labels == l:
                dev_correct += 1
                clean_dev_data.append([t, l, p])

    # dev_acc = 100. * (dev_correct / dev_total)
    dev_acc = 100.

    class CleanNet(nn.Module):
        def __init__(self, model_, num_layers):
            super(CleanNet, self).__init__()
            self.model = model_
            self.num_layers = num_layers
            self.up_bound = torch.ones([num_layers, 768]).to(device)
            self.margin = torch.ones([num_layers, 768]).to(device)

            self.up_bound.requires_grad = True
            self.margin.requires_grad = True

        def bound_init(self):
            self.up_bound = torch.ones([self.num_layers, 768]).to(device)
            self.margin = torch.ones([self.num_layers, 768]).to(device)

            self.up_bound.requires_grad = True
            self.margin.requires_grad = True

        def forward(self, x):

            self.low_bound = self.up_bound - torch.exp(self.margin)
            input_ids = x['input_ids'].to(device)
            attention_mask = x['attention_mask'].to(device)
            input_shape = input_ids.size()

            extended_attention_mask = self.model.plm.roberta.get_extended_attention_mask(
                attention_mask, input_shape, device=device)

            out = self.model.plm.roberta.embeddings(input_ids)

            for k in range(12):  # RoBERTa also has 12 layers (or adjust if different)
                out = self.model.plm.roberta.encoder.layer[k](out, attention_mask=extended_attention_mask)[0]

                if k < self.num_layers:
                    out_ = out.clone().to(device)
                    up_clip = torch.min(out_, self.up_bound[k])
                    out_clip = torch.max(up_clip, self.low_bound[k])
                    out[attention_mask.bool()] = out_clip[attention_mask.bool()]
                    out = out.contiguous()

            out = self.model.plm.classifier(out)

            return out

    Num_layers = 12
    Clean_Net = CleanNet(victim, num_layers=Num_layers).to(device)
    batch_size = 32


    random.shuffle(clean_dev_data)
    split_idx = int(len(clean_dev_data) * 0.8)
    clean_dev_train = clean_dev_data[:split_idx]
    clean_dev_test = clean_dev_data[split_idx:]
    trainloader = get_dataloader(clean_dev_train, batch_size, shuffle=True)

    acc, _ = Security_inference(Clean_Net, clean_dev_test, target_label)
    print('Initial Net Acc: {:.3f} '.format(acc * 100.))

    def learning_bound(c, a):
        acc_after = 0.
        optimizer = torch.optim.Adam([Clean_Net.up_bound, Clean_Net.margin], lr=0.01)
        mse = nn.MSELoss()

        for epoch in range(50):
            for step, batch in enumerate(tqdm(trainloader,
                                              desc=f"Learning Bound - Epoch {epoch}")):
                optimizer.zero_grad()
                batch_inputs, batch_labels = Clean_Net.model.process(batch)

                ref_out = Clean_Net.model(batch_inputs).logits
                outputs = Clean_Net(batch_inputs)

                loss1 = mse(outputs, ref_out)
                loss2 = torch.norm(torch.exp(Clean_Net.margin))
                loss = loss1 + c * loss2
                loss.backward()
                optimizer.step()

                # print('loss {:.4f}   norm  {:.4f}'.format(loss1.item(), loss2.item()))

            print("Epoch {} - Evaluate bounds on clean-dev-test set".format(epoch))
            acc_after, _ = Security_inference(Clean_Net, clean_dev_test, target_label)
            acc_after = acc_after * 100.

            if epoch > 10 and epoch % 5 == 0:
                if acc_after >= dev_acc * 0.98:
                    c *= a
                else:
                    c /= a

        print('Acc: %.3f' % acc_after)

        return acc_after

    c = 0.1
    a = 1.2
    dev_cacc = learning_bound(c, a)

    print('Dev CACC before learning: %.3f' % dev_cacc)
    while dev_cacc < 90:#97
        c = c / 2
        Clean_Net.bound_init()
        dev_cacc = learning_bound(c, a)

    if attack_type in ["llmbkd", "attrbkd"]:
        folder_name = os.path.join('./bounds', str(dataset), str(victim_model), str(attack_type), str(style))
    else:
        folder_name = os.path.join('./bounds', str(dataset), str(victim_model), str(attack_type))

    os.makedirs(folder_name, exist_ok=True)
    torch.save(Clean_Net.up_bound.detach().cpu(), folder_name + 'up_bound.pt')
    torch.save(Clean_Net.margin.detach().cpu(), folder_name + 'margin.pt')

    # load clean test set
    # print("\nLoading clean test set ...")
    poison_gt = []
    clean_test_set = []


    if attack_type in ["attrbkd", "llmbkd"]:
        clean_test_path = os.path.join('../meta_poison_data/', llm, data, str(target_label),
                                      attack_type, style, 'filter', 'test-clean.csv')
    else:
        clean_test_path = os.path.join('../meta_poison_data/baselines', data, str(target_label),
                                          attack_type, 'filter', 'test-clean.csv')

    clean_test_texts = pd.read_csv(clean_test_path)
    for _, t, l, _ in clean_test_texts.values:
        clean_test_set.append([t, l, target_label])
        if dataset != 'sst-2':
            if l != target_label:
                poison_gt.append(l)
        else:
            poison_gt.append(l)

    # load poison test set
    # print("\nLoading poison test set ...")
    poison_test_set = []
    if attack_type in ["attrbkd", "llmbkd"]:
        poison_test_path = os.path.join('../meta_poison_data/', llm, data, str(target_label),
                                      attack_type, style, 'test-poison.csv')
    else:
        poison_test_path = os.path.join('../meta_poison_data/baselines', data, str(target_label),
                                          attack_type, 'filter', 'test-poison.csv')
    poison_texts = pd.read_csv(poison_test_path)
    poison_texts = poison_texts.dropna()

    if label_consistency is True:
        for i, (_, t, l, ps) in enumerate(poison_texts.values):
            # print("\n\n------- t, poison_gt[i], target_label---", t, poison_gt[i], target_label)
            poison_test_set.append([t, l, ps])
    else:
        for i, (_, t, l, _) in enumerate(poison_texts.values):
            # print("\n\n------- t, poison_gt[i], target_label---", t, poison_gt[i], target_label)
            poison_test_set.append([t, poison_gt[i], target_label])

    print("Inference on clean-test...")
    Clean_ACC, _ = Security_inference(Clean_Net, clean_test_set, target_label)
    print("Inference on poison-test...")
    Poison_ACC, Poison_ASR = Security_inference(Clean_Net, poison_test_set, target_label)
    if attack_type in ['llmbkd', 'attrbkd']:
        print('Dataset: ' + str(dataset) + '  Attack: ' + str(attack_type) + '  Variant: ' + str(style))
    else:
        print('Dataset: ' + str(dataset) + '  Attack: ' + str(attack_type))
    print('Clean ACC {:.3f}  Posion ACC {:.3f}  ASR {:.3f} '.format(Clean_ACC, Poison_ACC,
                                                                    Poison_ASR))


    log_file.close()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/lws_config.json')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config_path = args.config_path
    with open(config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    main(config)