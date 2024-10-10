'''
trainer w/ visualization for studying defense
'''

from openbackdoor.victims import Victim
from .trainer import Trainer
from openbackdoor.utils import logger, evaluate_classification
from openbackdoor.data import wrap_dataset, get_dataloader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from typing import *
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from umap import UMAP
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from collections import defaultdict


#
# def collate_fn(data):
#     texts = []
#     labels = []
#     poison_labels = []
#     style_labels = []
#     for text, label, poison_label, style_label in data:
#         texts.append(text)
#         labels.append(label)
#         poison_labels.append(poison_label)
#         style_labels.append(style_label)
#     labels = torch.LongTensor(labels)
#     batch = {
#         "text": texts,
#         "label": labels,
#         "poison_label": poison_labels,
#         "style_label": style_labels
#     }
#     return batch
#
# def get_dataloader(dataset: Union[Dataset, List],
#                     batch_size: Optional[int] = 4,
#                     shuffle: Optional[bool] = True):
#     return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
#
# def wrap_dataset(dataset: dict, batch_size: Optional[int] = 4,):
#     r"""
#     convert dataset (Dict[List]) to dataloader
#     """
#     dataloader = defaultdict(list)
#     for key in dataset.keys():
#         dataloader[key] = get_dataloader(dataset[key], batch_size=batch_size)
#     return dataloader


class VisualTrainer(Trainer):
    r"""
    Basic clean trainer with visualization for clean and poison learning.

    Args:
        name (:obj:`str`, optional): name of the trainer. Default to "Base".
        lr (:obj:`float`, optional): learning rate. Default to 2e-5.
        weight_decay (:obj:`float`, optional): weight decay. Default to 0.
        epochs (:obj:`int`, optional): number of epochs. Default to 10.
        batch_size (:obj:`int`, optional): batch size. Default to 4.
        gradient_accumulation_steps (:obj:`int`, optional): gradient accumulation steps. Default to 1.
        max_grad_norm (:obj:`float`, optional): max gradient norm. Default to 1.0.
        warm_up_epochs (:obj:`int`, optional): warm up epochs. Default to 3.
        ckpt (:obj:`str`, optional): checkpoint name. Can be "best" or "last". Default to "best".
        save_path (:obj:`str`, optional): path to save the model. Default to "./models/checkpoints".
        loss_function (:obj:`str`, optional): loss function. Default to "ce".
        visualize (:obj:`bool`, optional): whether to visualize the hidden states. Default to False.
        poison_setting (:obj:`str`, optional): the poisoning setting. Default to mix.
        poison_method (:obj:`str`, optional): name of the poisoner. Default to "Base".
        poison_rate (:obj:`float`, optional): the poison rate. Default to 0.1.
        embed_model: the model used for learning embeddings for clustering
        model: victim model

    """

    def __init__(
            self,
            visualize: Optional[bool] = True,
            cls: Optional[str] = "all",
            visual_method: [str] = 'tsne',
            layer: Optional[int] = -1,
            save_path: Optional[str] = "./models",
            **kwargs):
        super().__init__(**kwargs)
        # new save path
        self.visualize = visualize
        self.cls = cls
        self.visual_method = visual_method
        self.layer = layer
        self.save_path = save_path
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H:%M:%S")
        base_model_dir = os.path.join(self.save_path, f'{self.data}', f'{self.model}', f'{self.poison_setting}')



        if self.poison_method in ['attrbkd', "llmbkd"]:
            if self.filter:
                self.save_path = os.path.join(base_model_dir, "visual/", f'{self.poison_method}', f'{self.llm}', f'{self.style}',
                                              'filter/', f'{self.poison_rate}', str(timestamp))
            else:
                self.save_path = os.path.join(base_model_dir, "visual/", f'{self.poison_method}', f'{self.llm}', f'{self.style}',
                                              'nofilter/', f'{self.poison_rate}', str(timestamp))
            os.makedirs(self.save_path, exist_ok=True)
        else:
            if self.filter:
                self.save_path = os.path.join(base_model_dir, "visual/", f'{self.poison_method}', 'filter/',
                                              f'{self.poison_rate}', str(timestamp))

            else:
                self.save_path = os.path.join(base_model_dir, "visual/", f'{self.poison_method}', 'nofilter/',
                                              f'{self.poison_rate}', str(timestamp))

            os.makedirs(self.save_path, exist_ok=True)

        self.COLOR = ['royalblue', 'red', 'palegreen', 'violet', 'deepskyblue',
                      'green', 'mediumpurple', 'gold', 'paleturquoise']
        self.COLOR_ADD = ['lightcoral', 'khaki', 'mediumaquamarine', 'lightsteelblue', 'cornflowerblue',
                      'orchid']

        self.loss_function = nn.CrossEntropyLoss(reduction="none")

        #
        # for attr, value in self.__dict__.items():
        #     print(f"{attr}: {value}")
        # exit(0)


    def register(self, model: Victim, dataloader, metrics):
        r"""
        Register model, dataloader and optimizer
        """
        self.model = model
        self.metrics = metrics
        self.main_metric = self.metrics[0]
        self.split_names = dataloader.keys()
        self.model.train()
        self.model.zero_grad()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        train_length = len(dataloader["train"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=self.warm_up_epochs * train_length,
                                                         num_training_steps=self.epochs * train_length)

        self.poison_loss_all = []
        self.normal_loss_all = []
        if self.visualize:
            poison_loss_before_tuning, normal_loss_before_tuning = self.comp_loss(model, dataloader["train"])
            self.poison_loss_all.append(poison_loss_before_tuning)
            self.normal_loss_all.append(normal_loss_before_tuning)
            self.hidden_states, self.labels, self.poison_labels = self.compute_hidden(model, dataloader["train"],
                                                                                      layer=self.layer)

        # Train
        logger.info("***** Training *****")
        logger.info("  Num Epochs = %d", self.epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d\n", self.epochs * train_length)

    def train_one_epoch(self, epoch: int, epoch_iterator):
        """
        Train one epoch function.

        Args:
            epoch (:obj:`int`): current epoch.
            epoch_iterator (:obj:`torch.utils.data.DataLoader`): dataloader for training.

        Returns:
            :obj:`float`: average loss of the epoch.
        """
        self.model.train()
        total_loss = 0
        poison_loss_list, normal_loss_list = [], []
        for step, batch in enumerate(epoch_iterator):
            batch_inputs, batch_labels = self.model.process(batch)
            output = self.model(batch_inputs)
            logits = output.logits
            loss = self.loss_function(logits, batch_labels)

            if self.visualize:
                poison_labels = batch["poison_label"]
                for l, poison_label in zip(loss, poison_labels):
                    if poison_label == 1:
                        poison_loss_list.append(l.item())
                    else:
                        normal_loss_list.append(l.item())
                loss = loss.mean()

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
                self.model.zero_grad()

        avg_loss = total_loss / len(epoch_iterator)
        avg_poison_loss = sum(poison_loss_list) / len(poison_loss_list) if self.visualize else 0
        avg_normal_loss = sum(normal_loss_list) / len(normal_loss_list) if self.visualize else 0

        return avg_loss, avg_poison_loss, avg_normal_loss

    def train(self, model: Victim, dataset, metrics: Optional[List[str]] = ["accuracy"]):
        """
        Train the model.

        Args:
            model (:obj:`Victim`): victim model.
            dataset (:obj:`Dict`): dataset.
            metrics (:obj:`List[str]`, optional): list of metrics. Default to ["accuracy"].
        Returns:
            :obj:`Victim`: trained model.
        """

        dataloader = wrap_dataset(dataset, self.batch_size)

        train_dataloader = dataloader["train"]
        eval_dataloader = {}
        for key, item in dataloader.items():
            if key.split("-")[0] == "dev":
                eval_dataloader[key] = dataloader[key]
        self.register(model, dataloader, metrics)

        best_dev_score = 0

        for epoch in range(self.epochs):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            epoch_loss, poison_loss, normal_loss = self.train_one_epoch(epoch, epoch_iterator)
            self.poison_loss_all.append(poison_loss)
            self.normal_loss_all.append(normal_loss)
            logger.info('')
            logger.info('Epoch: {}, avg loss: {}'.format(epoch + 1, epoch_loss))
            dev_results, dev_score = self.evaluate(self.model, eval_dataloader, self.metrics)

            if self.visualize:
                hidden_state, labels, poison_labels = self.compute_hidden(model, epoch_iterator, layer=self.layer)
                self.hidden_states.extend(hidden_state)
                self.labels.extend(labels)
                self.poison_labels.extend(poison_labels)

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                if self.ckpt == 'best':
                    torch.save(self.model.state_dict(), self.model_checkpoint(self.ckpt))
                    logger.info("Best model saved. Dev Acc. -- {}".format(best_dev_score))

        if self.visualize:
            self.save_vis()

        if self.ckpt == 'last':
            torch.save(self.model.state_dict(), self.model_checkpoint(self.ckpt))

        logger.info("Training finished.")
        state_dict = torch.load(self.model_checkpoint(self.ckpt))
        self.model.load_state_dict(state_dict)
        # test_score = self.evaluate_all("test")
        return self.model

    def evaluate(self, model, eval_dataloader, metrics):
        """
        Evaluate the model.

        Args:
            model (:obj:`Victim`): victim model.
            eval_dataloader (:obj:`torch.utils.data.DataLoader`): dataloader for evaluation.
            metrics (:obj:`List[str]`, optional): list of metrics. Default to ["accuracy"].

        Returns:
            results (:obj:`Dict`): evaluation results.
            dev_score (:obj:`float`): dev score.
        """
        results, dev_score = evaluate_classification(model, eval_dataloader, metrics)
        return results, dev_score

    def compute_hidden(self, model: Victim, dataloader, layer: Optional[int] = -1):
        """
        Prepare the hidden states, ground-truth labels, and poison_labels of the dataset for visualization.

        Args:
            model (:obj:`Victim`): victim model.
            dataloader (:obj:`torch.utils.data.DataLoader`): non-shuffled dataloader for train set.
            layers (:list:): the layers where we get the hidden states

        Returns:
            hidden_state (:obj:`List`): hidden state of the training data.
            labels (:obj:`List`): ground-truth label of the training data.
            poison_labels (:obj:`List`): poison label of the poisoned training data.

        """
        logger.info('***** Computing hidden hidden_state *****')
        model.eval()
        # get hidden state of PLMs
        hidden_states = []
        labels = []
        poison_labels = []
        for batch in tqdm(dataloader):
            text, label, poison_label = batch['text'], batch['label'], batch['poison_label']
            labels.extend(label)
            poison_labels.extend(poison_label)
            batch_inputs, _ = model.process(batch)
            output = model(batch_inputs)
            hidden_state = output.hidden_states[layer]  # we only use the hidden state of the last layer
            try:  # bert
                pooler_output = getattr(model.plm, model.model_name.split('-')[0]).pooler(hidden_state)
            except:  # RobertaForSequenceClassification has no pooler
                dropout = model.plm.classifier.dropout
                dense = model.plm.classifier.dense
                try:
                    activation = model.plm.activation
                except:
                    activation = torch.nn.Tanh()
                pooler_output = activation(dense(dropout(hidden_state[:, 0, :])))
            hidden_states.extend(pooler_output.detach().cpu().tolist())
        model.train()
        return hidden_states, labels, poison_labels

    def visualization(self, hidden_states: List, labels: List, poison_labels: List,
                      fig_basepath: Optional[str] = "./clustering/visualization", fig_title: Optional[str] =
                      "vis"):
        """
        Visualize the latent representation of the victim model on the poisoned dataset and save to 'fig_basepath'.

        Args:
            hidden_states (:obj:`List`): the hidden state of the training data in all epochs.
            labels (:obj:`List`): ground-truth label of the training data.
            poison_labels (:obj:`List`): poison label of the poisoned training data.
            fig_basepath (:obj:`str`, optional): dir path to save the model. Default to "./visualization".
            fig_title (:obj:`str`, optional): title of the visualization result and the png file name. Default to "vis".
        """
        logger.info('***** Visualizing *****')

        dataset_len = int(len(poison_labels) / (self.epochs + 1))

        hidden_states = np.array(hidden_states)
        labels = np.array(labels)
        poison_labels = np.array(poison_labels, dtype=np.int64)

        # num_classes = len(set(labels))
        classes = set(labels)
        logger.info('No. of classes = {}'.format(classes))

        for epoch in tqdm(range(self.epochs + 1)):
            fig_title = f'Epoch {epoch}'

            hidden_state = hidden_states[epoch * dataset_len: (epoch + 1) * dataset_len]
            label = labels[epoch * dataset_len: (epoch + 1) * dataset_len]
            poison_label = poison_labels[epoch * dataset_len: (epoch + 1) * dataset_len]
            poison_idx = np.where(poison_label == np.ones_like(poison_label))[0]

            embedding_umap = self.dimension_reduction(hidden_state)
            embedding = pd.DataFrame(embedding_umap)

            # plot clean data
            for c in classes:
                idx = np.where(label == int(c) * np.ones_like(label))[0]
                # print("**** clean data label - {} in visualization function - {} *** ".format(c, list(set(idx))))
                idx = list(set(idx) ^ set(poison_idx))
                plt.scatter(embedding.iloc[idx, 0], embedding.iloc[idx, 1], c=self.COLOR_ADD[c], s=1, label=c)

            # plot poison data
            plt.scatter(embedding.iloc[poison_idx, 0], embedding.iloc[poison_idx, 1], s=1, c='gray', label='poison')

            plt.tick_params(labelsize='large', length=2)
            plt.legend(fontsize=8, markerscale=4, loc='lower right')
            os.makedirs(fig_basepath, exist_ok=True)

            # save visual cluster plots
            plt.savefig(os.path.join(fig_basepath, f'{fig_title}.pdf'))
            fig_path = os.path.join(fig_basepath, f'{fig_title}.pdf')
            logger.info(f'Saving clustering pdf to {fig_path}')
            plt.close()
        return embedding_umap

    def dimension_reduction(self, hidden_states: List,
                            components: Optional[int] = 20,
                            n_neighbors: Optional[int] = 100,
                            min_dist: Optional[float] = 0.5,
                            umap_components: Optional[int] = 2):

        if self.visual_method == "pca":
            reduction = PCA(n_components=components,
                      random_state=42)

        elif self.visual_method == 'kernel_pca':
            reduction = KernelPCA(n_components=components,
                                  kernel="rbf",
                                  random_state=42)

        elif self.visual_method == 'tsne':
            reduction = TSNE(n_components=2,
                             init='random',
                             random_state=42)

        umap = UMAP(n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=umap_components,
                    random_state=42,
                    transform_seed=42,
                    )

        embedding_reduction = reduction.fit_transform(hidden_states)
        embedding_umap = umap.fit(embedding_reduction).embedding_
        return embedding_umap


    def comp_loss(self, model: Victim, dataloader: torch.utils.data.DataLoader):
        poison_loss_list, normal_loss_list = [], []
        for step, batch in enumerate(dataloader):
            batch_inputs, batch_labels = self.model.process(batch)
            output = self.model(batch_inputs)
            logits = output.logits
            loss = self.loss_function(logits, batch_labels)

            poison_labels = batch["poison_label"]
            for l, poison_label in zip(loss, poison_labels):
                if poison_label == 1:
                    poison_loss_list.append(l.item())
                else:
                    normal_loss_list.append(l.item())

        avg_poison_loss = sum(poison_loss_list) / len(poison_loss_list) if self.visualize else 0
        avg_normal_loss = sum(normal_loss_list) / len(normal_loss_list) if self.visualize else 0

        return avg_poison_loss, avg_normal_loss

    def plot_curve(self, normal_loss, poison_loss,
                   fig_basepath: Optional[str] = "./clustering/learning_curve", fig_title: Optional[str] = "learning curve"):

        # bar of db score
        fig, ax1 = plt.subplots()

        # ax1.bar(range(self.epochs + 1), davies_bouldin_scores, width=0.5, color='royalblue',
        #         label='davies bouldin score')
        ax1.set_xlabel('Epoch')
        # ax1.set_ylabel('Davies Bouldin Score', size=14)

        # curve of loss
        # ax2 = ax1.twinx()
        ax2 = ax1
        ax2.plot(range(self.epochs + 1), normal_loss, linewidth=2.5, color='green',
                 label='Clean Loss')
        ax2.plot(range(self.epochs + 1), poison_loss, linewidth=2.5, color='orange',
                 label='Poison Loss')
        ax2.set_ylabel('Loss', size=14)
        ax2.legend()
        plt.title('Learning Curve', size=14)
        os.makedirs(fig_basepath, exist_ok=True)
        plt.savefig(os.path.join(fig_basepath, f'{fig_title}.pdf'))
        fig_path = os.path.join(fig_basepath, f'{fig_title}.pdf')
        logger.info(f'Saving learning curve pdf to {fig_path}')
        plt.close()

    def save_vis(self):
        # save visualization plots
        if self.poison_method in ["llmbkd", "attrbkd"]:
            fig_path = os.path.join('./visual/', 'visualization',
                                                                 self.data,
                                                                 'roberta', self.poison_method,
                                                                 self.llm,
                                                                 self.style, str(self.poison_rate), str(self.layer))

            # save learning curves
            curve_path = os.path.join('./visual/', 'learning_curve', self.data,
                                      'roberta', self.poison_method,
                                      self.llm, self.style, str(self.poison_rate), str(self.layer))

        elif self.poison_method not in ["llmbkd", "attrbkd"]:
            fig_path = os.path.join('./visual/', 'visualization',
                                    self.data,
                                    'roberta/baselines', self.poison_method,
                                    str(self.poison_rate), str(self.layer))

            # save learning curves
            curve_path = os.path.join('./visual/', 'learning_curve', self.data,
                                      'roberta/baselines', self.poison_method,
                                      str(self.poison_rate), str(self.layer))

        self.visualization(self.hidden_states, self.labels, self.poison_labels,
                                       fig_basepath=fig_path)
        os.makedirs(curve_path, exist_ok=True)
        # davies_bouldin_scores = self.clustering_metric(self.hidden_states, self.poison_labels, curve_path)

        np.save(os.path.join(curve_path, 'poison_loss.npy'), np.array(self.poison_loss_all))
        np.save(os.path.join(curve_path, 'normal_loss.npy'), np.array(self.normal_loss_all))

        self.plot_curve(self.normal_loss_all, self.poison_loss_all,
                        fig_basepath=curve_path)

    def model_checkpoint(self, ckpt: str):
        return os.path.join(self.save_path, f'{ckpt}.ckpt')

