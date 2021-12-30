import argparse
import logging
from typing import Any, List, Optional
from datetime import datetime
import datasets
import torch
from torch.nn import (
    CrossEntropyLoss,
)
import os
from pytorch_lightning import (
    LightningModule,
)
from transformers import (
    AdamW,
    get_cosine_schedule_with_warmup,
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
)
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import plotly.express as px
import plotly

from utils import (
    TYPE_TO_CONFIG,
    TYPE_TO_MODEL,
    TYPE_TO_SPECIAL_TOKENS_MAP,
    get_summary_loss_with_gpt
)


class SummaryModule(LightningModule):
    def __init__(self,
                 model_name_or_path: str = 'skt/kogpt2-base-v2',
                 model_type: str = 'gpt',
                 max_context_seq_len: int = 512,
                 max_label_seq_len: int = 30,
                 learning_rate: float = 2e-5,
                 adam_epsilon: float = 1e-8,
                 warmup_ratio: float = 0.01,
                 weight_decay: float = 0.01,
                 num_workers: int = 1,
                 top_k: int = 30,
                 top_p: float = 0.92,
                 temperature: float = 0.7,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model_type = self.hparams.model_type
        self.config = TYPE_TO_CONFIG[self.model_type].from_pretrained(
            self.hparams.model_name_or_path)
        if self.model_type == 'gpt':
            self.loss_fct = CrossEntropyLoss()
            self.config.max_length = self.hparams.max_context_seq_len + \
                self.hparams.max_label_seq_len + 3
        self.model = TYPE_TO_MODEL[self.model_type].from_pretrained(
            self.hparams.model_name_or_path, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.model_name_or_path,
            use_fast=True,
            **TYPE_TO_SPECIAL_TOKENS_MAP[self.model_type])
        self.metric = datasets.load_metric(
            "rouge", experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        self.pad_token_id = self.tokenizer.pad_token_id

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        '''
        TODO: delete code getting argument from cli
        '''
        parser.add_argument('--model_name_or_path', type=str,
                            default='hyunwoongko/kobart')
        parser.add_argument('--model_type', type=str, default='bart')
        parser.add_argument('--max_context_seq_len', type=int, default=512)
        parser.add_argument('--max_label_seq_len', type=int, default=30)
        parser.add_argument('--learning_rate', type=float, default=2e-5)
        parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        parser.add_argument('--warmup_ratio', type=float, default=0.01)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--top_k', type=int, default=30)
        parser.add_argument('--top_p', type=float, default=0.92)
        parser.add_argument('--temperature', type=float, default=0.7)

        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate, correct_bias=False)
        num_workers = self.hparams.num_workers
        data_len = self.hparams.total_train
        logging.info(
            f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(
            data_len / (self.hparams.train_batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )
        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def forward(self, **inputs):
        inputs['attention_mask'] = inputs['input_ids'].ne(
            self.pad_token_id).float()
        if hasattr(inputs, 'decoder_input_ids'):
            inputs['decoder_attention_mask'] = inputs['decoder_input_ids'].ne(
                self.pad_token_id).float()
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        if self.model_type == 'bart':
            outs = self(**batch)
            loss = outs.loss
        elif self.model_type == 'gpt':
            loss = get_summary_loss_with_gpt(self, batch)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self._shared_eval_end(outputs)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        predictions = self.model.generate(
            input_ids=batch['input_ids'],
            max_length=self.config.max_length if self.model_type == 'gpt' else self.hparams.max_label_seq_len,
            use_cache=True,
            **self.hparams
        )
        predictions = [self.tokenizer.decode(
            pred, skip_special_tokens=True) for pred in predictions.cpu().numpy().tolist()]
        return predictions

    def predict_epoch_end(self, results: List[Any]) -> None:
        results = list(itertools.chain(*results))
        return results

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self._shared_eval_end(outputs)

    def _shared_eval_step(self, batch, batch_idx):
        if self.model_type == 'gpt':
            loss = get_summary_loss_with_gpt(self, batch)
            batch['input_ids'] = batch['input_ids'][:,
                                                    :self.hparams.max_context_seq_len + 2]
        else:
            outs = self(**batch)
            loss = outs['loss']
        prediction = self.model.generate(
            input_ids=batch['input_ids'],
            max_length=self.config.max_length if self.model_type == 'gpt' else self.hparams.max_label_seq_len,
            use_cache=True,
            # sampling parameter
            top_p=0.92,
            top_k=30,
            temperature=0.7)
        prediction = prediction[:, self.hparams.max_context_seq_len +
                                2:] if self.model_type == 'gpt' else prediction
        label = batch['labels']

        predictions = prediction.cpu().numpy().tolist()
        labels = label.cpu().numpy().tolist()

        return (loss, predictions, labels)

    def _shared_eval_end(self, outputs):
        losses, predictions, labels = [], [], []
        for loss, prediction, label in outputs:
            losses.append(loss)
            predictions.extend(prediction)
            labels.extend(label)

        self.metric.add_batch(predictions=predictions, references=labels)
        score = self.metric.compute()

        self.log("rouge_l", score['rouge1'].mid.fmeasure)
        self.log("rouge_1", score['rouge2'].mid.fmeasure)
        self.log("rouge_2", score['rougeL'].mid.fmeasure)
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)

        pred_texts = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True)

        labels = [list(filter(lambda x: x != -100, label)) for label in labels]
        label_texts = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        prediction_path = os.path.join(self.trainer.log_dir, 'predictions.csv')
        df = pd.DataFrame(list(zip(pred_texts, label_texts)),
                          columns=[['predictions', 'labels']])
        df.to_csv(prediction_path)
        label_set = unique_labels(df['labels'], df['predictions'])
        cm = confusion_matrix(
            y_true=df['labels'], y_pred=df['predictions'], labels=label_set)
        fig = px.imshow(
            cm,
            x=label_set,
            y=label_set,
            color_continuous_scale='deep',
            labels={
                'x': 'predictions',
                'y': 'labels',
                'color': 'count'
            }
        )
        confusion_matrix_path = os.path.join(
            self.trainer.log_dir, 'confusion_matrix.html')
        plotly.offline.plot(fig, filename=confusion_matrix_path)
