import argparse
import glob
import os

from common.io.multi import MultiObject
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from pytorch_lightning import (
    LightningDataModule,
)
from torch.utils.data import DataLoader
from utils import (
    DATA_SPLIT_TEMPLATE,
    LOADER_COLUMNS,
    TYPE_TO_MODEL,
    TYPE_TO_CONFIG,
    TYPE_TO_SPECIAL_TOKENS_MAP,
)


class SummaryDataModule(LightningDataModule):
    def __init__(self,
                 model_name_or_path: str = 'hyunwoongko/kobart',
                 data_files_dir: str = '',
                 max_context_seq_len: int = 512,
                 max_label_seq_len: int = 30,
                 train_batch_size: int = 4,
                 eval_batch_size: int = 4,
                 num_processes: int = 1,
                 model_type: str = 'bart',
                 task: str = 'summarization',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.params = self.hparams
        self.model_name_or_path = self.params.model_name_or_path
        self.data_files_dir = self.params.data_files_dir
        self.max_context_seq_len = self.params.max_context_seq_len
        self.max_label_seq_len = self.params.max_label_seq_len
        self.train_batch_size = self.params.train_batch_size
        self.eval_batch_size = self.params.eval_batch_size
        self.num_processes = self.params.num_processes
        self.model_type = self.params.model_type
        self.task = self.params.task
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.params.model_name_or_path,
            use_fast=True,
            **TYPE_TO_SPECIAL_TOKENS_MAP[self.model_type]
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        '''
        TODO: delete code getting argument from cli
        '''
        parser.add_argument('--task', type=str, default='summarization')
        parser.add_argument('--data_files_dir', type=str,
                            default='data_dir')
        parser.add_argument('--train_batch_size', type=int, default=16)
        parser.add_argument('--eval_batch_size', type=int, default=64)
        return parser

    def setup(self, stage):
        file_list = glob.glob(os.path.join(self.cache_dir, '*.jsonl'))
        self.dataset = load_dataset(
            'json', data_files=file_list, split=DATA_SPLIT_TEMPLATE[stage])
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=False,
                remove_columns=[
                    c for c in self.dataset[split].column_names if c not in LOADER_COLUMNS[stage]],
                num_proc=self.num_processes
            )
            self.columns = [
                c for c in self.dataset[split].column_names if c in LOADER_COLUMNS[stage]]
            setattr(self, f'total_{split}', len(self.dataset[split]))

        if self.model_type == 'bart' and stage in ['fit', 'test']:
            self.collate_fn = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=TYPE_TO_MODEL[self.model_type].from_pretrained(
                    self.model_name_or_path),
            )
        else:
            self.collate_fn = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer
            )

    def prepare_data(self):
        '''
        TODO: delete code getting argument from cli
        '''
        file_obj = MultiObject(self.data_files_dir)
        file_obj.download()
        self.cache_dir = file_obj.local_path()

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], shuffle=True, batch_size=self.train_batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset['valid'], shuffle=False, batch_size=self.eval_batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dataset['valid'], shuffle=False, batch_size=self.eval_batch_size, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.dataset['valid'], shuffle=False, batch_size=self.eval_batch_size, collate_fn=self.collate_fn)

    def convert_to_features(self, example_batch, indices=None):
        '''
        dataset has to have two features
        - context: full text of dialogue or document
        - sumamry: abstractive or extractive sequence of summary
        '''
        texts = example_batch['context']
        if isinstance(texts, list):
            texts = ' '.join(texts)

        features = self.tokenizer.encode_plus(
            text=texts,
            max_length=self.max_context_seq_len,
            truncation='only_first',
            pad_to_max_length=True,
            add_special_tokens=False
        )

        if example_batch.get('summary') is not None:
            summary = example_batch['summary']
            labels = self.tokenizer.encode_plus(
                text=summary,
                max_length=self.max_label_seq_len,
                truncation='only_first',
                pad_to_max_length=True,
                add_special_tokens=False,
            )
            features['labels'] = labels['input_ids']

        if self.model_type == 'gpt':
            features = {k: v for k, v in features.items() if k in [
                'input_ids', 'labels']}
            features['input_ids'] = [self.tokenizer.bos_token_id] + features['input_ids'] + \
                [self.tokenizer.mask_token_id] + \
                features['labels'] + [self.tokenizer.eos_token_id]
            features['labels'] = [-100] * \
                (self.max_context_seq_len + 2) + \
                features['labels'] + [self.tokenizer.eos_token_id]
        return features
