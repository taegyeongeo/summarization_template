from dataclasses import dataclass
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning import (
    LightningModule,
)
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerBase,
)
import os
import itertools
import pandas as pd
from typing import Any, List, Optional, Union
from transformers.file_utils import PaddingStrategy

import logging
import warnings
from typing import List, Sequence

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


LOADER_COLUMNS = {
    'fit': ["input_ids", "labels"],
    'test': ["input_ids", "labels"],
    'predict': ["input_ids"]
}


DATA_SPLIT_TEMPLATE = {
    # set validation percentage
    'fit': {'train': 'train[:90%]', 'valid': 'train[90%:]'},
    'test': {'valid': 'train'},
    'predict': {'valid': 'train'}
}

'''
for addtional model/tasks
'''

TPYE_TO_MODEL_HUB = {
    "bart": "hyunwoongko/kobart",
    "gpt": "~",
    "electra": "~"
}

TYPE_TO_MODEL = {
    "bart": BartForConditionalGeneration,
    "gpt": GPT2LMHeadModel,
    "electra": None
}

TYPE_TO_CONFIG = {
    "bart": BartConfig,
    "gpt": GPT2Config,
    "electra": None
}

TYPE_TO_SPECIAL_TOKENS_MAP = {
    "bart": {},
    "gpt": {
        'bos_token': '</s>',
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'pad_token': '<pad>',
        'mask_token': '<mask>'},
    "electra": {}
}


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval: str):
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_on_batch_end(
        self, trainer, pl_module: LightningModule, prediction: Any, batch_indices: List[int], batch: Any,
        batch_idx: int, dataloader_idx: int
    ):
        return prediction

    def write_on_epoch_end(
            self, trainer, pl_module: LightningModule, predictions: List[Any], batch_indices: List[Any]
    ):
        prediction_path = os.path.join(self.output_dir, 'predictions.csv')
        predictions = list(itertools.chain(*predictions[0]))
        pd.DataFrame(predictions, columns=[
            ['predictions']]).to_csv(prediction_path)


def get_summary_loss_with_gpt(gpt_module, batch):
    input_ids = batch['input_ids']
    labels = batch['labels']
    logits = gpt_module(input_ids=input_ids).logits
    shift_logits = logits[...,
                          gpt_module.hparams.max_context_seq_len:-1, :].contiguous()
    shift_labels = labels[...,
                          gpt_module.hparams.max_context_seq_len + 1:].contiguous()
    loss = gpt_module.loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "test_after_training",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)
