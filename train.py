from pytorch_lightning import Trainer
from pytorch_lightning import loggers
from pytorch_lightning.trainer import trainer
from model import (
    SummaryModule,
)
from dataset import (
    SummaryDataModule,
)
import logging
from argparse import ArgumentParser
import hydra
from omegaconf import OmegaConf, DictConfig
from utils import print_config
from pytorch_lightning.loggers import TensorBoardLogger


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print_config(cfg)

    logger = hydra.utils.instantiate(cfg.logger)

    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.prepare_data()
    dm.setup('fit')

    trainer = hydra.utils.instantiate(
        cfg.trainer, logger=list(logger.values()))

    model = hydra.utils.instantiate(
        cfg.model,
        train_batch_size=dm.train_batch_size,
        max_epochs=trainer.max_epochs,
        total_train=dm.total_train)

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
