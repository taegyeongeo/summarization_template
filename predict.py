import logging
from argparse import ArgumentParser
from pytorch_lightning.trainer.trainer import Trainer
from dataset import (
    SummaryDataModule,
)
from utils import (
    TYPE_TO_MODEL,
    CustomWriter,
)


def main(args):
    model = TYPE_TO_MODEL[args.model_type].load_from_checkpoint(
        args.checkpoint_path)
    model_args = {k: v for k, v in model.hparams.items()
                  if k not in vars(args)}
    vars(args).update(model_args)
    prediction_writer = CustomWriter(
        output_dir=args.output_dir,
        write_interval='epoch'
    )
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=[prediction_writer]
    )
    dm = SummaryDataModule(args)
    dm.prepare_data()
    dm.setup('predict')
    trainer.predict(
        model=model,
        dataloaders=dm
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = SummaryDataModule.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--checkpoint_path', type=str,
                        default='lightning_logs/version_2/checkpoints/epoch=0-step=287.ckpt')
    parser.add_argument('--model_type', type=str, default='kobart')
    parser.add_argument('--output_dir', type=str, default='lightning_logs')
    args = parser.parse_args()
    logging.info(args)
    main(args)
