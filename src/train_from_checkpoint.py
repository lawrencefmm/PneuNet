from cnn import PneuNet
from data_module import XRayDataModule
from lightning.pytorch import Trainer
import yaml
import argparse
from pathlib import Path
import torch

def __main__(model_path):
    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)["model"]


    model_path = Path(model_path)

    checkpoint = [ckpt for ckpt in model_path.glob("**/*.ckpt")][0]
    print(checkpoint, model_path)
    checkpoint_path = str(checkpoint)

    hparams = [hp for hp in model_path.glob("**/hparams.yaml")][0]
    hparams_path = str(hparams)

    torch.set_float32_matmul_precision("medium")

    model = PneuNet.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hparams_file=hparams_path,
        map_location=None,
        lr=1e-5
    )

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    max_epochs = config["num_epochs"]
    num_classes = config["num_classes"]
    weight_decay = config["weight_decay"]

    data_module = XRayDataModule(data_dir="./chest_xray", batch_size=batch_size, num_workers=num_workers)


    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        check_val_every_n_epoch=2,
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue train from a checkpoint")
    parser.add_argument("model_path", type=str, help="Path to the model directory.")
    args = parser.parse_args()
    output = __main__(args.model_path)
    __main__()