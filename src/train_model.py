from cnn import PneuNet
from data_module import XRayDataModule
from lightning.pytorch import Trainer
import yaml
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

def __main__():
    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)["train"]

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    max_epochs = config["num_epochs"]
    num_classes = config["num_classes"]
    weight_decay = config["weight_decay"]
    data_split_seed = config["data_split_seed"]

    lr = config["lr"]

    data_module = XRayDataModule(data_dir="./chest_xray", batch_size=batch_size, num_workers=num_workers, seed=data_split_seed)

    torch.set_float32_matmul_precision("high")

    model = PneuNet(num_classes=num_classes, weight_decay=weight_decay, batch_size=batch_size, lr=lr)

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        check_val_every_n_epoch=3,
        log_every_n_steps=1,
        enable_checkpointing=True,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min",
                verbose=True
            )
        ]
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    __main__()