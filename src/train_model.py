from cnn import PneuNet
from data_module import XRayDataModule
from lightning.pytorch import Trainer
import yaml

def __main__():
    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)["model"]

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    max_epochs = config["num_epochs"]
    num_classes = config["num_classes"]
    weight_decay = config["weight_decay"]

    data_module = XRayDataModule(data_dir="./chest_xray", batch_size=batch_size, num_workers=num_workers)

    model = PneuNet(num_classes=num_classes, weight_decay=weight_decay)

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        check_val_every_n_epoch=2,
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    __main__()