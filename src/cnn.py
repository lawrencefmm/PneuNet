import lightning.pytorch as pl
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall


class PneuNet(pl.LightningModule):
    def __init__(self, num_classes=3, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.mp = {
            0: "PNEUMONIA_BACTERIA",
            1: "NORMAL",
            2: "PNEUMONIA_VIRUS"
        }

        # Input: (batch_size, 1, H=256, W=256)
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, padding="same"),   # Output: (batch_size, 4, H=256, W=256)
            nn.ReLU(),
            nn.Conv2d(4, 16, kernel_size=5, padding="same"),  # Output: (batch_size, 16, H=256, W=256)
            nn.ReLU(),
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=5, padding="same"), # Output: (batch_size, 64, H=256, W=256)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding="same"), # Output: (batch_size, 64, H=256, W=256)
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2)                        # Output: (batch_size, 64, H=128, W=128)
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=5, padding="same"),# Output: (batch_size, 256, H=128, W=128)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, padding="same"),# Output: (batch_size, 256, H=128, W=128)
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2)                         # Output: (batch_size, 256, H=64, W=64)
        )

        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, padding="same"),# Output: (batch_size, 512, H=64, W=64)
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=5, padding="same"),# Output: (batch_size, 512, H=64, W=64)
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2)                          # Output: (batch_size, 512, H=32, W=32)
        )

        self.conv_layer_5 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=5, padding="same"),# Output: (batch_size, 256, H=32, W=32)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, padding="same"),# Output: (batch_size, 256, H=32, W=32)
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2)                          # Output: (batch_size, 256, H=16, W=16)
        )

        self.conv_layer_6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, padding="same"),# Output: (batch_size, 128, H=16, W=16)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding="same"),# Output: (batch_size, 128, H=16, W=16)
            nn.ReLU(),
            nn.AvgPool2d(4, stride=4)                          # Output: (batch_size, 128, H=4, W=4)
        )

        # Flattened size: 128 * H/64 * W/64 = 128 * 4 * 4 = 2048
        self.fully_connected = nn.Sequential(
            nn.BatchNorm1d(2048),                              # Input: (batch_size, 2048)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(2048, 256),                              # Output: (batch_size, 256)
            nn.ReLU(),
            nn.Linear(256, 64),                                # Output: (batch_size, 64)
            nn.ReLU(),
            nn.Linear(64, num_classes),                        # Output: (batch_size, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(average="macro", num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(average="macro", num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(average="macro", num_classes=num_classes)
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average=None)
        self.test_recall = MulticlassRecall(num_classes=num_classes, average=None)
        self.test_precision_macro = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.test_recall_macro = MulticlassRecall(num_classes=num_classes, average='macro')


    def forward(self, x):
        x = self.conv_layer_1(x)   
        x = self.conv_layer_2(x)   
        x = self.conv_layer_3(x)   
        x = self.conv_layer_4(x)   
        x = self.conv_layer_5(x)   
        x = self.conv_layer_6(x)  
        x = torch.flatten(x, 1) 
        x = self.fully_connected(x) 
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        train_acc = self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_acc", train_acc, prog_bar=True, on_step=True, on_epoch=False)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_acc.update(logits, y)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.test_acc.update(logits, y)
        self.test_precision.update(logits, y)
        self.test_recall.update(logits, y)
        self.test_precision_macro.update(logits,y)
        self.test_recall_macro.update(logits,y)

        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_precision_macro", self.test_precision_macro, on_step=False, on_epoch=True)
        self.log("test_recall_macro", self.test_recall_macro, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        final_precision = self.test_precision.compute()
        final_recall = self.test_recall.compute()

        for i, precision in enumerate(final_precision):
           self.log(f"precision_{self.mp[i]}", precision)
        for i, recall in enumerate(final_recall):
           self.log(f"recall_{self.mp[i]}", recall)

        self.test_precision.reset()
        self.test_recall.reset()


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, tuple) or isinstance(batch, list):
             x = batch[0] 
        else:
             x = batch
        return self(x)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',  
            factor=0.1,      
            patience=10,     
            verbose=True     
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  
                "interval": "epoch",    
                "frequency": 6,        
                "strict": True,         
            },
        }

