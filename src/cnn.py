import lightning.pytorch as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix
)
import pandas as pd
import io

class ConvBlock(nn.Module):
    def __init__(self, C_in, C_out, W):
        super().__init__()

        self.conv1 = nn.Conv2d(C_in, C_out, kernel_size=3, padding='same')
        self.norm1 = nn.LayerNorm([C_out, W, W])
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(C_out, C_out, kernel_size=3, padding='same')
        self.norm2 = nn.LayerNorm([C_out, W, W])
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(C_out, C_out, kernel_size=3, padding='same')
        self.norm3 = nn.LayerNorm([C_out, W, W])
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(C_out, C_out, kernel_size=3, padding='same')
        self.norm4 = nn.LayerNorm([C_out, W, W])
        self.relu4 = nn.ReLU()

    def forward(self, x):
        residual = self.norm1(self.conv1(x))
        residual = self.relu1(residual)

        x = self.conv2(residual)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x += residual
        x = self.relu4(x)

        return x

class PneuNet(pl.LightningModule):
    def __init__(self, num_classes=3, lr=1e-3, weight_decay=1e-4, batch_size=32):
        super().__init__()
        self.save_hyperparameters("num_classes", "lr", "weight_decay", "batch_size")

        # cost matrix to reduce false positives
        self.cost_matrix =  torch.tensor(
            [[0, 1, 1],
            [10, 0, 10],
            [1, 1, 0]]
        )

        self.mp = {
            0: "PNEUMONIA_BACTERIA",
            1: "NORMAL",
            2: "PNEUMONIA_VIRUS"
        }
        self.class_names = [self.mp[i] for i in range(self.hparams.num_classes)]

        # (B, 1, 256, 256)
        self.conv_layer_1 = nn.Sequential(
            ConvBlock(1, 2, 256),
            nn.AvgPool2d(kernel_size=4, stride=4)  # (B, 2, 64, 64)
        )
        self.conv_layer_2 = nn.Sequential(
            ConvBlock(2, 16, 64),
            nn.AvgPool2d(kernel_size=2, stride=2)  # (B, 16, 32, 32)
        )
        self.conv_layer_3 = nn.Sequential(
            ConvBlock(16, 32, 32),
            nn.AvgPool2d(kernel_size=2, stride=2)  # (B, 32, 16, 16)
        )
        self.conv_layer_4 = nn.Sequential(
            ConvBlock(32, 64, 16),
            nn.AvgPool2d(kernel_size=2, stride=2)  # (B, 64, 8, 8)
        )
        self.conv_layer_5 = nn.Sequential(
            ConvBlock(64, 64, 8),
            nn.AvgPool2d(kernel_size=2, stride=2)  # (B, 64, 4, 4)
        )
        self.conv_layer_6 = nn.Sequential(
            ConvBlock(64, 64, 4),
            nn.AvgPool2d(kernel_size=2, stride=2)  # (B, 64, 2, 2)
        )
        self.fully_conected = nn.Sequential(
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, self.hparams.num_classes)
        )

        self.train_acc = MulticlassAccuracy(average="macro", num_classes=self.hparams.num_classes)
        self.val_acc = MulticlassAccuracy(average="macro", num_classes=self.hparams.num_classes)
        self.test_acc = MulticlassAccuracy(average="macro", num_classes=self.hparams.num_classes)
        self.test_precision = MulticlassPrecision(num_classes=self.hparams.num_classes, average=None)
        self.test_recall = MulticlassRecall(num_classes=self.hparams.num_classes, average=None)
        self.test_precision_macro = MulticlassPrecision(num_classes=self.hparams.num_classes, average='macro')
        self.test_recall_macro = MulticlassRecall(num_classes=self.hparams.num_classes, average='macro')
        self.test_cm = MulticlassConfusionMatrix(num_classes=self.hparams.num_classes)

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.conv_layer_5(x)
        x = self.conv_layer_6(x)
        x = torch.flatten(x, 1)
        x = self.fully_conected(x)
        return x
    
    def loss_fn(self, logits, y):
        log_p = F.log_softmax(logits, dim=1)  # [B, C]
        nll   = F.nll_loss(log_p, y, reduction="none")  # [B]

        cost_matrix = self.cost_matrix.to(y.device)

        p = torch.exp(log_p)             # [B, C]
        sample_cost = (cost_matrix[y] * p).sum(dim=1)  # [B]

        loss = (nll * sample_cost).mean()
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        train_acc = self.train_acc(preds, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train_acc", train_acc, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.test_acc.update(preds, y)
        self.test_precision.update(preds, y)
        self.test_recall.update(preds, y)
        self.test_precision_macro.update(preds, y)
        self.test_recall_macro.update(preds, y)
        self.test_cm.update(preds, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_precision_macro", self.test_precision_macro, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_recall_macro", self.test_recall_macro, on_step=False, on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self):
        final_acc = self.test_acc.compute()
        final_prec_macro = self.test_precision_macro.compute()
        final_recall_macro = self.test_recall_macro.compute()
        final_precision_per_class = self.test_precision.compute()
        final_recall_per_class = self.test_recall.compute()
        final_cm = self.test_cm.compute()

        print("\n" + "="*40)
        print("        Test Results Summary")
        print("="*40)
        print(f"Overall Test Accuracy : {final_acc:.4f}")
        print(f"Macro Avg Precision   : {final_prec_macro:.4f}")
        print(f"Macro Avg Recall      : {final_recall_macro:.4f}")
        print("-"*40)
        print("Per-Class Metrics:")
        for i in range(self.hparams.num_classes):
            class_name = self.class_names[i]
            print(f"  Class: {class_name}")
            print(f"    Precision: {final_precision_per_class[i]:.4f}")
            print(f"    Recall   : {final_recall_per_class[i]:.4f}")
        print("-"*40)

        print("Confusion Matrix:")
        cm_df = pd.DataFrame(final_cm.cpu().numpy(),
                             index=self.class_names,
                             columns=self.class_names)
        print(cm_df)
        print("="*40 + "\n")

        for i, precision in enumerate(final_precision_per_class):
            self.log(f"test_precision_{self.class_names[i]}", precision, sync_dist=True)
        for i, recall in enumerate(final_recall_per_class):
            self.log(f"test_recall_{self.class_names[i]}", recall, sync_dist=True)

        buf = io.StringIO()
        cm_df.to_csv(buf)
        if hasattr(self.logger.experiment, 'add_text'):
             self.logger.experiment.add_text("Confusion Matrix", f"<pre>{cm_df.to_markdown()}</pre>", self.current_epoch)

        self.test_precision.reset()
        self.test_recall.reset()
        self.test_cm.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (tuple, list)):
             x = batch[0]
        else:
             x = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        return probs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        return {
            "optimizer": optimizer,
        }
