import torch
from torch import nn
import torch.optim as optim
from torch import amp

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torch.cuda.amp import autocast, GradScaler

from transformers import AutoModel
from transformers import AutoTokenizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

from model.Multi_Stage_Transformer import MultiStageFinBERT, FinBERTChunksDataset
from utils import prepare_training_val_set, device_selector

class Trainer:
    def __init__(
        self,
        model,
        train_loader=None,
        val_loader=None,
        criterion=nn.MSELoss(),
        optimizer=None,
        device=None,
        lr=2e-5,
        ckpt_dir="checkpoints",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer or optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = amp.GradScaler()
        self.ckpt_dir = ckpt_dir

        # Suivi des métriques
        self.train_losses, self.val_losses = [], []
        self.train_covs, self.val_covs = [], []
        self.train_corrs, self.val_corrs = [], []

        self.best_val_loss = float("inf")
    
    def _step(self, batch, train = True):
        mgmt_ids, mgmt_masks, rf_ids, rf_masks, labels = [
            x.to(self.device) for x in batch
        ]

        if train:
            optimizer.zero_grad()

        with torch.amp.autocast(device_type=device):
            preds = model(mgmt_ids, mgmt_masks, rf_ids, rf_masks)
            loss = criterion(preds, labels)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        return loss, preds, labels

    def _evaluate(self):
        self.model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                loss, preds, labels = self._step(batch, train=False)
                val_loss += loss.item()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_val_loss = val_loss / len(self.val_loader)
        all_preds, all_labels = torch.cat(all_preds), torch.cat(all_labels)

        val_cov, val_corr = self._compute_cov_corr(all_preds, all_labels)
        return avg_val_loss, val_cov, val_corr

    @staticmethod
    def _compute_cov_corr(preds, labels):
        preds_std = (preds - preds.mean()) / preds.std()
        labels_std = (labels - labels.mean()) / labels.std()
        cov = torch.cov(torch.stack([preds_std, labels_std]))[0, 1].item()
        corr = torch.corrcoef(torch.stack([preds, labels]))[0, 1].item()
        return cov, corr

    def save_checkpoint(self, epoch, val_loss):
        path = f"{self.ckpt_dir}/best_model_epoch{epoch}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"🔥 Nouveau checkpoint sauvegardé: {path} (Val Loss={val_loss:.4f})")

    def fit(self, n_epochs=10):
        for epoch in range(1, n_epochs + 1):
            # --- TRAIN ---
            self.model.train()
            total_loss = 0.0
            all_preds, all_labels = [], []

            for step, batch in enumerate(self.train_loader):
                loss, preds, labels = self._step(batch, train=True)
                total_loss += loss.item()
                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())

                if step % 20 == 0:
                    print(f"[Train] Epoch {epoch} Step {step}/{len(self.train_loader)} "
                          f"- Loss: {loss.item():.4f}")

            avg_train_loss = total_loss / len(self.train_loader)
            all_preds, all_labels = torch.cat(all_preds), torch.cat(all_labels)
            train_cov, train_corr = self._compute_cov_corr(all_preds, all_labels)

            self.train_losses.append(avg_train_loss)
            self.train_covs.append(train_cov)
            self.train_corrs.append(train_corr)

            # --- VAL ---
            avg_val_loss, val_cov, val_corr = self._evaluate()
            self.val_losses.append(avg_val_loss)
            self.val_covs.append(val_cov)
            self.val_corrs.append(val_corr)

            # --- CHECKPOINT ---
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_checkpoint(epoch, avg_val_loss)

            # --- SUMMARY ---
            print(f"\nEpoch {epoch}/{epochs} finished")
            print(f"  Train Loss: {avg_train_loss:.4f}, Cov: {train_cov:.4f}, Corr: {train_corr:.4f}")
            print(f"  Val   Loss: {avg_val_loss:.4f}, Cov: {val_cov:.4f}, Corr: {val_corr:.4f}\n")


if __name__ == "__main__":
    device = device_selector()

    #parameters
    n_epochs = 10
    batch_size = 32
    lr = 2e-5

    #model
    model = MultiStageFinBERT().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler() #AMP

    # load dataset
    dataset = FinBERTChunksDataset("finbert_chunks.pt")
    train_dataset, val_dataset = prepare_training_val_set(dataset, training_ratio = 0.8)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)




    ent = Trainer(
            model,
            criterion=nn.MSELoss(),
            optimizer=optim.AdamW(model.parameters(), lr=2e-5),
            device=device,
            train_loader=train_loader,
            val_loader=val_loader
        )
    ent.fit(n_epochs = n_epochs)








