# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import torch
from torch.special import digamma
import pytorch_lightning as pl
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from yacs.config import CfgNode as CN
from sklearn.metrics import roc_auc_score


class MAELitModule(pl.LightningModule):
    def __init__(self, args):
        ...
    def forward(self, x, mask_ratio=None):
        ...
    def training_step(self, batch, _):
        ...
    def on_validation_epoch_start(self):
        self.val_feats = []
        self.val_labels = []
    
    def validation_step(self, batch, _):
        ### aggregate embeddings and evaluate
        input = batch['input'].to(self.device, dtype=torch.float32)
        
        ### calculate loss
        loss, _, _ = self(input)
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        
        ### calculate embeddings
        embed, _, __ = self.model.forward_encoder(input, mask_ratio=0.0)
        embed = embed.mean(dim=1).detach()
        self.val_feats.append(embed)
        self.val_labels.append(batch['label'])
        
    def on_validation_epoch_end(self):
        feats_local = torch.cat(self.val_feats, dim=0)
        labels_local = torch.cat(self.val_labels, dim=0)
        
        ### calculate memory usage and assert
        if self.current_epoch == 0 and self.global_rank == 0:
            size = feats_local.shape[0]**2 * self.trainer.world_size * feats_local.shape[1] * 4 / 1024**3  
            print("vec dim : ", feats_local.shape[1])
            print("num samples : ", feats_local.shape[0] * self.trainer.world_size)
            print(f"Total size: {size:.2f} GB")    
            assert size < 10, f"Total size {size:.2f} GB is too large. Reduce batch size"
                
        feats_global = [None] * self.trainer.world_size
        torch.distributed.all_gather_object(feats_global, feats_local.cpu())

        feats_global = torch.cat(feats_global, dim=0).to(self.device)        
        dist_loc2glob = torch.cdist(feats_local, feats_global, p = 2)
        
        dist = [None] * self.trainer.world_size
        labels = [None] * self.trainer.world_size
        torch.distributed.all_gather_object(dist, dist_loc2glob.cpu())
        torch.distributed.all_gather_object(labels, labels_local.cpu())
        
        
        if self.trainer.is_global_zero:
            dist = torch.cat(dist, dim=0).to(self.device) ## (N, N)
            labels = torch.cat(labels, dim=0).to(self.device) ## (N, 5)
            
            TASK = ['NORMAL', 'MI', 'STTC', 'CD', 'HYP']
            
            ### calculate kNN AUROC
            k = 30
            nn_idx = dist.topk(k+1, dim=1, largest=False).indices[:, 1:] ## shape: (N, k)
            nn_labels = labels[nn_idx] ## shape: (N, k, 5)
            pred = nn_labels.sum(dim=1) / k ## shape: (N, 5)
            
            for i, task in enumerate(TASK):
                auroc = roc_auc_score(labels[:, i].cpu().numpy(), pred[:, i].cpu().numpy())
                self.log(f"auc_{task}", auroc, prog_bar=True, logger=True, rank_zero_only=True, sync_dist=False)
            
            ### calculate divergence
            k = 3
            for i, task in enumerate(TASK):
                idx_p = torch.where(labels[:, i] == 1)[0]
                idx_q = torch.where(labels[:, i] == 0)[0]
                n, m = len(idx_p), len(idx_q)
                
                d2P = dist[idx_p][:, idx_p]
                d2Q = dist[idx_p][:, idx_q]
                
                rho2p = torch.kthvalue(d2P, k+1, 1).values
                rho2q = torch.kthvalue(d2Q, k, 1).values

                rho = (rho2p > rho2q) * rho2p + (rho2p <= rho2q) * rho2q

                count_p = (d2P <= rho.unsqueeze(1)).sum(1) - 1
                count_q = (d2Q <= rho.unsqueeze(1)).sum(1)

                divergence = digamma(count_p) - digamma(count_q)
                divergence = (divergence.mean() + torch.log(torch.tensor(m/(n-1)))).item()
                self.log(f"div_{task}", divergence, prog_bar=True, logger=True, rank_zero_only=True, sync_dist=False)

            
    def configure_optimizers(self):
        ...