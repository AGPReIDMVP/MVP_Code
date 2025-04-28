import logging
import os
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from torch.cuda import amp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.meter import AverageMeter
from loss.supcontrast import SupConLoss

class ISCLoss(nn.Module):
    def __init__(self):
        """ISC Loss: Intra-modality Spatial Consistency Loss"""
        super().__init__()

    def forward(self, V_g, V_a, T_g, T_a):
        """
        ISC Loss: Calculation
        V_g, V_a: image features
        T_g, T_a: text features
        """
        term_1 = torch.mean((torch.cosine_similarity(V_g, V_a, dim=-1) - torch.cosine_similarity(T_g, T_a, dim=-1)) ** 2)
        term_2 = torch.mean((torch.matmul(V_g, V_g.T) - torch.matmul(V_a, V_a.T)) ** 2) + torch.mean((torch.matmul(T_g, T_g.T) - torch.matmul(T_a, T_a.T)) ** 2)
        return term_1 + term_2  #  ISC Loss

def train_one_stage(cfg, model, train_loader, optimizer, scheduler, 
                    center_criterion, optimizer_center, local_rank, loss_fn):
    """
    **Single-stage training: optimizing text & image features simultaneously**
    """
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD

    logger = logging.getLogger("transreid.train")
    logger.info('Start one-stage training')

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    contrastive_loss_fn = SupConLoss(device)   
    isc_loss_fn = ISCLoss()   
    

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()
        # print(f"{epoch} epoch")

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                
                image_features = model(img, target, get_image=True)  
                text_features = model(label=target, get_text=True, view_label=target_view, use_view_ctx=False)  
            
                # **ISC Loss**
                isc_loss = isc_loss_fn(image_features, image_features, text_features, text_features) * 0.1  # 权重 0.1

                # **Contrastive Loss**
                loss_i2t = contrastive_loss_fn(image_features, text_features, target, target)
                loss_t2i = contrastive_loss_fn(text_features, image_features, target, target)
                contrastive_loss = (loss_i2t + loss_t2i) / 2

                score, feat, _ = model(x = img, label = target, cam_label=None, view_label=None)
                logits = image_features @ text_features.t()

                loss_stage2 = loss_fn(score, feat, target, target_cam,logits)
                total_loss = contrastive_loss + isc_loss + loss_stage2 

            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            

            loss_meter.update(total_loss.item(), img.shape[0])

            if (n_iter + 1) % log_period == 0:
                logger.info(f"Epoch[{epoch}] Iter[{n_iter+1}/{len(train_loader)}] Loss: {loss_meter.avg:.3f}")

      
        if epoch % checkpoint_period == 0:
            model_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_one_stage_{epoch}.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved checkpoint: {model_path}")

    logger.info("Training complete!")
