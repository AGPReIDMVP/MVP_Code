import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
import os
from torchvision.utils import save_image

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class ISCLoss(nn.Module):
    def __init__(self):
        """
        ISC Loss: Intra-modality Spatial Consistency Loss
        """
        super().__init__()

    def forward(self, V_g, V_a, T_g, T_a):
        """
        Parameters.
            V_g: feature for image 1 (ground view), shape (batch_size, embed_dim)
            V_a: feature for image 2 (sky/ground view), shape (batch_size, embed_dim)
            T_g: feature for text 1 (ground view cue), shape (batch_size, embed_dim)
            T_a: feature for text 2 (sky/ground view hints), shape (batch_size, embed_dim)
        Returns.
            isc_loss: ISC loss
        """
        batch_size = V_a.size(0)

        
        term_1 = torch.mean(
            (torch.cosine_similarity(V_g, V_a, dim=-1) - torch.cosine_similarity(T_g, T_a, dim=-1)) ** 2)

        V_g_dot = torch.matmul(V_g, V_g.T)  # (batch_size, batch_size)
        V_a_dot = torch.matmul(V_a, V_a.T)  # (batch_size, batch_size)
        T_g_dot = torch.matmul(T_g, T_g.T)  # (batch_size, batch_size)
        T_a_dot = torch.matmul(T_a, T_a.T)  # (batch_size, batch_size)

        term_2 = torch.mean((V_g_dot - V_a_dot) ** 2) + torch.mean((T_g_dot - T_a_dot) ** 2)

        #  ISC Loss
        isc_loss = term_1 + term_2
        return isc_loss


def do_train_stage1(cfg,
                    model,
                    train_loader_stage1,
                    optimizer,
                    scheduler,
                    local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD

    logger = logging.getLogger("transreid.train")
    logger.info('start stage1 training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    epoch_losses = []  #Used to record the average loss for each round


    # train
    import time
    from datetime import timedelta
    from PIL import Image

    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))

    
    image_features = []
    labels = []
    target_views = []
    text_features = []

   
    save_dir = "output_image_pairs"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device)
            target = vid.to(device)

            targetview = target_view.to(device)

            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image=True)

                for i, img_feat, tv in zip(target, image_feature, targetview):
                    labels.append(i)  # len = 1

                    image_features.append(img_feat.cpu())

                    target_views.append(tv)  # len = 1

        labels_list = torch.stack(labels, dim=0).cuda()  # N
        image_features_list = torch.stack(image_features, dim=0).cuda()

        target_views_list = torch.stack(target_views, dim=0).cuda()


    batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
    num_image = labels_list.shape[0]
    i_ter = num_image // batch
    del labels, image_features

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()
        iter_list = torch.arange(num_image, device=device)  # 51451

        for i in range(i_ter + 1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i * batch:(i + 1) * batch]
            else:
                b_list = iter_list[i * batch:num_image]

            target = labels_list[b_list]
            target_all = labels_list[b_list]
            image_features = image_features_list[b_list]
            target_views = target_views_list[b_list]

            # Find eligible pairs of images in the current batch
            selected_pairs = []
            for m in range(len(b_list)):
                for n in range(m + 1, len(b_list)):
                    # Ensure that both images belong to the same ID and satisfy the perspective condition
                    if target[m] == target[n] and \
                            ((target_views[m] == 0 and target_views[n] in [0, 1]) or (
                                    target_views[n] == 0 and target_views[m] in [0, 1])):
                        selected_pairs.append((m, n))
            if len(selected_pairs) == 0:
                continue

            
            image_features1 = torch.stack([image_features[pair[0]] for pair in selected_pairs])
            image_features2 = torch.stack([image_features[pair[1]] for pair in selected_pairs])
            targets = torch.stack([target[pair[0]] for pair in selected_pairs])  # 相同 ID 的 target

            with amp.autocast(enabled=True):
                text_features = model(label=target, get_text=True,
                                      view_label=target_views, use_view_ctx=True)
                    

   

 



            text_features1 = torch.stack([text_features[pair[0]] for pair in selected_pairs])
            text_features2 = torch.stack([text_features[pair[1]] for pair in selected_pairs])

            isc_loss_fn = ISCLoss()
            loss_isc = isc_loss_fn(image_features1, image_features2, text_features1, text_features2)
            loss_isc = loss_isc * 0.1


            loss_i2t_1 = xent(image_features1, text_features1, targets, targets)
            loss_t2i_1 = xent(text_features1, image_features1, targets, targets)
            loss_i2t_2 = xent(image_features2, text_features2, targets, targets)
            loss_t2i_2 = xent(text_features2, image_features2, targets, targets)

            avg_loss = (loss_i2t_1 + loss_t2i_1 + loss_i2t_2 + loss_t2i_2) / 2 + loss_isc


            
            scaler.scale(avg_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            
            loss_meter.update(avg_loss.item(), len(selected_pairs))

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        
        epoch_losses.append(loss_meter.avg)  # Saves the average loss of the current round

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
   

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))


   
    np.save("cargo_stage1_epoch_losses.npy", np.array(epoch_losses))
    print(f"[✅] Saved epoch losses to {os.path.join(cfg.OUTPUT_DIR, 'stage1_losses.npy')}")
