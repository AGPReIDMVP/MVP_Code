import logging
import os
import time
from datetime import timedelta
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
import numpy as np
from thop import profile



def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start stage2 training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    epoch_losses = []

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    all_start_time = time.monotonic()

    # Calculate text features in advance
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes - batch * (num_classes // batch)
    if left != 0:
        i_ter = i_ter + 1
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i + 1 != i_ter:
                l_list = torch.arange(i * batch, (i + 1) * batch)
            else:
                l_list = torch.arange(i * batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label=l_list, get_text=True, use_view_ctx=False)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step()

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                # Calculate image features
                score, feat, image_features = model(x=img, label=target, cam_label=None, view_label=None)

                # Calculate logits
                logits = image_features @ text_features.t()

                
                selected_pairs = []
                for i in range(len(target)):
                    for j in range(i + 1, len(target)):
                        if target[i] == target[j] and (
                                (target_view[i] == 0 and target_view[j] in [0, 1]) or
                                (target_view[j] == 0 and target_view[i] in [0, 1])
                        ):
                            selected_pairs.append((i, j))
                if len(selected_pairs) == 0:
                    continue

                
                image_features1 = torch.stack([image_features[pair[0]] for pair in selected_pairs])
                image_features2 = torch.stack([image_features[pair[1]] for pair in selected_pairs])

                
                
                # Text features remain unchanged
                logits1 = image_features1 @ text_features.t()
                logits2 = image_features2 @ text_features.t()

                # Calculate loss
                loss1 = loss_fn(score, feat, target, target_cam, logits1)
                loss2 = loss_fn(score, feat, target, target_cam, logits2)
                final_loss = 0.5 * loss1 + 0.5 * loss2


            scaler.scale(final_loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(final_loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        
        epoch_losses.append(loss_meter.avg)
        
        end_time = time.time()
        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()

             

            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    camids = camids.to(device) if cfg.MODEL.SIE_CAMERA else None
                    target_view = target_view.to(device) if cfg.MODEL.SIE_VIEW else None
                    feat = model(img, cam_label=camids, view_label=target_view)
                    evaluator.update((feat, vid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()

    np.save("cargo_stage2_epoch_losses.npy", np.array(epoch_losses))
    print("✅ Stage2 loss per epoch saved to stage2_epoch_losses.npy")

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)

    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]



 