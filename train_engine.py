import os
import time
import datetime
import argparse
import numpy as np
import yaml
import random
import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from code.scheduler.build import build_lr_scheduler
from code.utils.logger import create_logger
from code.utils.metric import AverageMeter
from code.utils.checkpoint import save_checkpoint
import code.models.clip.clip as clip
from code.models.model import GVGNet_KD
from code.models.gaze_teacher import GazeModel_Teacher
from code.models.gaze_student import GazeModel_Student, SimKD
from code.datasets.dataset import TableGazeDataSet
from eval_engine import validate

import logging
logging.getLogger('PIL').setLevel(logging.WARNING)


def train_one_epoch(cfg, module_list, optimizer, scheduler, data_loader, writer, epoch, criterion_list):
    module_list[0].train()
    module_list[1].train()
    module_list[2].train()
    module_list[3].eval()
    module_list[4].eval()

    model_mcn = module_list[0]
    model_s = module_list[1]
    model_kd = module_list[2]
    model_t = module_list[3]
    model_clip = module_list[4]

    criterion_cls = criterion_list[0]
    criterion_kd = criterion_list[1]

    num_iters = len(data_loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    losses_det = AverageMeter('LossDet', ':.4f')
    losses_seg = AverageMeter('LossSeg', ':.4f')
    losskd = AverageMeter('LossKD', '.4f')
    losscls = AverageMeter('LossCls', '.4f')

    start = time.time()
    end = time.time()
    for idx, (ref, image, image_gaze, face, head_img, mask, box, gt_box, mask_id, info, gaze_label, gaze_pts) in enumerate(data_loader):
        data_time.update(time.time() - end)

        ref = ref.cuda(non_blocking=True)
        image = image.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        box = box.cuda(non_blocking=True)
        image_gaze = image_gaze.cuda(non_blocking=True)
        face = face.cuda(non_blocking=True)
        head_img = head_img.cuda(non_blocking=True)
        gaze_label = gaze_label.cuda(non_blocking=True)

        if epoch < cfg["train"]["freeze_gaze"]:
            with torch.no_grad():
                fusion_feat_s = model_s(image_gaze, head_img, face)
                heatmap_t, fusion_feat_t, inout = model_t(image_gaze, head_img, face)
                cls_t = model_t.get_deconv()
                trans_feat_s, trans_feat_t, heatmap_s = model_kd(fusion_feat_s, fusion_feat_t, cls_t)
        else:
            fusion_feat_s = model_s(image_gaze, head_img, face)
            with torch.no_grad():
                heatmap_t, fusion_feat_t, inout = model_t(image_gaze, head_img, face)
            cls_t = model_t.get_deconv()
            trans_feat_s, trans_feat_t, heatmap_s = model_kd(fusion_feat_s, fusion_feat_t, cls_t)

        heatmap_s = heatmap_s.squeeze(1)
        loss_kd = criterion_kd(trans_feat_s, trans_feat_t) * 1
        loss_cls = criterion_cls(heatmap_s, gaze_label) * 100

        image_feat = model_clip.encode_image(image)
        ref = torch.squeeze(ref, dim=1)
        ref_feat = model_clip.encode_text(ref)
        loss, loss_det, loss_seg = model_mcn(image_feat, ref_feat, trans_feat_s, det_label=box, seg_label=mask)

        loss = loss + loss_cls + loss_kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), image.size(0))
        losses_det.update(loss_det.item(), image.size(0))
        losses_seg.update(loss_seg.item(), image.size(0))
        losskd.update(loss_kd.item(), image.size(0))
        losscls.update(loss_cls.item(), image.size(0))

        global_step = epoch * num_iters + idx
        writer.add_scalar("loss/train", losses.avg, global_step=global_step)
        writer.add_scalar("loss_det/train", losses_det.avg, global_step=global_step)
        writer.add_scalar("loss_seg/train", losses_seg.avg, global_step=global_step)
        writer.add_scalar("loss_kd/train", losskd.avg, global_step=global_step)
        writer.add_scalar("loss_cls/train", losscls.avg, global_step=global_step)

        if idx % cfg["train"]["log_period"] == 0 or idx == len(data_loader):
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Train: [{epoch}/{cfg["train"]["epochs"]}][{idx}/{num_iters}]  '
                f'lr {lr:.7f}  '
                f'Time {batch_time.val:.4f}  '
                f'Loss {losses.avg:.4f}  '
                f'Det Loss {losses_det.avg:.4f}  '
                f'Seg Loss {losses_seg.avg:.4f}  '
                f'Kd Loss {losskd.avg:.4f}  '
                f'Cls Loss {losscls.avg:.4f}  '
            )

        batch_time.update(time.time() - end)
        end = time.time()

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GVGNet_KD")
    args = parser.parse_args()
    with open('./configs/config.yaml', encoding='utf-8') as config_file:
        cfg = yaml.load(config_file, Loader=yaml.FullLoader)

    if cfg["env"]["deterministic"]:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        cudnn.deterministic = True
        cudnn.benchmark = False

    output_dir = cfg["train"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=cfg["train"]["output_dir"])

    torch.cuda.set_device(0)

    cfg["dataset"]["split"] = "train"
    train_set = TableGazeDataSet(cfg)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=cfg["train"]["batch_size"],
                              shuffle=True,
                              num_workers=cfg["train"]["num_workers"])

    cfg["dataset"]["split"] = "test"
    val_set = TableGazeDataSet(cfg)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=cfg["train"]["batch_size"],
                            shuffle=False,
                            num_workers=cfg["train"]["num_workers"])

    model_t = GazeModel_Teacher()
    model_t.load_state_dict(torch.load(cfg["model"]["gaze_model_path"])['model'])
    model_s = GazeModel_Student()
    model_kd = SimKD(s_n=256, t_n=512, factor=2)

    model_clip = clip.load(cfg["model"]["clip_model_path"])
    model_mcn = GVGNet_KD()

    module_list = nn.ModuleList([])
    module_list.append(model_mcn)
    module_list.append(model_s)
    module_list.append(model_kd)
    module_list.append(model_t)
    module_list.append(model_clip)
    module_list.cuda()

    trainable_list = nn.ModuleList([])
    trainable_list.append(model_mcn)
    trainable_list.append(model_s)
    trainable_list.append(model_kd)

    optimizer = torch.optim.Adam(trainable_list.parameters(),
                                 lr=cfg["optim"]["lr"],
                                 betas=cfg["optim"]["betas"],
                                 eps=float(cfg["optim"]["eps"]))

    criterion_list = []
    criterion_cls = nn.MSELoss()
    criterion_kd = nn.MSELoss()
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_kd)

    total_params = sum([param.nelement() for param in module_list.parameters()])
    logger.info("Number of all params: %.2fM" % (total_params / 1e6))

    scheduler = build_lr_scheduler(cfg, optimizer, len(train_loader))
    writer = SummaryWriter(log_dir=cfg["train"]["output_dir"])

    start_epoch = 0
    global best_det_acc, best_seg_acc
    best_det_acc = 0.0
    best_seg_acc = 0.0

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        train_one_epoch(cfg, module_list, optimizer, scheduler, train_loader, writer, epoch, criterion_list)
        box_ap, mask_ap = validate(cfg, module_list, val_loader, writer, epoch, logger)

        if epoch % cfg["train"]["save_period"] == 0 or epoch == (cfg["train"]["epochs"] - 1):
            logger.info(f"saving checkpoints ...")

            save_checkpoint(cfg, epoch, trainable_list, optimizer, scheduler, logger)
            if box_ap > best_det_acc:
                save_checkpoint(cfg, epoch, trainable_list, optimizer, scheduler, logger, det_best=True)
            if mask_ap > best_seg_acc:
                save_checkpoint(cfg, epoch, trainable_list, optimizer, scheduler, logger, seg_best=True)
            logger.info(f"checkpoints saved !!!\n")