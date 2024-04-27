import os
import time
import numpy as np
import cv2

import torch
import torch.nn as nn
from code.datasets.utils import yolobox2label
from code.models.utils import batch_box_iou, mask_processing, mask_iou
from code.utils.metric import AverageMeter, multi_hot_targets, to_numpy, auc, argmax_pts, L2_dist

import logging
logging.getLogger('PIL').setLevel(logging.WARNING)


def validate(cfg, module_list, data_loader, writer, epoch, logger, prefix='Val'):
    for module in module_list:
        module.eval()

    model_mcn = module_list[0]
    model_s = module_list[1]
    model_kd = module_list[2]
    model_t = module_list[3]
    model_clip = module_list[4]

    batch_time = AverageMeter('Time', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    box_ap = AverageMeter('BoxIoU_0.5', ':6.2f')
    mask_ap = AverageMeter('MaskIoU', ':6.2f')
    inconsistency_error = AverageMeter('IE', ':6.2f')
    auc_score = AverageMeter('AUC', ':6.2f')
    min_dist = AverageMeter('Dist', ':6.2f')
    mask_aps = {}
    for item in np.arange(0.5, 1, 0.05):
        mask_aps[item] = []

    with torch.no_grad():
        end = time.time()
        for idx, (ref, image, image_gaze, face, head_img, mask, box, gt_box, mask_id, info, gaze_label, gaze_pts) in enumerate(data_loader):

            ref = ref.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            image_gaze = image_gaze.cuda(non_blocking=True)
            face = face.cuda(non_blocking=True)
            head_img = head_img.cuda(non_blocking=True)
            gaze_pts = gaze_pts.cuda(non_blocking=True)

            fusion_feat_s = model_s(image_gaze, head_img, face)
            heatmap_t, fusion_feat_t, inout = model_t(image_gaze, head_img, face)
            cls_t = model_t.get_deconv()
            trans_feat_s, trans_feat_t, heatmap_s = model_kd(fusion_feat_s, fusion_feat_t, cls_t)

            image_feat = model_clip.encode_image(image)
            ref = torch.squeeze(ref, dim=1)
            ref_feat = model_clip.encode_text(ref)
            box, mask = model_mcn(image_feat, ref_feat, trans_feat_s)

            gt_box = gt_box.squeeze(1)
            gt_box[:, 2] = (gt_box[:, 0] + gt_box[:, 2])
            gt_box[:, 3] = (gt_box[:, 1] + gt_box[:, 3])
            gt_box = gt_box.cpu().numpy()
            info = info.cpu().numpy()
            box = box.squeeze(1).cpu().numpy()

            for i in range(len(gt_box)):
                box[i] = yolobox2label(box[i], info[i])

            box_iou = batch_box_iou(torch.from_numpy(gt_box), torch.from_numpy(box)).cpu().numpy()
            seg_iou = []
            auc_list = []
            dist_list = []
            mask = mask.cpu().numpy()
            for i, mask_pred in enumerate(mask):
                # AUC
                imsize = torch.IntTensor([640, 480])
                multi_hot = multi_hot_targets((gaze_pts[i][0], gaze_pts[i][1]), imsize)
                heatmap_s = heatmap_s.squeeze(1)
                scaled_heatmap = cv2.resize(heatmap_s[i].cpu().detach().numpy(), (to_numpy(imsize)[0], to_numpy(imsize)[1]))
                score = auc(scaled_heatmap, multi_hot)
                auc_list.append(score)

                # Min dist
                pred_x, pred_y = argmax_pts(heatmap_s[i].cpu())
                norm_p = [pred_x / float(64), pred_y / float(64)]
                dist_list.append(L2_dist((gaze_pts[i][0].cpu(), gaze_pts[i][1].cpu()), norm_p))

                mask_gt = cv2.imread(os.path.join('./masks/test/{}.png'.format(mask_id[i])), cv2.IMREAD_UNCHANGED)
                mask_gt = np.array(mask_gt, dtype=np.float32)
                mask_pred = mask_processing(mask_pred, info[i])

                single_seg_iou, single_seg_ap = mask_iou(mask_gt, mask_pred)
                for item in np.arange(0.5, 1, 0.05):
                    mask_aps[item].append(single_seg_ap[item] * 100.)
                seg_iou.append(single_seg_iou)

            seg_iou = np.array(seg_iou).astype(np.float32)
            auc_list = np.array(auc_list).astype(np.float32)
            dist_list = np.array(dist_list).astype(np.float32)
            ie = (box_iou >= 0.5).astype(np.float32) * (seg_iou < 0.5).astype(np.float32) + (box_iou < 0.5).astype(np.float32) * (seg_iou >= 0.5).astype(np.float32)

            inconsistency_error.update(ie.mean() * 100., ie.shape[0])
            box_ap.update((box_iou > 0.5).astype(np.float32).mean() * 100., box_iou.shape[0])
            mask_ap.update(seg_iou.mean() * 100., seg_iou.shape[0])
            auc_score.update(auc_list.mean(), auc_list.shape[0])
            min_dist.update(dist_list.mean(), dist_list.shape[0])

            if idx % cfg["train"]["log_period"] == 0 or idx == (len(data_loader) - 1):
                logger.info(
                    f'Evaluation on {prefix}: [{idx}/{len(data_loader)}]  '
                    f'Time {batch_time.val:.3f}  '
                    f'Loss {losses.avg:.4f}  '
                    f'BoxIoU@0.5 {box_ap.avg:.4f}  '
                    f'MaskIoU {mask_ap.avg:.4f}  '
                    f'IE {inconsistency_error.avg:.4f}  '
                    f'AUC {auc_score.avg:.4f}  '
                    f'Dist {min_dist.avg:.4f}  '
                )
            batch_time.update(time.time() - end)
            end = time.time()

        if writer is not None:
            writer.add_scalar("Acc/AUC", auc_score.avg, global_step=epoch)
            writer.add_scalar("Acc/Dist", min_dist.avg, global_step=epoch)
            writer.add_scalar("Acc/BoxIoU_0.5", box_ap.avg, global_step=epoch)
            writer.add_scalar("Acc/MaskIoU", mask_ap.avg, global_step=epoch)
            writer.add_scalar("Acc/IE", inconsistency_error.avg, global_step=epoch)
            for item in mask_aps:
                writer.add_scalar("Acc/MaskIoU_%.2f" % item, np.array(mask_aps[item]).mean(), global_step=epoch)

        logger.info(f' * BoxIoU@0.5 {box_ap.avg:.3f} MaskIoU {mask_ap.avg:.3f} '
                    f' AUC {auc_score.avg:.3f} Dist {min_dist.avg:.3f}')

    return box_ap.avg, mask_ap.avg
