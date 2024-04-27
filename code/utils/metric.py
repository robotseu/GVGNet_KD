# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
import torch


class ProgressMeter(object):
    def __init__(self, version,num_epochs, num_batches, meters, prefix=""):
        self.fmtstr = self._get_epoch_batch_fmtstr(version,num_epochs, num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, epoch, batch):
        entries = [self.prefix + self.fmtstr.format(epoch, batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_epoch_batch_fmtstr(self, version,num_epochs, num_batches):
        num_digits_epoch = len(str(num_epochs // 1))
        num_digits_batch = len(str(num_batches // 1))
        epoch_fmt = '{:' + str(num_digits_epoch) + 'd}'
        batch_fmt = '{:' + str(num_digits_batch) + 'd}'
        return '[' 'version: '+version+' '+ epoch_fmt + '/' + epoch_fmt.format(num_epochs) + ']' + '[' + batch_fmt + '/' + batch_fmt.format(
            num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.avg_reduce = 0.

    def update(self, val, n=1):
        self.val = val
        if n==-1:
            self.sum=val
            self.count=1
        else:
            self.sum += val * n
            self.count += n
        self.avg = self.sum / self.count

    def update_reduce(self, val):
        self.avg_reduce = val

    def __str__(self):
        fmtstr = '{name} {avg_reduce' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def auc(heatmap, onehot_im, is_im=True):
    if is_im:
        auc_score = roc_auc_score(np.reshape(onehot_im, onehot_im.size), np.reshape(heatmap, heatmap.size))
    else:
        auc_score = roc_auc_score(onehot_im, heatmap)
    return auc_score


def ap(label, pred):
    return average_precision_score(label, pred)


def argmax_pts(heatmap):
    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    idx = (idx[0], idx[1])
    pred_y, pred_x = map(float, idx)
    return pred_x, pred_y


def L2_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def multi_hot_targets(gaze_pts, out_res):
    w, h = out_res
    target_map = np.zeros((h, w))
    x, y = map(int,[gaze_pts[0]*w.float(), gaze_pts[1]*h.float()])
    x = min(x, w-1)
    y = min(y, h-1)
    target_map[y, x] = 1

    return target_map

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray