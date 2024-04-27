import torch.nn as nn
import numpy as np
from code.models.mcn_heads import MCNhead
from code.layers.fusion_layer import MultiScaleFusion, GaranAttention, CrossAttention

class visual_proj(nn.Module):
    def __init__(self):
        super().__init__()
        self.v1_proj = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
        )
        self.v1_res = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )

        self.v2_proj = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
        )
        self.v2_res = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1)
        )

        self.v3_proj = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )
        self.v3_res = nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )
        self.act = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(1024)

    def forward(self, img_feat):
        x = self.v1_proj(img_feat[0])
        img_feat[0] = self.act(self.bn1(self.v1_res(x) + x))
        x = self.v2_proj(img_feat[1])
        img_feat[1] = self.act(self.bn2(self.v2_res(x) + x))
        x = self.v3_proj(img_feat[2])
        img_feat[2] = self.act(self.bn3(self.v3_res(x) + x))

        return img_feat

class GVGNet_KD(nn.Module):
    def __init__(self):
        super(GVGNet_KD, self).__init__()

        self.visual_proj = visual_proj()
        self.fusion_manner = CrossAttention()
        self.multi_scale_manner = MultiScaleFusion(v_planes=(256, 512, 1024), scaled=True)

        self.det_garan = GaranAttention(d_q=1024, d_v=512)
        self.seg_garan = GaranAttention(d_q=1024, d_v=512)

        self.head = MCNhead(hidden_size=512,
                            anchors=[[137, 256], [248, 272], [386, 271]],
                            arch_mask=[[0, 1, 2]],
                            layer_no=0,
                            in_ch=512,
                            n_classes=0,
                            ignore_thre=0.5)

    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, img_feat, lang_feat, gaze_feat, det_label=None, seg_label=None):

        img_feat[0] = img_feat[0].float()
        img_feat[1] = img_feat[1].float()
        img_feat[2] = img_feat[2].float()
        lang_feat = lang_feat.float()

        img_feat = self.visual_proj(img_feat)

        img_feat[-1] = self.fusion_manner(img_feat[-1], lang_feat, gaze_feat)
        bot_feats, mid_feats, top_feats = self.multi_scale_manner(img_feat)

        bot_feats, seg_map, seg_attn = self.seg_garan(lang_feat, bot_feats)
        top_feats, det_map, det_attn = self.det_garan(lang_feat, top_feats)

        if self.training:
            loss, loss_det, loss_seg = self.head(top_feats, mid_feats, bot_feats, det_label, seg_label, det_map, seg_map,
                                                 det_attn, seg_attn)
            return loss, loss_det, loss_seg
        else:
            box, mask = self.head(top_feats, mid_feats, bot_feats)
            return box, mask

