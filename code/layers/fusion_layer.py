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

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import darknet_conv


class CollectDiffuseAttention(nn.Module):
    ''' CollectDiffuseAttention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout_c = nn.Dropout(attn_dropout)
        self.dropout_d = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, kc, kd, v, mask=None):
        '''
        q: n*b,1,d_o
        kc: n*b,h*w,d_o
        kd: n*b,h*w,d_o
        v: n*b,h*w,d_o
        '''

        attn_col = torch.bmm(q, kc.transpose(1, 2))  # n*b,1,h*w
        attn_col_logit = attn_col / self.temperature
        attn_col = self.softmax(attn_col_logit)
        attn_col = self.dropout_c(attn_col)
        attn = torch.bmm(attn_col, v)  # n*b,1,d_o

        attn_dif = torch.bmm(kd, q.transpose(1, 2))  # n*b,h*w,1
        attn_dif_logit = attn_dif / self.temperature
        attn_dif = torch.sigmoid(attn_dif_logit)
        attn_dif = self.dropout_d(attn_dif)
        output = torch.bmm(attn_dif, attn)
        return output, attn_col_logit.squeeze(1)


class GaranAttention(nn.Module):
    """
    Garan Attention Module
    """

    def __init__(self, d_q, d_v, n_head=2, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_q = d_q
        self.d_v = d_v
        self.d_k = d_v
        self.d_o = d_v
        d_o = d_v

        self.w_qs = nn.Linear(d_q, d_o)
        self.w_kc = nn.Conv2d(d_v, d_o, 1)
        self.w_kd = nn.Conv2d(d_v, d_o, 1)
        self.w_vs = nn.Conv2d(d_v, d_o, 1)
        self.w_m = nn.Conv2d(d_o, 1, 3, 1, padding=1)
        self.w_o = nn.Conv2d(d_o, d_o, 1)

        self.attention = CollectDiffuseAttention(temperature=np.power(d_o // n_head, 0.5))
        self.layer_norm = nn.BatchNorm2d(d_o)
        self.layer_acti = nn.LeakyReLU(0.1, inplace=True)
        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, v, mask=None):
        d_k, d_v, n_head, d_o = self.d_k, self.d_v, self.n_head, self.d_o

        sz_b, c_q = q.size()
        sz_b, c_v, h_v, w_v = v.size()
        residual = v

        q = self.w_qs(q)
        kc = self.w_kc(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)
        kd = self.w_kd(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)
        v = self.w_vs(v).view(sz_b, n_head, d_o // n_head, h_v * w_v)
        q = q.view(sz_b, n_head, 1, d_o // n_head)
        # v=v.view(sz_b,h_v*w_v,n_head,c_v//n_head)

        q = q.view(-1, 1, d_o // n_head)  # (n*b) x lq x dk
        kc = kc.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # (n*b) x lk x dk
        kd = kd.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # (n*b) x lk x dk
        v = v.permute(0, 1, 3, 2).contiguous().view(-1, h_v * w_v, d_o // n_head)  # (n*b) x lv x dv

        output, m_attn = self.attention(q, kc, kd, v)
        # n * b, h * w, d_o
        output = output.view(sz_b, n_head, h_v, w_v, d_o // n_head)
        output = output.permute(0, 1, 4, 3, 2).contiguous().view(sz_b, -1, h_v, w_v)  # b x lq x (n*dv)
        m_attn = m_attn.view(sz_b, n_head, h_v * w_v)
        # m_attn=m_attn.mean(1)

        # residual connect
        output = self.w_o(output)
        attn = output
        m_attn = self.w_m(attn).view(sz_b, h_v * w_v)
        output = self.layer_norm(output)
        output = output + residual
        output = self.layer_acti(output)

        return output, m_attn, attn


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.v_proj = nn.Sequential(
            nn.Conv2d(1024, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )
        self.g_proj = nn.Sequential(
            nn.Conv2d(512, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.vg_fuse = nn.Sequential(
            nn.Conv2d(1024, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        self.vl_fuse = nn.Sequential(
            nn.Conv2d(1024, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(1024, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )
        self.bn = nn.BatchNorm2d(1024)
        self.act = nn.LeakyReLU(0.1)

        self.drop2d = nn.Dropout2d(0.1)

    def forward(self, visual_feat, lang_feat, gaze_feat):

        visual_feat = self.v_proj(visual_feat)                      

        gaze_feat = self.g_proj(gaze_feat)
        gaze_feat = torch.Tensor.repeat(gaze_feat, (1, 1, 13, 13)) 

        lang_feat = lang_feat.unsqueeze(2).unsqueeze(2)            
        lang_feat = torch.Tensor.repeat(lang_feat, (1, 1, 13, 13))  

        B, C, H, W = visual_feat.shape
        h_v_temp = visual_feat.view(B, C, -1)  
        h_g_temp = gaze_feat.view(B, C, -1)    
        h_l_temp = lang_feat.view(B, C, -1)    

        vg_tmp = h_g_temp @ h_v_temp.transpose(-2, -1)    
        vg_tmp = F.softmax(vg_tmp, dim=-1)            
        cross_vg = (vg_tmp @ h_v_temp).contiguous()  
        cross_vg = cross_vg.view(B, C, H, W)          
        cross_vg = self.vg_fuse(cross_vg)
        cross_vg = self.bn(self.act(cross_vg + visual_feat))

        lv_tmp = h_l_temp @ h_v_temp.transpose(-2, -1)
        lv_tmp = F.softmax(lv_tmp, dim=-1)
        cross_lv = (lv_tmp @ h_v_temp).contiguous()
        cross_lv = cross_lv.view(B, C, H, W)
        cross_lv = self.vl_fuse(cross_lv)
        cross_lv = self.bn(self.act(cross_lv + visual_feat))

        h_merge = visual_feat + cross_vg + cross_lv
        out = self.fuse(self.drop2d(h_merge))

        return out


class MultiScaleFusion(nn.Module):
    def __init__(self, v_planes=[256, 512, 1024], hiden_planes=512, scaled=True):
        super().__init__()
        self.up_modules = nn.ModuleList(
            [nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
                darknet_conv(v_planes[-2] + hiden_planes // 2, hiden_planes // 2, ksize=1),
                darknet_conv(hiden_planes // 2, hiden_planes // 2, 3),
            ),
            nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
                darknet_conv(v_planes[-1], hiden_planes // 2, ksize=1),
                darknet_conv(hiden_planes // 2, hiden_planes // 2, 3)
            )]
        )

        self.down_modules = nn.ModuleList(
            [nn.Sequential(
                nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
                darknet_conv(hiden_planes // 2 + v_planes[0], hiden_planes // 2, ksize=1),
                darknet_conv(hiden_planes // 2, hiden_planes // 2, 3),
            ),
            nn.Sequential(
                nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
                darknet_conv(hiden_planes + v_planes[1], hiden_planes // 2, ksize=1),
                darknet_conv(hiden_planes // 2, hiden_planes // 2, 3),
            )]
        )

        self.top_proj = darknet_conv(v_planes[-1] + hiden_planes // 2, hiden_planes, 1)
        self.mid_proj = darknet_conv(v_planes[1] + hiden_planes, hiden_planes, 1)
        self.bot_proj = darknet_conv(v_planes[0] + hiden_planes // 2, hiden_planes, 1)

    def forward(self, x):
        l, m, s = x
        m = torch.cat([self.up_modules[1](s), m], 1)
        l = torch.cat([self.up_modules[0](m), l], 1)

        m = torch.cat([self.down_modules[0](l), m], 1)
        s = torch.cat([self.down_modules[1](m), s], 1)

        # top prpj and bot proj
        top_feat = self.top_proj(s)
        mid_feat = self.mid_proj(m)
        bot_feat = self.bot_proj(l)

        return [bot_feat, mid_feat, top_feat]
