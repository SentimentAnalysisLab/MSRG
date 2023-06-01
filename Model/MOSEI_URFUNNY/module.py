import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch.fft as afft

class RGNCell(nn.Module):
    def __init__(self):
        super(RGNCell, self).__init__()
        self.dense1 = nn.Linear(256, 128)
        self.dense2 = nn.Linear(128, 128)
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(256, 128)
        self.dense5 = nn.Linear(256, 128, bias=False)

    def forward(self, s_t1, x_t):
        s_t1=s_t1.cuda()
        r_t = F.relu(self.dense1(x_t))
        m_t = torch.tanh(self.dense2(r_t))
        forget = torch.multiply(m_t, s_t1)

        i_t = torch.sigmoid(self.dense3(x_t))  # 64
        g_t = torch.tanh(self.dense4(x_t))
        read = torch.multiply(i_t, g_t)  # 64

        y_t = F.relu(forget + read)  # 64
        s_t = F.relu(F.relu(self.dense5(x_t)) + y_t)  # 64
        return s_t


class JCAF(nn.Module):
    def __init__(self, k, l, d):
        super(JCAF, self).__init__()

        self.affine_a = nn.Linear(l, l, bias=False)  # W.shape = [l, l]
        self.affine_v = nn.Linear(l, l, bias=False)
        self.affine_l = nn.Linear(l, l, bias=False)

        self.W_a = nn.Linear(l, k, bias=False)  # W_a.shape = [k, l]
        self.W_v = nn.Linear(l, k, bias=False)
        self.W_t = nn.Linear(l, k, bias=False)
        self.dense = nn.Linear(3*128,128,bias=False).cuda()

        self.W_ca = nn.Linear(2*d, k, bias=False)  # W_ca.shape = [k, d]
        self.W_cv = nn.Linear(2*d, k, bias=False)
        self.W_ct = nn.Linear(2*d, k, bias=False)

        self.W_ha = nn.Linear(k, l, bias=False)  # W_ha.shape = [k, l]
        self.W_hv = nn.Linear(k, l, bias=False)
        self.W_ht = nn.Linear(k, l, bias=False)
        amlp = AMLP(128)
        self.amlp = amlp.cuda()


    def forward(self, f1_norm, f2_norm, f3_norm):  # [k, l, d]
        fin_audio_features = []
        fin_visual_features = []
        fin_text_features = []

        txt_fts = f1_norm.cuda()  # [k, l, 128]
        aud_fts = f2_norm.cuda()  # [k, l, 128]
        vis_fts = f3_norm.cuda()  # [k, l, 128]

        G=self.amlp(txt_fts,aud_fts,vis_fts)
        for i in range(f1_norm.shape[0]):
            G_ag = torch.concat((aud_fts[i], G[i, :, :]), 1)  # [l, 128]连接[l, 128] -> [l, 256]
            a_j = self.affine_a(torch.transpose(G_ag, 0, 1))  # [256, l]*[l, l] -> [256, l]
            att_aud = torch.mm(torch.transpose(aud_fts[i], 0, 1), torch.transpose(a_j, 0, 1))  # [128, l]*[l, 256] -> [128, 256]
            C_aud_att = torch.tanh(torch.divide(att_aud, np.sqrt(G_ag.shape[1])))  # [128, 256]

            G_vg = torch.concat((vis_fts[i], G[i, :, :]), 1)
            v_j = self.affine_v(torch.transpose(G_ag, 0, 1))
            att_vis = torch.matmul(torch.transpose(vis_fts[i], 0, 1), torch.transpose(v_j, 0, 1))
            C_vis_att = torch.tanh(torch.divide(att_vis, np.sqrt(G_vg.shape[1])))  # [128, 256]

            G_lg = torch.concat((txt_fts[i], G[i, :, :]), 1)
            l_j = self.affine_l(torch.transpose(G_lg, 0, 1))
            att_txt = torch.matmul(torch.transpose(txt_fts[i], 0, 1), torch.transpose(l_j, 0, 1))
            C_txt_att = torch.tanh(torch.divide(att_txt, np.sqrt(G_lg.shape[1])))  # [128, 256]

            H_a = F.relu(self.W_ca(C_aud_att) + self.W_a(torch.transpose(aud_fts[i], 0, 1)))  # [128, 256]*[256, k]->[128, k] + [128, l]*[l, k]->[128, k]
            H_v = F.relu(self.W_cv(C_vis_att) + self.W_v(torch.transpose(vis_fts[i], 0, 1)))  # [128, k]
            H_t = F.relu(self.W_ct(C_txt_att) + self.W_t(torch.transpose(txt_fts[i], 0, 1)))  # [128, k]

            att_audio_features = torch.transpose(self.W_ha(H_a), 0, 1) + aud_fts[i]  # [128, k]*[k, l] + [l, 128] = [l, 128]
            att_visual_features = torch.transpose(self.W_hv(H_v), 0, 1) + vis_fts[i]  # [l, 128]
            att_text_features = torch.transpose(self.W_ht(H_t), 0, 1) + txt_fts[i]  # [l, 128]

            fin_audio_features.append(att_audio_features)
            fin_visual_features.append(att_visual_features)
            fin_text_features.append(att_text_features)

        fin_audio_features = torch.stack(fin_audio_features)  # [k, l, 128]
        fin_visual_features = torch.stack(fin_visual_features)  # [k, l, 128]
        fin_text_features = torch.stack(fin_text_features)  # [k, l, 128]

        return fin_text_features, fin_audio_features, fin_visual_features

class AMLP(nn.Module):
    def __init__(self,feat_size):
        super(AMLP, self).__init__()
        self.proj_i = nn.Linear(feat_size, 2*feat_size)
        self.proj_q = nn.Linear(feat_size, 2*feat_size)
        self.proj_v = nn.Linear(feat_size, 2*feat_size)
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, feat1, feat2, feat3):
        feat1 = self.proj_i(feat1)
        feat2 = self.proj_q(feat2)
        feat3 = self.proj_v(feat3)

        norm1 = torch.norm(feat1, p=2)
        norm2 = torch.norm(feat2, p=2)
        norm3 = torch.norm(feat3, p=2)
        factor1 = norm1 / (norm1 + norm2)
        factor2 = norm2 / (norm1 + norm2)
        factor3 = norm3/(norm1 + norm2+norm3)
        exp_out = factor1 * feat1 + factor2 * feat2 + factor3*norm3
        exp_out = self.dropout(exp_out)
        z = self.pool(exp_out) * 2
        z = F.normalize(z)
        return z