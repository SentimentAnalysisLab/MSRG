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
        s_t1=s_t1
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
        self.dense = nn.Linear(3*128,128,bias=False)

        self.W_ca = nn.Linear(2*d, k, bias=False)  # W_ca.shape = [k, d]
        self.W_cv = nn.Linear(2*d, k, bias=False)
        self.W_ct = nn.Linear(2*d, k, bias=False)

        self.W_ha = nn.Linear(k, l, bias=False)  # W_ha.shape = [k, l]
        self.W_hv = nn.Linear(k, l, bias=False)
        self.W_ht = nn.Linear(k, l, bias=False)
        self.mode = BiAMLP(128)  #MFB(2,128) CBP(128,128,128) ,MLB(2,128)

    def forward(self, f1_norm, f2_norm, f3_norm):  # [k, l, d]
        fin_audio_features = []
        fin_visual_features = []
        fin_text_features = []

        txt_fts = f1_norm # [k, l, 128]
        aud_fts = f2_norm  # [k, l, 128]
        vis_fts = f3_norm  # [k, l, 128]

        # 用于AMLP消融 (和BiAMLP MFB,CBP,MLB 对比)
        G1 = self.mode(txt_fts, aud_fts)
        G2 = self.mode(txt_fts, vis_fts)
        G3 = self.mode(aud_fts, vis_fts)

        ##############################################################################
        #G=G1 #T,A模态
        #G=G2 #T,V模态
        #G=G3 #A,V模态
        # G=G1+G2+G3   # 对应加
        # G=self.Dense(torch.concat([G1, G2, G3], dim=2)) # 全连接
        # G = G1 * G2 * G3  # 对应乘
        ##############################################################################

        G=G1

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


class BiAMLP(nn.Module):#AMLP的衍生物，我们将它提出以应用自适应因子思想于双模态，它同样应用于我们的消融实验
    def __init__(self,feat_size):
        super(AMLP, self).__init__()
        self.proj_i = nn.Linear(feat_size, 2*feat_size)
        self.proj_q = nn.Linear(feat_size, 2*feat_size)
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, feat1, feat2):
        feat1 = self.proj_i(feat1)
        feat2 = self.proj_q(feat2)

        norm1 = torch.norm(feat1, p=2)
        norm2 = torch.norm(feat2, p=2)

        factor1 = norm1 / (norm1 + norm2)
        factor2 = norm2 / (norm1 + norm2)

        exp_out = factor1 * feat1 + factor2 * feat2
        exp_out = self.dropout(exp_out)
        z = self.pool(exp_out) * 2
        z = F.normalize(z)
        return z

class MFB(nn.Module):
    def __init__(self, C,feat_size):
        super(MFB, self).__init__()
        self.C = C
        self.proj_i = nn.Linear(feat_size,C*feat_size)
        self.proj_q = nn.Linear(feat_size, C*feat_size)
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.AvgPool1d(C, stride=C)

    def forward(self, feat1, feat2):
        feat1 = self.proj_i(feat1)
        feat2 = self.proj_q(feat2)
        exp_out = feat1 * feat2
        exp_out = self.dropout(exp_out)
        z = self.pool(exp_out) * self.C
        z = F.normalize(z)
        return z

class MLB(nn.Module):
    def __init__(self, C, feat_size):
        super(MLB, self).__init__()
        self.C = C
        self.proj_i = nn.Linear(feat_size, feat_size)
        self.proj_q = nn.Linear(feat_size, feat_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, feat1, feat2):
        feat1 = self.proj_i(feat1)
        feat2 = self.proj_q(feat2)
        exp_out = feat1 * feat2
        z = self.dropout(exp_out)
        z = F.normalize(z)
        return z

class CBP(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim,
                 sum_pool=True, rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None):
        super(CBP, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool

        if rand_h_1 is None:
            np.random.seed(1)
            rand_h_1 = np.random.randint(output_dim, size=self.input_dim1)
        if rand_s_1 is None:
            np.random.seed(3)
            rand_s_1 = 2 * np.random.randint(2, size=self.input_dim1) - 1

        self.sparse_sketch_matrix1 = Variable(self.generate_sketch_matrix(
            rand_h_1, rand_s_1, self.output_dim))

        if rand_h_2 is None:
            np.random.seed(5)
            rand_h_2 = np.random.randint(output_dim, size=self.input_dim2)
        if rand_s_2 is None:
            np.random.seed(7)
            rand_s_2 = 2 * np.random.randint(2, size=self.input_dim2) - 1

        self.sparse_sketch_matrix2 = Variable(self.generate_sketch_matrix(
            rand_h_2, rand_s_2, self.output_dim))


    def forward(self, bottom1, bottom2):


        batch_size, height, width = bottom1.size()

        bottom1_flat = bottom1.permute(0, 2, 1).contiguous().view(-1, self.input_dim1)
        bottom2_flat = bottom2.permute(0, 2, 1).contiguous().view(-1, self.input_dim2)

        sketch_1 = bottom1_flat.mm(self.sparse_sketch_matrix1)
        sketch_2 = bottom2_flat.mm(self.sparse_sketch_matrix2)

        fft1 = afft.fft(sketch_1)
        fft2 = afft.fft(sketch_2)

        fft_product = fft1 * fft2
        #print(fft_product.shape)

        cbp_flat = afft.ifft(fft_product).real

        cbp = cbp_flat.view(batch_size, height,self.output_dim)

        return cbp

    @staticmethod
    def generate_sketch_matrix(rand_h, rand_s, output_dim):
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert(rand_h.ndim == 1 and rand_s.ndim ==
               1 and len(rand_h) == len(rand_s))
        assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        rand_s = torch.from_numpy(rand_s)
        sparse_sketch_matrix = torch.sparse.FloatTensor(
            indices.t(), rand_s, torch.Size([input_dim, output_dim]))
        return sparse_sketch_matrix.to_dense()
