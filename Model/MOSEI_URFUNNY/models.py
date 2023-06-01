import torch
from torch import nn
import torch.nn.functional as F
from encoders import LanguageEmbeddingLayer, SeqEncoder
from module import JCAF,RGNCell

class MSRG(nn.Module):
    def __init__(self, hyp_params):
        super(MSRG, self).__init__()
        self.hp = hp = hyp_params
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hp.orig_d_l, hp.orig_d_a, hp.orig_d_v
        self.embed_dropout = hp.embed_dropout
        self.output_dim = hp.output_dim
        self.embedding = LanguageEmbeddingLayer(self.hp)
        self.SequenceEncoder = SeqEncoder(self.hp)

        self.dh_l = self.dh_a = self.dh_v = 128
        self.project_a = nn.GRU(self.dh_a, 64, bidirectional=True, batch_first=True).cuda()
        self.project_v = nn.GRU(self.dh_v, 64, bidirectional=True, batch_first=True).cuda()
        self.project_l = nn.GRU(self.dh_l, 64, bidirectional=True, batch_first=True).cuda()

        self.bi_lstm = nn.LSTM(self.dh_v + self.dh_l + self.dh_a, 128, bidirectional=True, batch_first=True).cuda()
        self.bi_lstm1 = nn.GRU(128 * 3, 64, batch_first=True, bidirectional=True).cuda()
        rgn0 = RGNCell()
        self.rgn0 = rgn0.cuda()
        rgn1= RGNCell()
        self.rgn1 =rgn1.cuda()
        fc3 = nn.Linear(256, 128)
        self.fc3=fc3.cuda()
        fc1 = nn.Linear(128, 32)
        self.fc1 = fc1.cuda()
        fc2 = nn.Linear(32, 1)
        self.fc2 = fc2.cuda()
            
    def forward(self, sentences, video, acoustic):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """

        enc_word = self.embedding(sentences) # (seq_len, batch_size, emb_size)
        enc_word = F.dropout(enc_word, p=self.embed_dropout, training=self.training)
        # Project the textual/visual/audio features
        proj_res = self.SequenceEncoder(enc_word, acoustic, video)
        
        seq_l = proj_res['l'] # (seq_len, batch_size, emb_size)
        seq_v = proj_res['v']
        seq_a = proj_res['a']
        x_a_origin, d = self.project_a(seq_a)
        x_v_origin, d = self.project_v(seq_v)
        x_l_origin, d = self.project_l(seq_l)
        x_a_origin,x_v_origin,x_l_origin = x_a_origin.permute(1, 0, 2) ,x_v_origin.permute(1, 0, 2) ,x_l_origin.permute(1, 0, 2)

        jcaf = JCAF(x_a_origin.shape[0], x_a_origin.shape[1], x_a_origin.shape[2])
        self.jcaf = jcaf.cuda()
        fin_txt, fin_aud, fin_vis = self.jcaf(x_l_origin, x_a_origin, x_v_origin)
        tstep = torch.concat((fin_txt, fin_aud, fin_vis), dim=2)
        step_fusion, (_, _) = self.bi_lstm(tstep)
        state0 = torch.randn([step_fusion.shape[0], 128])
        state1 = torch.randn([step_fusion.shape[0], 128])
        for step in torch.unbind(step_fusion, dim=1):
            state0 = out0 = self.rgn0(state0, step)
        for step in torch.unbind(torch.flip(step_fusion, dims=[1]), dim=1):
            state1 = out1 = self.rgn1(state1, step)
        final_output = self.fc2(torch.nn.functional.leaky_relu(self.fc1(out0 + out1)))
        return final_output