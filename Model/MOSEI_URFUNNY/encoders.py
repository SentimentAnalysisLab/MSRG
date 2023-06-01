from torch import nn
class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "glove" or "Bert"
    """
    def __init__(self, hyp_params):
        super(LanguageEmbeddingLayer, self).__init__()
        self.hp = hp = hyp_params
        self.embed = nn.Embedding(len(hp.word2id), hp.orig_d_l)

    def forward(self, sentences):
        # extract features from text modality
        output = self.embed(sentences)
        return output


class SeqEncoder(nn.Module):
    """Encode all modalities with assigned network.
    """
    def __init__(self, hyp_params):
        super(SeqEncoder, self).__init__()
        self.hp = hp = hyp_params
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hp.orig_d_l, hp.orig_d_a, hp.orig_d_v
        self.d_l = self.d_a = self.d_v = hp.h_dim

        self.gru_l = nn.GRU(self.orig_d_l, self.d_l, 2)
        self.gru_a = nn.GRU(self.orig_d_a, self.d_a, 2)
        self.gru_v = nn.GRU(self.orig_d_v, self.d_v, 2)


    def forward_enc(self, input_l, input_a, input_v):

        a_seq, h_a = self.gru_a(input_a)
        v_seq, h_v = self.gru_v(input_v)
        l_seq, h_l = self.gru_l(input_l)

        return {'l': (l_seq), 'v': (v_seq), 'a': (a_seq)}

    ##################################
    # TODO: Correct input shapes here
    #################################
    def forward(self, input_l, input_v, input_a):
        """Encode Sequential data from all modalities
        """
        return self.forward_enc(input_l, input_v, input_a)