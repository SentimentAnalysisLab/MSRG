import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *
from create_dataset import MOSEI, UR_FUNNY, PAD
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class MSADataset(Dataset):
    def __init__(self, config):
        self.config = config

        ## Fetch data
        if "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
        elif "ur_funny" in str(config.data_dir).lower():
            dataset = UR_FUNNY(config)
        else:
            print("Dataset not defined correctly")
            exit()
        
        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        self.len = len(self.data)

        config.visual_size = self.data[0][0][1].shape[1]
        config.acoustic_size = self.data[0][0][2].shape[1]

        config.word2id = self.word2id
        config.pretrained_emb = self.pretrained_emb

    @property
    def lav_dim(self):
        l_dim =  300
        return l_dim, self.data[0][0][2].shape[1], self.data[0][0][1].shape[1]

    @property
    def lav_len(self):
        # keep this part dummy for code compatibility. lengths are only used in CTC module
        return 0, 0, 0

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def get_loader(hp, config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)
    
    print(config.mode)
    config.data_len = len(dataset)

    config.lav_dim = dataset.lav_dim
    config.lav_len = dataset.lav_len
    
    if config.mode == 'train':
        hp.n_train = len(dataset)
    elif config.mode == 'valid':
        hp.n_valid = len(dataset)
    elif config.mode == 'test':
        hp.n_test = len(dataset)

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
        
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things


        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        
        # MOSEI sentiment labels locate in the first column of sentiment matrix
        if labels.size(1) == 7:
            labels = labels[:,0][:,None]

        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

        ## BERT-based features input prep

        SENT_LEN = sentences.size(0)
        # Create bert indices using tokenizer

        bert_details = []
        for sample in batch:
            text = " ".join(str(sample[0][3]))
            encoded_bert_sent = bert_tokenizer.encode_plus(
                text, max_length=SENT_LEN+2, add_special_tokens=True, truncation=True, padding='max_length')
            bert_details.append(encoded_bert_sent)

        # Bert things are batch_first
        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])

        return sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader