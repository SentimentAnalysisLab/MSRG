from torch import nn
import sys
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.multiprocessing as mp
from utils.eval_metrics import *
from utils.tools import *
from models import MSRG

class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.is_train = is_train
        self.model = model

        # initialize the model
        if model is None:
            self.model = MSRG(hp)
        
        # Initialize weight of Embedding matrix with Glove embeddings
        if self.hp.pretrained_emb is not None:
            self.model.embedding.embed.weight.data = self.hp.pretrained_emb
        self.model.embedding.embed.requires_grad = False

        if hp.use_cuda:
            print(hp.use_cuda)
            self.model = self.model.cuda()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # optimizer
        if self.is_train:
            self.optimizer = getattr(torch.optim, self.hp.optim)(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.hp.lr)
        
        # criterion
        if self.hp.dataset == "ur_funny":
            self.criterion = nn.CrossEntropyLoss(reduction="mean")
        else:
            self.criterion = nn.MSELoss(reduction="mean")

        # Final list
        for name, param in self.model.named_parameters():
            # Bert freezing customizations 
            if self.hp.data in ["mosei"]:
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= 8:
                        param.requires_grad = False
            elif self.hp.data == "ur_funny":
                if "bert" in name:
                    param.requires_grad = False
            
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=20, factor=0.1, verbose=True)

    def train_and_eval(self):
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        criterion = self.criterion

        def train(model, optimizer, criterion):
            epoch_loss = 0
            model.train()

            for i_batch, batch_data in enumerate(self.train_loader):
                text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch_data
                model.zero_grad()
                if self.hp.use_cuda:
                    with torch.cuda.device(0):
                        text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                        text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), \
                        bert_sent_type.cuda(), bert_sent_mask.cuda()
                        if self.hp.dataset=="ur_funny":
                            y = y.squeeze()

                batch_size = y.size(0)
                batch_chunk = self.hp.batch_chunk
                combined_loss = 0
                # If parallel fails due to limited space, then increase batch_chunk to decrease GPU utilization
                if batch_chunk > 1:
                    pass
                else:
                    preds = model(text, visual, audio)
                    if self.hp.dataset == "ur_funny":
                        y = y.unsqueeze(-1)
                    raw_loss = criterion(preds, y.float())
                    combined_loss = raw_loss
                    combined_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                optimizer.step()
                epoch_loss += combined_loss.item()* batch_size
                    
            return epoch_loss / self.hp.n_train

        def evaluate(model, criterion, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            results = []
            truths = []
            with torch.no_grad():
                for batch in loader:
                    text, vision, audio, y, lengths, bert_sent, bert_sent_type, bert_sent_mask = batch
                    # eval_attr = y.squeeze(dim=-1) # if num of labels is 1

                    if self.hp.use_cuda:
                        with torch.cuda.device(0):
                            text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
                            lengths = lengths.cuda()
                            bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()
                            if self.hp.dataset == 'ur_funny':
                                y = y.squeeze()

                    preds= model(text, vision, audio)
                    if self.hp.dataset in ['mosei', 'mosei_senti'] and test:
                        criterion = nn.L1Loss()

                    batch_size = lengths.size(0)
                    total_loss += criterion(preds, y.float()).item()* batch_size

                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(preds)
                    truths.append(y)
            
            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)
            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths

        best_valid = 1e8
        best_epoch = -1
        patience = 20

        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()
            self.epoch = epoch
            train_loss=train(model, optimizer, criterion)
            val_loss,  _, _ = evaluate(model, criterion, test=False)
            test_loss, results, truths = evaluate(model, criterion, test=True)
            
            end = time.time()
            duration = end-start
            scheduler.step(val_loss)    # Decay learning rate by validation loss

            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec| Train Loss {:5.4f} | Valid Loss {:5.4f}  | Test Loss {:5.4f}'.format(epoch, duration, train_loss,val_loss, test_loss))
            print("-"*50)

            if best_valid > val_loss:
                best_epoch = epoch
                patience = 20
                best_valid = val_loss
                if self.hp.dataset == "ur_funny":
                    eval_humor(results, truths, True)
                elif self.hp.dataset in ["mosei_senti", "mosei"]:
                    eval_mosei_senti(results, truths, True)
                    print(f"Saved model at pre_trained_models/{self.hp.name}.pt!")
                    save_model(self.hp, model, name=self.hp.name)
            else:
                patience -= 1
                if patience == 0:
                    break

        model = load_model(self.hp, name=self.hp.name)
        print(f'Best epoch: {best_epoch}')
        sys.stdout.flush()

