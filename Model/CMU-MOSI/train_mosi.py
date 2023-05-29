import time
import numpy as np

seed = 123
np.random.seed(seed)
import random
import torch

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import  h5py
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
import sys
from module import RGNCell,JCAF


def load_saved_data():
    h5f = h5py.File('data/X_train.h5', 'r')
    X_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('data/y_train.h5', 'r')
    y_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('data/X_valid.h5', 'r')
    X_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('data/y_valid.h5', 'r')
    y_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('data/X_test.h5', 'r')
    X_test = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('data/y_test.h5', 'r')
    y_test = h5f['data'][:]
    h5f.close()
    return X_train, y_train, X_valid, y_valid, X_test, y_test


class MSRG(nn.Module):
    def __init__(self, config):
        super(MSRG, self).__init__()
        [self.d_l, self.d_a, self.d_v] = config["input_dims"]
        [self.dh_l, self.dh_a, self.dh_v] = config["h_dims"]

        self.gru_l = nn.GRU(self.d_l, self.dh_l, 2)
        self.gru_a = nn.GRU(self.d_a, self.dh_a, 2)
        self.gru_v = nn.GRU(self.d_v, self.dh_v, 2)

        self.project_a = nn.GRU(self.dh_a, 64, bidirectional=True, batch_first=True)
        self.project_v = nn.GRU(self.dh_v, 64, bidirectional=True, batch_first=True)
        self.project_l = nn.GRU(self.dh_l, 64, bidirectional=True, batch_first=True)

        self.bi_lstm = nn.LSTM(128*3, 128, bidirectional=True)

        self.rgn0 = RGNCell()
        self.rgn1 = RGNCell()

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x_l = x[:, :, :self.d_l]  # [20, 32, 300]
        x_a = x[:, :, self.d_l:self.d_l + self.d_a]  # [20, 32, 5]
        x_v = x[:, :, self.d_l + self.d_a:]  # [20, 32, 20]

        ##################################################################################
        ##                               特征提取模块                                     ##
        ##################################################################################
        all_x_a, h_a = self.gru_a(x_a)  #batch在第二个维度，不用设置batch_first=True
        all_x_v, h_v = self.gru_v(x_v)
        all_x_l, h_l = self.gru_l(x_l)

        #print(all_x_v.shape, all_x_l.shape, all_x_v.shape)
        x_a_origin,d = self.project_a(all_x_a.permute(1, 0, 2)) #[b,20,_] ->[b,20,64]
        x_v_origin,d = self.project_v(all_x_v.permute(1, 0, 2)) #[b,20,_]->[b,20,64]
        x_l_origin,d = self.project_l(all_x_l.permute(1, 0, 2)) #[b,20,_]->[b,20,64]
        ##################################################################################


        ##################################################################################
        ##                              情感注意力强度模块                                 ##
        ##################################################################################
        jcaf = JCAF(x_a_origin.shape[0], x_a_origin.shape[1], x_a_origin.shape[2])
        fin_txt, fin_aud, fin_vis = jcaf(x_l_origin, x_a_origin, x_v_origin)
        ##################################################################################


        ##################################################################################
        ##                              时间步长级别融合模块                               ##
        ##################################################################################
        tstep = torch.concat((fin_txt, fin_aud, fin_vis), dim=2)    #[b,20,384]
        step_fusion, (_, _) = self.bi_lstm(tstep)

        state0 = torch.randn([step_fusion.shape[0], 128])
        state1 = torch.randn([step_fusion.shape[0], 128])

        for step in torch.unbind(step_fusion, dim=1):
            state0 = out0 = self.rgn0(state0, step)
        for step in torch.unbind(torch.flip(step_fusion, dims=[1]), dim=1):
            state1 = out1 = self.rgn1(state1, step)
        ##################################################################################

        ##################################################################################
        ##                                 情感推理模块                                   ##
        ##################################################################################
        final_output = self.fc2(torch.nn.functional.leaky_relu(self.fc1(out0+out1)))
        return final_output
        ##################################################################################

def train_MSRG(X_train, y_train, X_valid, y_valid, X_test, y_test, config):
    p = np.random.permutation(X_train.shape[0])
    X_train = X_train[p]
    y_train = y_train[p]

    X_train = X_train.swapaxes(0, 1)
    X_valid = X_valid.swapaxes(0, 1)
    X_test = X_test.swapaxes(0, 1)

    model = MSRG(config)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    criterion = nn.L1Loss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.5, verbose=True)

    def train(model, batchsize, X_train, y_train, optimizer, criterion):
        epoch_loss = 0
        model.train()
        total_n = X_train.shape[1]
        num_batches = total_n // batchsize
        for batch in range(num_batches):
            start = batch * batchsize
            end = (batch + 1) * batchsize
            optimizer.zero_grad()
            batch_X = torch.Tensor(X_train[:, start:end])
            batch_y = torch.Tensor(y_train[start:end])
            predictions = model.forward(batch_X).squeeze(1)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / num_batches

    def evaluate(model, X_valid, y_valid, criterion):
        model.eval()
        with torch.no_grad():
            batch_X = torch.Tensor(X_valid)
            batch_y = torch.Tensor(y_valid)
            predictions = model.forward(batch_X).squeeze(1)
            epoch_loss = criterion(predictions, batch_y).item()
        return epoch_loss

    def predict(model, X_test):
        model.eval()
        with torch.no_grad():
            batch_X = torch.Tensor(X_test)
            predictions = model.forward(batch_X).squeeze(1)
            predictions = predictions.cpu().data.numpy()
        return predictions

    best_valid = 999999.0
    rand = random.randint(0, 100000)
    trainloss = []
    devloss = []
    for epoch in range(config["num_epochs"]):
        train_loss = train(model, config["batchsize"], X_train, y_train, optimizer, criterion)
        valid_loss = evaluate(model, X_valid, y_valid, criterion)
        scheduler.step(valid_loss)
        if valid_loss <= best_valid:
            # save model
            best_valid = valid_loss
            print(epoch, train_loss, valid_loss, "Find Better!")
            trainloss.append(train_loss)
            devloss.append(valid_loss)
            torch.save(model, 'temp_models/mfn_%d.pt' % rand)
        else:
            print(epoch, train_loss, valid_loss)
            trainloss.append(train_loss)
            devloss.append(valid_loss)

    print('model number is:', rand)
    model = torch.load('temp_models/mfn_%d.pt' % rand)

    predictions = predict(model, X_test)
    mae = np.mean(np.absolute(predictions - y_test))
    print("mae: ", mae)
    corr = np.corrcoef(predictions, y_test)[0][1]
    print("corr: ", corr)
    mult = round(sum(np.round(predictions) == np.round(y_test)) / float(len(y_test)), 5)
    print("mult_acc: ", mult)
    f_score = round(f1_score(np.round(predictions), np.round(y_test), average='weighted'), 5)
    print("mult f_score: ", f_score)
    true_label = (y_test >= 0)
    predicted_label = (predictions >= 0)
    print("Confusion Matrix :")
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :")
    print(classification_report(true_label, predicted_label, digits=5))
    print("Accuracy ", accuracy_score(true_label, predicted_label))
    print(trainloss)
    print(devloss)
    print(valid_loss)
    sys.stdout.flush()




X_train, y_train, X_valid, y_valid, X_test, y_test = load_saved_data()

config = dict()
config["input_dims"] = [300, 5, 20]
hl = random.choice([32, 64, 88, 128, 156, 256])
ha = random.choice([8, 16, 32, 48, 64, 80])
hv = random.choice([8, 16, 32, 48, 64, 80])
config["h_dims"] = [hl, ha, hv]
config["batchsize"] = random.choice([32 , 64, 128, 256])
config["lr"] = random.choice([0.001, 0.002, 0.005, 0.008, 0.01])

# 超参数
config["lr"] = 0.005
config["num_epochs"] = 21
config["batchsize"] = 32
config["h_dims"] = [190, 95, 95]
print(config)

# 记录运行时间
start = time.clock()
train_MSRG(X_train, y_train, X_valid, y_valid, X_test, y_test, config)
end = time.clock()
print("运行时间：", end-start)

