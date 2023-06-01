import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

total_words = 10000  # 词汇表大小N_vocab
max_review_len = 250  # 最大句子长度，大于的句子部分将截断，小的填充
batchsz = 32  # 批量大小
embedding_len = 256  # 词向量的特征长度

def get_data():
    # 加载数据集 此处的数字采用数字编码，一个数字代表一个单词
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=total_words)
    # 打印输入的形状， 标签的形状
    print(x_train.shape, len(x_train[0]), y_train.shape)
    print(x_test.shape, len(x_test[0]), y_test.shape)
    return x_train, y_train, x_test, y_test

def process_data(x_train, y_train, x_test, y_test):
    # 截断和填充句子，使得等长，此处长句子保留句子后面的部分，短句子在前面填充
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_review_len)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_review_len)

    # 构建数据集，打散，批量，并丢掉最后一个不够 batchsz 的 batch
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.batch(batchsz, drop_remainder=True)
    # 统计数据集属性
    print('x_train shape:', x_train.shape, tf.reduce_max(y_train),
          tf.reduce_min(y_train))
    print('x_test shape:', x_test.shape, tf.reduce_max(y_test),
          tf.reduce_min(y_test))
    return db_train, db_test

class RGNCell(tf.keras.Model):

    def __init__(self, unites):
        super(RGNCell,self).__init__()
        self.dense1=tf.keras.layers.Dense(2*unites, input_shape=(256, ))
        self.dense2=tf.keras.layers.Dense(unites, input_shape=(128, ))
        self.dense3=tf.keras.layers.Dense(unites, input_shape=(256, ))
        self.dense4=tf.keras.layers.Dense(unites, input_shape=(256, ))
        self.dense5=tf.keras.layers.Dense(unites, input_shape=(256, ), use_bias=False)

    def call(self,s_t1,x_t):

        r_t = tf.nn.relu(self.dense1(x_t))
        m_t = tf.nn.tanh(self.dense2(r_t))
        forget = tf.multiply(m_t, s_t1)

        i_t = tf.nn.sigmoid(self.dense3(x_t))                                           #64
        g_t = tf.nn.tanh(self.dense4(x_t))
        read = tf.multiply(i_t, g_t)                                                    #64

        y_t = tf.nn.relu(forget + read)                                                 #64
        s_t = tf.nn.relu(tf.nn.relu(self.dense5(x_t)) + y_t)                            #64

        return s_t


class MyRGN(tf.keras.Model):
    # Cell 方式构建多层网络
    # 构建分类网络，用于将 CELL 的输出特征进行分类，2 分类
    # # [b, 250, 256] => [b, 64] => [b, 1]
    def __init__(self, unites):
        super(MyRGN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(total_words, embedding_len, input_length=max_review_len)

        self.state0 = tf.random.normal([batchsz, unites])
        self.state1 = tf.random.normal([batchsz, unites])
        self.rmn_cell0 = RGNCell(unites)
        self.outlayer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = inputs # [b, 250]
        # 获取词向量: [b, 250] => [b, 250, 256]
        x = self.embedding(x)

        # 通过 2 个 RGN CELL,[b, 250, 256] => [b, 64]
        state0 = self.state0
        for word in tf.unstack(x, axis=1):  # word: [128, 128] 250个
            state0 = out0 = self.rmn_cell0(state0, word)

        # 末层最后一个输出作为分类网络的输入: [b, 64] => [b, 1]
        out = self.outlayer(out0)
        # 通过激活函数，p(y is pos|x)
        prob = tf.sigmoid(out)
        return prob

def main():
    epochs = 10
    unites = 64
    #加载数据集
    x_train, y_train, x_test, y_test = get_data()
    #数据预处理
    db_train, db_test = process_data(x_train, y_train, x_test, y_test)
    model = MyRGN(unites)  # 创建模型
    # 模型装配
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                 loss=tf.keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    #训练和验证

    history = model.fit(db_train, epochs=epochs, validation_data=db_test)
    print(history)

    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
    y_test = tf.expand_dims(y_test, -1)
    x_test = x_test[:batchsz*100, :]
    y_test = y_test[:batchsz*100, :]

    loss, acc = model.evaluate(x_test, y_test)
    test_predictions = model.predict(x_test)
    true_labels = y_test
    recall = recall_score(true_labels, test_predictions.round())
    f1 = f1_score(true_labels, test_predictions.round())
    precision = precision_score(true_labels, test_predictions.round())

    print('accuracy: ', acc)
    print('loss: ', loss)
    print('recall: ', recall)
    print('precision: ', precision)
    print('f1: ', f1)

    sns.lineplot(x=history.epoch, y=history.history['loss'])
    plt.show()

main()