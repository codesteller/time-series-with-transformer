import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import argparse


parser = argparse.ArgumentParser(description='Transformer model for time-series forecasting')
parser.add_argument('--batch_size', type=int, default=50, help='')
#parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
args = parser.parse_args()

#TODO
# gpu_devices = ','.join([str(id) for id in args.gpu_devices])
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

tf.random.set_seed(0)
np.random.seed(0)
input_window = 100
output_window = 5

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits = tf.where(tf.equal(scaled_attention_logits, 0.0), tf.ones_like(scaled_attention_logits) * -1e9, scaled_attention_logits)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model)
  ])

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    def call(self, x, mask, training):
        attn_output = self.self_att(x, x, x, mask)
        attn_output = x + self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, enc_layers, num_layers):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.enc_layers = [enc_layers for _ in range(num_layers)]
    def call(self, x, mask, training):
        for mod in self.enc_layers:
          x = mod(x, mask, training)
        return x

"""
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pos = np.arange(max_len)[:, np.newaxis]
        i =  np.arange(d_model)[np.newaxis, :]
        pe = pos * (1 / np.power(10000, (2 * (i//2)) / np.float32(d_model)))
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        self.pe = pe[np.newaxis, ...]
    def forward(self, x):
        return x + tf.cast(self.pe[:,:tf.shape(x)[1], :], dtype=tf.float32)"""

# TODO: Can convert this function into a graph example shown above
#(mine was not working may be some problem in decalaraction)

def PositionalEncoding(x, d_model, max_len=5000):
    pos = np.arange(max_len)[:, np.newaxis]
    i =  np.arange(d_model)[np.newaxis, :]
    pe = pos * (1 / np.power(10000, (2 * (i//2)) / np.float32(d_model)))
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    pe = pe[np.newaxis, ...]
    return x + tf.cast(pe[:,:tf.shape(x)[1], :], dtype=tf.float32)

class TransAM(tf.keras.Model):
    def __init__(self, feature_size=250, num_layers=1, num_heads=10, dim_feedforward=2048, dropout=0.1):
        super(TransAM, self).__init__()
        self.src_mask = None
        self.feature_size = feature_size
        self.encoder_layer = TransformerEncoderLayer(d_model=feature_size, num_heads=num_heads, dff=dim_feedforward, rate=dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder =  tf.keras.layers.Dense(1)

    def call(self, src, training=True):
        if self.src_mask is None or tf.shape(self.src_mask)[0] != tf.shape(src)[1]:
            mask = 1 - tf.linalg.band_part(tf.ones((tf.shape(src)[1], tf.shape(src)[1])), -1, 0)
            cond = mask == 0.0
            mask = tf.where(cond, mask, float("-inf"))
            self.src_mask = mask

        src = PositionalEncoding(src, self.feature_size)
        output = self.transformer_encoder(src, self.src_mask, training=training)
        output = self.decoder(output)
        return output

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = np.append(input_data[i:i+tw][:-output_window] , output_window * [0])
        train_label = input_data[i:i+tw]
        #train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq ,train_label))
    return tf.cast(inout_seq, dtype=tf.float32)

def get_data():
    series = read_csv('daily-min-temperatures_new.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    amplitude = series.to_numpy()
    amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    samples = 8000
    train_data = amplitude[:samples]
    test_data = amplitude[samples:]
    train_sequence = create_inout_sequences(train_data,input_window)
    train_sequence = train_sequence[:-output_window] #todo: fix hack?
    test_data = create_inout_sequences(test_data,input_window)
    test_data = train_sequence[:-output_window] #todo: fix hack?
    return train_sequence, test_data

def get_batch(source, i,batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]
    input = tf.stack([item[0] for item in data])# 1 is feature size
    target = tf.stack([item[1] for item in data])
    input = tf.expand_dims(tf.transpose(input), axis=2)
    target = tf.expand_dims(tf.transpose(target), axis=2)
    return input, target


def train(train_data, model, optimizer, epoch, batch_size, loss_object):
    total_loss = 0
    start_time=time.time()
    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i,batch_size)
        data, targets = tf.transpose(targets, perm=[1, 0, 2]), tf.transpose(targets, perm=[1, 0, 2])
        with tf.GradientTape() as tape:
            output = model(data, training=True)
            loss = loss_object(targets, output)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            total_loss += loss.numpy()
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, optimizer.learning_rate(optimizer.iterations),
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source, loss_object):
    total_loss = 0.
    eval_batch_size = 50
    for i in range(0, len(data_source) - 1, eval_batch_size):
        data, targets = get_batch(data_source, i, eval_batch_size)
        data, targets = tf.transpose(targets, perm=[1, 0, 2]), tf.transpose(targets, perm=[1, 0, 2])
        output = eval_model(data, training=False)
        total_loss += len(data[0])* loss_object(output, targets).numpy()
    return total_loss / i


#TODO: cuda device
def main():
    batch_size = args.batch_size
    train_data, val_data = get_data()
    trans_model  = TransAM(feature_size=250, num_layers=1, num_heads=10, dim_feedforward=2048, dropout=0.1)
    #TODO: Dataparallel like in tensoflow
    model = trans_model

    #TODO: Automate lr_scedular like StepLR
    boundaries = [30, 60] #step_size =30
    values = [0.05, 0.0005, 0.00005] # gamma= 0.1
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss_object = tf.keras.losses.MeanSquaredError()
    epochs = 100
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train(train_data, model, optimizer, epoch, batch_size, loss_object)
        val_loss = evaluate(model, val_data, loss_object)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print('-' * 89)

if __name__ == "__main__":
    main()
