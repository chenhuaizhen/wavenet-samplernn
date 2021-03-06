import tensorflow as tf
import numpy as np
import os
import scipy.io.wavfile as wav
import random
from model import SampleRnnModel

trainTime = 100000
batch_size = 64
frame_size = 16
q_levels = 256
rnn_type = 'GRU'
rnn_dim = 1024
n_rnn = 3
len_of_data = 512*200
len_of_bptt = 512
emb_size = 256

rate_of_wav = 16000
l2_regularization_strength = 0
learning_rate = 1e-3
modelAdd = "Model/model.ckpt"
train_data_add = "../music/music_train.npy"
valid_data_add = "../music/music_valid.npy"
is_init = True

def _normalize(data):
    """To range [0., 1.]"""
    data -= data.min(axis=1)[:, None]
    data /= data.max(axis=1)[:, None]
    return data

def initData():
    # mean_std = np.load("music/music_train_mean_std.npy")
    # mean = mean_std[0]
    # std = mean_std[1]
    train_data = np.load(train_data_add)
    valid_data = np.load(valid_data_add)
    # train_data = (train_data - mean) / std
    # valid_data = (valid_data - mean) / std
    train_data = _normalize(train_data)
    valid_data = _normalize(valid_data)
    eps = np.float64(1e-5)

    train_data *= (255. - eps)
    train_data += eps / 2

    valid_data *= (255. - eps)
    valid_data += eps / 2

    trainData = train_data[:, 1:] - train_data[:, :-1] + 128
    validData = valid_data[:,1:] - valid_data[:,:-1] + 128

    trainData = trainData.astype(np.int32)
    validData = validData.astype(np.int32)

    return validData,trainData

def getData(data,length):
    index = random.randint(0, len(data)-1)
    start = random.randint(0,len(data[index])-length-2)
    return data[index][start:start+length]

def getBatchData(data,batch_size,length):
    batch_data = []
    for i in range(batch_size):
        batch_data.append(getData(data,length))
    batch_data = np.array(batch_data).reshape([batch_size,length,1])
    return batch_data

data_input = tf.placeholder("float", shape=[None,len_of_bptt,1],name="X-input")
cell_state = tf.placeholder("float",shape=[n_rnn,batch_size,rnn_dim])

def main():
    ValidData, TrainData = initData()
    print("data done")
    # Create coordinator.
    coord = tf.train.Coordinator()

    # Create network.
    net = SampleRnnModel(
        batch_size=batch_size,
        frame_size=frame_size,
        q_levels=q_levels,
        rnn_type=rnn_type,
        dim=rnn_dim,
        n_rnn=n_rnn,
        seq_len=len_of_bptt,
        emb_size=emb_size)

    state = tf.unstack(cell_state, axis=0)
    rnn_tuple_state = tuple(
        [state[id] for id in range(n_rnn)]
    )
    loss,accuracy,final_frame_state = net.loss(data_input,
                                               rnn_tuple_state,
                                               l2_regularization_strength)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver()
    if(not is_init):
        saver.restore(sess, modelAdd)

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ValidMax = 10000
    for step in range(trainTime):
        data = getBatchData(TrainData, batch_size, len_of_data)
        _final_frame = np.zeros([n_rnn,batch_size,rnn_dim], dtype=np.float32)
        index = 0
        meanLoss = 0.
        meanAcc = 0.
        for x in range(len_of_data // len_of_bptt):
            bpttData = data[:, index:index + len_of_bptt, :]
            _total_loss, _train_step, _acc, _final_frame = sess.run(
                [loss, optim, accuracy, final_frame_state],
                feed_dict={
                    data_input: bpttData,
                    cell_state: _final_frame
                })
            print("SStep:", step, x,"loss:", _total_loss, "acc:", _acc)
            index += len_of_bptt - frame_size
            meanLoss += _total_loss
            meanAcc += _acc

        print("Step:", step, "loss:", meanLoss / (len_of_data / len_of_bptt), "acc:",
              meanAcc / (len_of_data / len_of_bptt))
        if step % 5 == 0:
            data = getBatchData(ValidData, batch_size, len_of_bptt)
            validLoss, validAcc = sess.run(
                [loss, accuracy],
                feed_dict={
                    data_input: data,
                    cell_state: np.zeros([n_rnn,batch_size,rnn_dim], dtype=np.float32)
                })
            if (validLoss < ValidMax):
                ValidMax = validLoss
                saver.save(sess, modelAdd)

            print("ValidLoss:", validLoss, "ValidAcc:", validAcc)

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
