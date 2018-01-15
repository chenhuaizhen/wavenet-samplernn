import tensorflow as tf
import numpy as np
import os
import scipy.io.wavfile as wav
import random
from testmodel import SampleRnnModel

trainTime = 10
batch_size = 1
big_frame_size = 8
frame_size = 2
q_levels = 256
rnn_type = 'GRU'
rnn_dim = 512
n_rnn = 1
len_of_data = 520*200
len_of_bptt = 520
emb_size = 256
is_init = False

rate_of_wav = 16000
l2_regularization_strength = 0
learning_rate = 1e-3
modelAdd = "Model/model.ckpt"
train_data_add = "../music/music_train.npy"
valid_data_add = "../music/music_valid.npy"

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

    trainData = train_data[:,1:] - train_data[:,:-1] + 128
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
big_cell_state = tf.placeholder("float",shape=[batch_size,rnn_dim])
cell_state = tf.placeholder("float",shape=[batch_size,rnn_dim])

def main():
    ValidData, TrainData = initData()
    print("data done")
    # Create coordinator.
    coord = tf.train.Coordinator()

    # Create network.
    net = SampleRnnModel(
        batch_size=batch_size,
        big_frame_size=big_frame_size,
        frame_size=frame_size,
        q_levels=q_levels,
        rnn_type=rnn_type,
        dim=rnn_dim,
        n_rnn=n_rnn,
        seq_len=len_of_bptt,
        emb_size=emb_size)

    # if(n_rnn>1):
    #     bigState = tf.unstack(big_cell_state, axis=0)
    #     state = tf.unstack(cell_state, axis=0)
    #     big_state = tuple(
    #         [bigState[id] for id in range(n_rnn)]
    #     )
    #     _state = tuple(
    #         [state[id] for id in range(n_rnn)]
    #     )
    loss, accuracy, final_big_frame_state, final_frame_state,_p,_t = net.loss(
        data_input, big_cell_state, cell_state,
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
    if (not is_init):
        saver.restore(sess, modelAdd)

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ValidMax = 10000
    for step in range(trainTime):
        data = getBatchData(TrainData, batch_size, len_of_data)
        _final_big = np.zeros([batch_size,rnn_dim], dtype=np.float32)
        _final_frame = np.zeros([batch_size,rnn_dim], dtype=np.float32)
        index = 0
        meanLoss = 0.
        meanAcc = 0.
        for x in range(len_of_data//len_of_bptt):
            print(step,x,200)
            bpttData = data[:,index:index+len_of_bptt,:]
            _total_loss, _acc, _final_big, _final_frame, _pre,_tar = sess.run(
                [loss, accuracy, final_big_frame_state, final_frame_state,_p,_t],
                feed_dict={
                    data_input: bpttData,
                    big_cell_state: _final_big,
                    cell_state: _final_frame
                })
        #     print("SStep:", step, x,"loss:", _total_loss, "acc:", _acc)
            index += len_of_bptt - big_frame_size
            with open("pre.txt","a") as file:
                for p in _pre:
                    file.write(str(p)+",")
            with open("tar.txt", "a") as file:
                for t in _tar:
                    file.write(str(t)+',')
        with open("pre.txt", "a") as file:
            file.write("\n")
        with open("tar.txt", "a") as file:
            file.write("\n")
        #     meanLoss += _total_loss
        #     meanAcc += _acc
        #
        # print("Step:", step, "loss:", meanLoss/(len_of_data/len_of_bptt), "acc:", meanAcc/(len_of_data/len_of_bptt))


        # print("Step:", step, "loss:", _total_loss, "acc:", _acc)

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()