import tensorflow as tf
import numpy as np
import os
import scipy.io.wavfile as wav
import random
from model import SampleRnnModel

trainTime = 100000
batch_size = 32
big_frame_size = 8
frame_size = 2
q_levels = 4119
rnn_type = 'GRU'
rnn_dim = 512
n_rnn = 1
len_of_data = 520
emb_size = 50
is_init = True

rate_of_wav = 16000
l2_regularization_strength = 0
learning_rate = 1e-3
modelAdd = "Model/model.ckpt"
fileAdd = "../data.txt"

def initData(fileAdd):
    dict = {}
    with open(fileAdd, "r") as file:
        txt = file.read()
        index = 0
        for t in txt:
            if (t not in dict):
                dict[t] = index
                index += 1
        return txt,dict

def tranToDict(data,dict):
    res = []
    for d in data:
        res.append(dict[d])
    return np.array(res)

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

data_input = tf.placeholder("float", shape=[None,len_of_data,1],name="X-input")

def main():
    TrainData, dictionary = initData(fileAdd)
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
        seq_len=len_of_data,
        emb_size=emb_size)

    loss,accuracy,final_big_frame_state,final_frame_state = net.loss(
                                                    data_input,
                                                    net.big_cell.zero_state(net.batch_size, tf.float32),
                                                    net.cell.zero_state(net.batch_size, tf.float32),
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
    wholeLen = len(TrainData)
    start = -1
    data = np.zeros([batch_size, len_of_data, 1])
    for step in range(trainTime):
        for t in range(batch_size):
            start = TrainData.find(".", start + 1)
            if (start + len_of_data > wholeLen):
                start = TrainData.find(".", 0)
            data[t, :] = tranToDict(TrainData[start + 1:start + len_of_data + 1], dictionary).reshape(
                [len_of_data, 1]).astype(np.int32)
        _total_loss, _train_step, _acc = sess.run(
            [loss, optim, accuracy],
            feed_dict={
                data_input: data
            })

        print("Step:", step, "loss:", _total_loss, "acc:", _acc)
        if step % 1000 == 0:
            # data = getBatchData(ValidData, batch_size, len_of_data)
            # validLoss,validAcc = sess.run(
            #     [loss,accuracy],
            #     feed_dict={
            #         data_input: data
            #     })
            # if (validLoss < ValidMax):
            #     ValidMax = validLoss
            saver.save(sess, modelAdd)

            # print("ValidLoss:", validLoss, "ValidAcc:", validAcc)

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()