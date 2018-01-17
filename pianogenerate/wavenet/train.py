import tensorflow as tf
import numpy as np
import os
import scipy.io.wavfile as wav
import random
from model import WaveNetModel

trainTime = 100000
batch_size = 2
rate_of_wav = 16000
len_of_data = rate_of_wav * 5
dilations = [ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
filter_width = 2
residual_channels = 32
dilation_channels = 32
skip_channels = 512
quantization_channels = 256
use_biases = True
l2_regularization_strength = 0
learning_rate = 1e-3
is_init = True
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

    tD = trainData.astype(np.int32)
    vD = validData.astype(np.int32)

    return vD,tD

def getData(data,length):
    index = random.randint(0, len(data)-1)
    start = random.randint(0,len(data[index])-length-2)
    return data[index][start:start+length],data[index][start+1:start+length+1]

def getBatchData(data,batch_size,length):
    batch_data = []
    batch_label = []
    for i in range(batch_size):
        _data,_label = getData(data,length)
        batch_data.append(_data)
        batch_label.append(_label)
    batch_data = np.array(batch_data).reshape([batch_size,length])
    batch_label = np.array(batch_label).reshape([batch_size, length])
    return batch_data,batch_label

data_input = tf.placeholder("float", shape=[None,len_of_data],name="X-input")
label_input = tf.placeholder("float", shape=[None,len_of_data],name="X-input")

def main():
    ValidData, TrainData = initData()
    print("data done")
    # Create coordinator.
    coord = tf.train.Coordinator()

    # Create network.
    net = WaveNetModel(
        batch_size=batch_size,
        dilations=dilations,
        filter_width=filter_width,
        residual_channels=residual_channels,
        dilation_channels=dilation_channels,
        skip_channels=skip_channels,
        quantization_channels=quantization_channels,
        use_biases=use_biases)

    loss,accuracy = net.loss(data_input, label_input, l2_regularization_strength)
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

    ValidMax = 100
    for step in range(trainTime):
        data, label = getBatchData(TrainData, batch_size, len_of_data)
        _total_loss, _train_step, _acc = sess.run(
            [loss, optim, accuracy],
            feed_dict={
                data_input: data,
                label_input: label
            })

        print("WaveStep:", step, "loss:", _total_loss, "acc:", _acc)
        if step % 1000 == 0:
            data, label = getBatchData(ValidData, batch_size, len_of_data)
            validLoss,validAcc = sess.run(
                [loss,accuracy],
                feed_dict={
                    data_input: data,
                    label_input: label
                })
            if (validLoss < ValidMax):
                ValidMax = validLoss
                saver.save(sess, modelAdd)

            print("WaveValidLoss:",validLoss,"WaveValidAcc:", validAcc)

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()