import tensorflow as tf
import numpy as np
import random
import io
from model import WaveNetModel

trainTime = 100000
batch_size = 64
len_of_data = 1024
dilations = [ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
filter_width = 2
residual_channels = 32
dilation_channels = 32
skip_channels = 512
quantization_channels = 4118
embedding_channels = 256
use_biases = True
l2_regularization_strength = 0
learning_rate = 1e-3
modelAdd = "Model/model.ckpt"
fileAdd = "../data.txt"
isInit = True
saveDict = "dict.npy"
saveReDict = "reDict.npy"

def initData(fileAdd):
    dict = {}
    reDict = {}
    with io.open(fileAdd, mode="r", encoding="utf-8") as file:
        txt = file.read()
        index = 0
        for t in txt:
            if (t not in dict):
                dict[t] = index
                reDict[index] = t
                index += 1
        print(len(dict),len(reDict)) # 4118
        np.save(saveReDict,reDict)
        np.save(saveDict,dict)
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

data_input = tf.placeholder("float", shape=[None,len_of_data],name="X-input")
label_input = tf.placeholder("float", shape=[None,len_of_data],name="X-input")

def main():
    TrainData,dictionary = initData(fileAdd)
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
        embedding_channels=embedding_channels,
        use_biases=use_biases)

    loss, accuracy = net.loss(data_input, label_input, l2_regularization_strength)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    saver = tf.train.Saver()
    if(isInit):
        init = tf.global_variables_initializer()
        sess.run(init)
    else:
        # Saver for storing checkpoints of the model.
        saver.restore(sess, modelAdd)

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    wholeLen = len(TrainData)
    start = -1

    data = np.zeros([batch_size, len_of_data])
    label = np.zeros([batch_size, len_of_data])
    for step in range(trainTime):
        for t in range(batch_size):
            start = TrainData.find(".", start + 1)
            if (start + len_of_data + 2 > wholeLen):
                start = TrainData.find(".", 0)
            wholedata = tranToDict(TrainData[start + 1:start + len_of_data + 2], dictionary).astype(np.int32)
            data[t,:] = wholedata[:-1]
            label[t,:] = wholedata[1:]

        _total_loss, _train_step, _acc = sess.run(
            [loss, optim, accuracy],
            feed_dict={
                data_input: data,
                label_input: label
            })

        print("Step:", step, "loss:", _total_loss, "acc:", _acc)
        if step % 1000 == 0:
            saver.save(sess, modelAdd)

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
