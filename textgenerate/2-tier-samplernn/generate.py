import tensorflow as tf
import numpy as np
import random
import io
from model import SampleRnnModel

batch_size = 10
frame_size = 16
q_levels = 4118
rnn_type = 'GRU'
rnn_dim = 1024
n_rnn = 3
emb_size = 50
len_of_data = 1024

l2_regularization_strength = 0
modelAdd = "Model/model.ckpt"
dataAdd = "../data.txt"
startAdd = "start.txt"
saveDict = "dict.npy"
saveReDict = "reDict.npy"

def initDict(dictAdd,reDictAdd):
    dict = {}
    reDict = {}
    d1 = np.load(dictAdd)
    d2 = np.load(reDictAdd)
    for i in d1.item():
        dict[i] = d1.item().get(i)
    for i in d2.item():
        reDict[i] = d2.item().get(i)
    return dict,reDict

def tranToData(data,dict):
    res = ""
    for d in data:
        res += dict[d]
    return res

def initStartCtx(fileAdd,dict):
    output = []
    with io.open(fileAdd,mode="r",encoding="utf-8") as file:
        txt = file.read()
        for t in txt:
            output.append(dict[t])
    output = np.array(output).astype(np.int32)
    return output

def generate_and_save_samples(startCtx, net, infe_para, sess, length):
    samples = np.zeros((net.batch_size, length, 1), dtype='int32')
    samples[:, :net.frame_size,:] = startCtx

    final_s = sess.run([net.initial_state])
    frame_out = None
    sample_out = None
    for t in range(net.frame_size, length):
        print(t,length)
        #frame
        if t % net.frame_size == 0:
            frame_input_sequences = samples[:, t-net.frame_size:t,:].astype('float32')
            frame_out, final_s= \
            sess.run([infe_para['infe_frame_outp'],
                infe_para['infe_final_frame_state']],
                  feed_dict={
        infe_para['infe_frame_inp'] : frame_input_sequences,
        infe_para['infe_frame_state'] : final_s})
        #sample
        sample_input_sequences = samples[:, t-net.frame_size:t,:]
        frame_output_idx = t%net.frame_size
        sample_out= \
        sess.run(infe_para['infe_sample_outp'],
                 feed_dict={
                    infe_para['infe_frame_outp_slices'] : frame_out[:,[frame_output_idx],:],
                    infe_para['infe_sample_inp'] : sample_input_sequences})
        sample_next_list = []
        for row in sample_out:
            sample_next = np.random.choice(
                np.arange(net.q_levels), p=row )
            sample_next_list.append(sample_next)
        samples[:, t] = np.array(sample_next_list).reshape([-1,1])

    return samples

def main():
    # Create coordinator.
    coord = tf.train.Coordinator()
    with tf.variable_scope(tf.get_variable_scope()):
        # Create network.
        net = SampleRnnModel(
            batch_size=batch_size,
            frame_size=frame_size,
            q_levels=q_levels,
            rnn_type=rnn_type,
            dim=rnn_dim,
            n_rnn=n_rnn,
            seq_len=len_of_data,
            emb_size=emb_size)

    infe_para = net.create_gen_wav_para()

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('infe' in var.name)}
    saver = tf.train.Saver(variables_to_restore)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    saver.restore(sess, modelAdd)

    Dict, resDict = initDict(saveDict,saveReDict)
    startCtx = initStartCtx(startAdd, Dict)
    start = np.tile(startCtx[:net.frame_size], [batch_size, 1]) \
        .reshape([batch_size, net.frame_size, 1])
    print(start)
    print("Start generating.")
    result = generate_and_save_samples(start, net, infe_para, sess, len_of_data)
    result = np.reshape(result, [batch_size, len_of_data])
    for i in range(batch_size):
        with io.open(("output" + str(i) + ".txt"), mode="a", encoding="utf-8") as file:
            for ii in result[i]:
                file.write(resDict[ii])

    print("Finished generating.")
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()