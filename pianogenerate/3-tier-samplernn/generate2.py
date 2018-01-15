import tensorflow as tf
import numpy as np
import os
import scipy.io.wavfile as wav
import random
from model import SampleRnnModel

batch_size = 5
big_frame_size = 8
frame_size = 2
q_levels = 256
rnn_type = 'GRU'
rnn_dim = 512
n_rnn = 1
emb_size = 256
rate_of_wav = 16000
len_of_data = int(rate_of_wav * 5)
len_of_data_first = 520

l2_regularization_strength = 0
modelAdd = "Model1/model.ckpt"
saveAdd = "output.wav"
test_data_add = "../music/music_test.npy"

def _normalize(data):
    """To range [0., 1.]"""
    data -= data.min(axis=1)[:, None]
    data /= data.max(axis=1)[:, None]
    return data

def initData():
    test_data = np.load(test_data_add)
    test_data = _normalize(test_data)
    eps = np.float64(1e-5)
    test_data *= (255. - eps)
    test_data += eps / 2
    # testData = test_data[:,1:] - test_data[:,:-1] + 128
    testData = test_data.astype(np.int32)
    return testData

def generate_and_save_samples(start, net, infe_para, sess, length):
    samples = np.zeros((net.batch_size, length, 1), dtype='int32')
    samples[:, :net.big_frame_size,:] = start

    final_big_s,final_s = sess.run([net.big_initial_state,net.initial_state])
    big_frame_out = None
    frame_out = None
    sample_out = None
    for t in range(net.big_frame_size, length):
        print(t,length)
        #big frame
        if t % net.big_frame_size == 0:
            big_frame_out = None
            big_input_sequences = samples[:, t-net.big_frame_size:t,:].astype('float32')
            big_frame_out, final_big_s= \
            sess.run([infe_para['infe_big_frame_outp'] ,
                infe_para['infe_final_big_frame_state'] ],
                   feed_dict={
                      infe_para['infe_big_frame_inp'] : big_input_sequences,
                      infe_para['infe_big_frame_state'] : final_big_s})
        #frame
        if t % net.frame_size == 0:
            frame_input_sequences = samples[:, t-net.frame_size:t,:].astype('float32')
            big_frame_output_idx = (t/net.frame_size)%(net.big_frame_size/net.frame_size)
            frame_out, final_s= \
            sess.run([infe_para['infe_frame_outp'],
                infe_para['infe_final_frame_state']],
                  feed_dict={
        infe_para['infe_big_frame_outp_slices'] : big_frame_out[:,[big_frame_output_idx],:],
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
    # for i in range(0, net.batch_size):
    #     temp = samples[i].astype(np.float32)
    #     temp = temp - 128.
    #     samples[i] = temp
        # samples[i] = np.ceil(32768*np.sign(temp/128)*((np.power(256,np.abs(temp/128))-1)/255)).astype(np.int16)
    return samples

data_input = tf.placeholder("float", shape=[None,len_of_data,1],name="X-input")

def main():
    testData = initData()
    # Create coordinator.
    coord = tf.train.Coordinator()
    with tf.variable_scope(tf.get_variable_scope()):
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
        # tf.get_variable_scope().reuse_variables()

    # loss, accuracy, final_big_frame_state, final_frame_state = net.loss(
    #     data_input,
    #     net.big_cell.zero_state(net.batch_size, tf.float32),
    #     net.cell.zero_state(net.batch_size, tf.float32),
    #     l2_regularization_strength)
    # print("done")
    # infe_para = net.create_gen_wav_para()
    # tf.get_variable_scope().reuse_variables()

    rel_output = net.getOutput(data_input)

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

    start = testData[:batch_size,:net.big_frame_size*2-1].reshape([batch_size,-1,1])
    print("Start generating.")
    result = np.zeros([batch_size,len_of_data,1], dtype='int32')
    result[:,:net.big_frame_size*2-1,1] = start
    for i in range(net.big_frame_size*2-1,len_of_data):
        print(i,len_of_data)
        result[:, i, :] = sess.run(rel_output,feed_dict={
            data_input:result[:,i-net.big_frame_size*2+1:i+1,1]
        })

    for i in range(batch_size):
        wav.write("output"+str(i)+".wav", rate_of_wav, result[i].reshape([-1]))
    print("Finished generating.")
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()