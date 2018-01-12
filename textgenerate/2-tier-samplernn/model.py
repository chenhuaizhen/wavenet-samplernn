import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops


class SampleRnnModel(object):
    def __init__(self, batch_size, frame_size,
                 q_levels, rnn_type, dim, n_rnn, seq_len, emb_size):
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.q_levels = q_levels
        self.rnn_type = rnn_type
        self.dim = dim
        self.n_rnn = n_rnn
        self.seq_len = seq_len
        self.emb_size = emb_size

        def single_cell():
            return tf.contrib.rnn.GRUCell(self.dim)

        if 'LSTM' == self.rnn_type:
            def single_cell():
                return tf.contrib.rnn.BasicLSTMCell(self.dim)
        self.cell = single_cell()
        if self.n_rnn > 1:
            self.cell = tf.contrib.rnn.MultiRNNCell(
                [single_cell() for _ in range(self.n_rnn)])
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=self.q_levels,
                dtype=tf.float32)
            shape = [self.batch_size, -1, self.q_levels]
            encoded = tf.reshape(encoded, shape)
        return encoded

    def _create_network_Frame(self,
                              num_steps,
                              frame_state,
                              input_sequences):
        with tf.variable_scope('Frame_layer'):
            input_frames = tf.reshape(input_sequences, [
                tf.shape(input_sequences)[0],
                tf.shape(input_sequences)[1] / self.frame_size,
                self.frame_size])
            input_frames = (input_frames / (self.q_levels / 2.0)) - 1.0
            input_frames *= 2.0

            frame_outputs = []
            frame_proj_weights = tf.get_variable(
                "frame_proj_weights", [self.dim, self.dim * self.frame_size], dtype=tf.float32)

            with tf.variable_scope("FRAME_RNN"):
                for time_step in range(num_steps):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    cell_input = tf.reshape(input_frames[:, time_step, :], [-1, self.frame_size])
                    (frame_cell_output, frame_state) = self.cell(cell_input, frame_state)
                    frame_outputs.append(math_ops.matmul(frame_cell_output, frame_proj_weights))
            final_frame_state = frame_state
            frame_outputs = tf.stack(frame_outputs)
            frame_outputs = tf.transpose(frame_outputs, perm=[1, 0, 2])

            frame_outputs = tf.reshape(frame_outputs,
                                       [tf.shape(frame_outputs)[0],
                                        tf.shape(frame_outputs)[1] * self.frame_size,
                                        -1])
            return frame_outputs, final_frame_state

    def _create_network_Sample(self,
                               frame_outputs,
                               sample_input_sequences):
        with tf.variable_scope('Sample_layer'):
            sample_shap = [tf.shape(sample_input_sequences)[0],
                           tf.shape(sample_input_sequences)[1] * self.emb_size,
                           1]
            embedding = tf.get_variable("embedding", [self.q_levels, self.emb_size])
            sample_input_sequences = tf.one_hot(tf.reshape(sample_input_sequences, [-1]),
                                                            depth=self.q_levels,
                                                            dtype=tf.float32)
            sample_input_sequences = tf.matmul(sample_input_sequences,embedding)
            # sample_input_sequences = embedding_ops.embedding_lookup(
            #                            embedding, tf.reshape(sample_input_sequences,[-1]))
            sample_input_sequences = tf.reshape(sample_input_sequences, sample_shap)

            '''Create a convolution filter variable with the specified name and shape,
            and initialize it using Xavier initialition.'''
            filter_initializer = tf.contrib.layers.xavier_initializer_conv2d()
            sample_filter_shape = [self.emb_size * self.frame_size, 1, self.dim]
            sample_filter = tf.get_variable("sample_filter", sample_filter_shape,
                                            initializer=filter_initializer)
            out = tf.nn.conv1d(sample_input_sequences,
                               sample_filter,
                               stride=self.emb_size,
                               padding="VALID",
                               name="sample_conv")

            out = out + frame_outputs
            sample_mlp1_weights = tf.get_variable(
                "sample_mlp1", [self.dim, self.dim], dtype=tf.float32)
            sample_mlp2_weights = tf.get_variable(
                "sample_mlp2", [self.dim, self.dim], dtype=tf.float32)
            sample_mlp3_weights = tf.get_variable(
                "sample_mlp3", [self.dim, self.q_levels], dtype=tf.float32)
            out = tf.reshape(out, [-1, self.dim])
            out = math_ops.matmul(out, sample_mlp1_weights)
            out = tf.nn.relu(out)
            out = math_ops.matmul(out, sample_mlp2_weights)
            out = tf.nn.relu(out)
            out = math_ops.matmul(out, sample_mlp3_weights)
            out = tf.reshape(out, [-1, sample_shap[1] / self.emb_size - self.frame_size + 1, self.q_levels])
            return out

    def _create_network_SampleRnn(self,
                                  train_frame_state):
        with tf.name_scope('SampleRnn_net'):
            # frame
            input_sequences = tf.cast(self.encoded_input_rnn, tf.float32)[:, :-self.frame_size, :]
            frame_num_steps = (self.seq_len-self.frame_size) / self.frame_size
            frame_outputs, final_frame_state = \
                self._create_network_Frame(num_steps=frame_num_steps,
                                           frame_state=train_frame_state,
                                           input_sequences=input_sequences)
            # sample
            sample_input_sequences = self.encoded_input_rnn[:, :-1, :]
            sample_output = self._create_network_Sample(frame_outputs,
                                                        sample_input_sequences=sample_input_sequences)
            return sample_output, final_frame_state

    def loss(self,
             train_input_batch_rnn,
             train_frame_state,
             l2_regularization_strength=None,
             name='sample'):
        with tf.name_scope(name):
            # self.encoded_input_rnn = mu_law_encode(train_input_batch_rnn, self.q_levels)
            self.encoded_input_rnn = tf.cast(train_input_batch_rnn, tf.int32)
            encoded_rnn = self._one_hot(self.encoded_input_rnn)
            raw_output, final_frame_state = \
                self._create_network_SampleRnn(train_frame_state)
            with tf.name_scope('loss'):
                target_output_rnn = encoded_rnn[:, self.frame_size:, :]
                target_output_rnn = tf.reshape(target_output_rnn, [-1, self.q_levels])
                prediction = tf.reshape(raw_output, [-1, self.q_levels])

                loss = tf.nn.softmax_cross_entropy_with_logits(
                    logits=prediction,
                    labels=target_output_rnn)
                reduced_loss = tf.reduce_mean(loss)
                tf.summary.scalar('loss', reduced_loss)

                correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(target_output_rnn, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                if l2_regularization_strength is None or l2_regularization_strength == 0:
                    return reduced_loss, accuracy, final_frame_state
                else:
                    # L2 regularization for all trainable parameters
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                        for v in tf.trainable_variables()
                                        if not ('bias' in v.name)])

                    # Add the regularization term to the loss
                    total_loss = (reduced_loss +
                                  l2_regularization_strength * l2_loss)

                    tf.summary.scalar('l2_loss', l2_loss)
                    tf.summary.scalar('total_loss', total_loss)

                    return total_loss, accuracy, final_frame_state

    def create_gen_wav_para(self):
        with tf.name_scope('infe_para'):
            infe_para = dict()
            infe_para['infe_frame_inp'] = \
                tf.get_variable("infe_frame_inp",
                                [self.batch_size, self.frame_size, 1], dtype=tf.float32)
            infe_para['infe_frame_outp'] = \
                tf.get_variable("infe_frame_outp",
                                [self.batch_size, self.frame_size, self.dim], dtype=tf.float32)

            infe_para['infe_frame_outp_slices'] = \
                tf.get_variable("infe_frame_outp_slices",
                                [self.batch_size, 1, self.dim], dtype=tf.float32)
            infe_para['infe_sample_inp'] = \
                tf.get_variable("infe_sample_inp",
                                [self.batch_size, self.frame_size, 1], dtype=tf.int32)

            infe_para['infe_frame_state'] = self.cell.zero_state(self.batch_size, tf.float32)


            infe_para['infe_frame_outp'], \
            infe_para['infe_final_frame_state'] = \
                self._create_network_Frame(num_steps=1,
                                           frame_state=infe_para['infe_frame_state'],
                                           input_sequences=infe_para['infe_frame_inp'])

            sample_out = \
                self._create_network_Sample(frame_outputs=infe_para['infe_frame_outp_slices'],
                                            sample_input_sequences=infe_para['infe_sample_inp'])
            sample_out = \
                tf.reshape(sample_out, [-1, self.q_levels])
            infe_para['infe_sample_outp'] = tf.cast(
                tf.nn.softmax(tf.cast(sample_out, tf.float64)), tf.float32)

        return infe_para
