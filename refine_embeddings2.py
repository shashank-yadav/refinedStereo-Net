import better_exceptions

import numpy as np
import tensorflow as tf

rnn = tf.contrib.rnn

class EmbeddingsRefiner(object):
    """ Class to refine embeddings before matching
    """

    def __init__(self, embedding_dimensions=128):
        self.num_refinement_steps = 5
        self.use_left_refinement = True
        self.use_right_refinement = True
        self.embedding_dimensions = embedding_dimensions

    def refine(self, left_hypercolumns, right_hypercolumns, L, refine_left=True, refine_right=True):
        """ refine hypercolumn embeddings of both left and right image
        :param left_hypercolumns - [L, batch_size, 128]
        :param right_hypercolumns - length L list of [batch_size, 128] tensors
        """
        with tf.variable_scope('refiner', reuse=tf.AUTO_REUSE) as scope:
            if refine_right:
                right_features_refined = self.fce_right(right_hypercolumns) # (L, batch_size, 128)
            else:
                right_features_refined = tf.convert_to_tensor(right_hypercolumns)

            if refine_left:
                left_features_refined = self.fce_left(left_hypercolumns, right_features_refined, L)
            else:
                left_features_refined = left_hypercolumns

        return left_features_refined, right_features_refined

    def fce_left(self, left_hypercolumns, right_hypercolumns, L):
        """ refine hypercolumn for left image
        f(x_i, S) = attLSTM(f'(x_i), g(S), K)
        hypercolumn refinement is done by running LSTM for fixed no. of steps (num_refinement_steps)
        attention over hypercolumns of points on epipolar line in right image used as
        context vector of LSTM
        :param left_hypercolumns - [L, batch_size, 128]
        :param right_hypercolumns - [L, batch_size, 128]
        """
        batch_size = tf.shape(left_hypercolumns)[1]

        cells = [None] * L
        prev_states = [None] * L
        outputs = [None] * L

        for i in range(L):
            cells[i] = rnn.BasicLSTMCell(self.embedding_dimensions)
            prev_states[i] = cells[i].zero_state(batch_size, tf.float32)   # state[0] is c, state[1] is h

        for step in xrange(self.num_refinement_steps):
            for i in range(L):
                outputs[i], state = cells[i](left_hypercolumns[i], prev_states[i])  # output: (batch_size, 128)

                h_k = tf.add(outputs[i], left_hypercolumns[i]) # (batch_size, 128)

                content_based_attention = tf.nn.softmax(tf.multiply(prev_states[i][1], right_hypercolumns))    # (L, batch_size, 128)
                r_k = tf.reduce_sum(tf.multiply(content_based_attention, right_hypercolumns), axis=0)      # (batch_size, 128)

                prev_states[i] = rnn.LSTMStateTuple(state[0], tf.add(h_k, r_k))

        left_features_refined = tf.convert_to_tensor(outputs, dtype=tf.float32)
        return left_features_refined


    def fce_right(self, right_hypercolumns):
        """ refine hypercolumn for right image
        g(x_i, S) = h_i(->) + h_i(<-) + g'(x_i)
        Set information is incorporated into embedding using bidirectional LSTM
        :param right_hypercolumns - length L list of [batch_size, 128] tensors (point features)
        """
        # dimension of fw and bw is half, so that output has size embedding_dimensions
        fw_cell = rnn.BasicLSTMCell(self.embedding_dimensions / 2)
        bw_cell = rnn.BasicLSTMCell(self.embedding_dimensions / 2)

        outputs, state_fw, state_bw = rnn.static_bidirectional_rnn(fw_cell, bw_cell, right_hypercolumns, dtype=tf.float32)

        # # Add lstm gates summary
        # fw_weights, fw_biases = fw_cell.variables
        # bw_weights, bw_biases = bw_cell.variables
        # print 'Forward weights'
        # print fw_weights.shape
        # fw_input, fw_new_input, fw_forget, fw_output = tf.split(fw_weights, 4, axis=1)
        # bw_input, bw_new_input, bw_forget, bw_output = tf.split(bw_weights, 4, axis=1)

        right_features_refined = tf.add(tf.stack(right_hypercolumns), tf.stack(outputs))
        return right_features_refined


def main():
    embedding_dimensions = 32
    refiner = EmbeddingsRefiner(embedding_dimensions=embedding_dimensions)

    sess = tf.InteractiveSession()
    L = 10
    batch_size = 4

    left_hypercolumns = tf.constant(np.random.randn(L, batch_size, embedding_dimensions), dtype=tf.float32)
    right_hypercolumns = [None] * L
    for i in xrange(L):
        right_hypercolumns[i] = tf.constant(np.random.randn(batch_size, embedding_dimensions), dtype=tf.float32)

    left_features_refined, right_features_refined = refiner.refine(left_hypercolumns, right_hypercolumns, L)
    sess.run(tf.global_variables_initializer())

    left_fs, right_fs = sess.run([left_features_refined, right_features_refined])

    print(len(left_fs))
    print(left_fs[0].shape)
    print(len(right_fs))
    print(right_fs[0].shape)

    sess.close()


if __name__ == '__main__':
    main()
