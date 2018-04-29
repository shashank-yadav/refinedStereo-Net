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

    def refine(self, left_hypercolumn, right_hypercolumns, refine_left=True, refine_right=True):
        """ refine hypercolumn embeddings of both left and right image
        :param left_hypercolumns - [batch_size, 128]
        :param right_hypercolumns - length L list of [batch_size, 128] tensors
        """
        with tf.variable_scope('refiner', reuse=tf.AUTO_REUSE) as scope:
            if refine_right:
                right_features_refined = self.fce_right(right_hypercolumns) # (L, batch_size, 128)
            else:
                right_features_refined = right_hypercolumns

            if refine_left:
                left_feature_refined = self.fce_left(left_hypercolumn, right_features_refined)
            else:
                left_feature_refined = left_hypercolumn

        return left_feature_refined, right_features_refined

    def fce_left(self, left_hypercolumn, right_hypercolumns):
        """ refine hypercolumn for left image
        f(x_i, S) = attLSTM(f'(x_i), g(S), K)
        hypercolumn refinement is done by running LSTM for fixed no. of steps (num_refinement_steps)
        attention over hypercolumns of points on epipolar line in right image used as
        context vector of LSTM
        :param left_hypercolumn - [batch_size, 128] tensor (point feature)
        :param right_hypercolumns - [L, batch_size, 128]
        """
        # assert(tf.shape(left_hypercolumn) == tf.shape(right_hypercolumns[0]))
        batch_size = tf.shape(left_hypercolumn)[0]

        cell = rnn.BasicLSTMCell(self.embedding_dimensions)
        prev_state = cell.zero_state(batch_size, tf.float32)   # state[0] is c, state[1] is h

        for step in xrange(self.num_refinement_steps):
            output, state = cell(left_hypercolumn, prev_state)  # output: (batch_size, 128)

            cell_weights, cell_biases = cell.variables
            tf.summary.histogram("left_refiner/weights_{}".format(step), cell_weights)
            tf.summary.histogram("left_refiner/biases_{}".format(step), cell_biases)          

            h_k = tf.add(output, left_hypercolumn) # (batch_size, 128)

            content_based_attention = tf.nn.softmax(tf.multiply(prev_state[1], right_hypercolumns))    # (L, batch_size, 128)
            r_k = tf.reduce_sum(tf.multiply(content_based_attention, right_hypercolumns), axis=0)      # (batch_size, 128)

            prev_state = rnn.LSTMStateTuple(state[0], tf.add(h_k, r_k))

        return output


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

        fw_weights, fw_biases = fw_cell.variables
        bw_weights, bw_biases = bw_cell.variables

        tf.summary.histogram("right_refiner/fw/weights", fw_weights)
        tf.summary.histogram("right_refiner/fw/biases", fw_biases)
        tf.summary.histogram("right_refiner/bw/weights", bw_weights)
        tf.summary.histogram("right_refiner/bw/biases", bw_biases)

        right_hypercolumns_tensor = tf.stack(right_hypercolumns)
        outputs_tensor = tf.stack(outputs)

        tf.summary.histogram("right_hypercolumns", right_hypercolumns_tensor)
        tf.summary.histogram("outputs", outputs_tensor)

        right_features_refined = tf.add(right_hypercolumns_tensor, outputs_tensor)
        return right_features_refined


def main():
    embedding_dimensions = 32
    refiner = EmbeddingsRefiner(embedding_dimensions=embedding_dimensions)

    sess = tf.InteractiveSession()
    L = 10
    batch_size = 4

    left_hypercolumn = tf.constant(np.random.randn(batch_size, embedding_dimensions), dtype=tf.float32)
    right_hypercolumns = [None] * L
    for i in xrange(L):
        right_hypercolumns[i] = tf.constant(np.random.randn(batch_size, embedding_dimensions), dtype=tf.float32)

    left_feature_refined, right_features_refined = refiner.refine(left_hypercolumn, right_hypercolumns)
    sess.run(tf.global_variables_initializer())

    left_f, right_fs = sess.run([left_feature_refined, right_features_refined])

    print(left_f.shape)
    print(len(right_fs))
    print(right_fs[0].shape)

    sess.close()


if __name__ == '__main__':
    main()
