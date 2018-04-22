from model import ResNet, Regularization_block
from ops import CostVolume
import tensorflow as tf
import numpy as np

batch_size = 2
height = 224
width = 224
max_disp = 192
channels = 3
embedding_dimensions = 320
num_steps = 1000000
learning_rate = 1e-5
max_pts = 1000
log_dir = "logs/"

# filelist = get_filelist('train/out_colored_0')

# l = len(filelist)
# train_filelist = filelist[0:int(0.9*l)]
# test_filelist = filelist[int(0.9*l):]
# print filelist

X1 = tf.placeholder( tf.float32, shape=[batch_size, height, width, channels], name='X1' )
X2 = tf.placeholder( tf.float32, shape=[batch_size, height, width, channels], name='X2' )

resnet = ResNet({})
regularizer = Regularization_block({})
out1 = resnet(X1)
out2 = resnet(X2)

input1 = np.ones( [batch_size, height, width, channels], np.float32 )
input2= np.ones( [batch_size, height, width, channels], np.float32 )

C = CostVolume( (out1,out2), max_disp )

D = regularizer(C)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	writer = tf.summary.FileWriter(log_dir, sess.graph)
	sess.run(init)
	result = sess.run(D, feed_dict={X1:input1, X2:input2})
	print(result.shape)