import tensorflow as tf
from ops import Conv2d, Residual_block, Conv3d, Conv3d_block, Downsample_block, Upsample_block
from tensorflow.contrib.layers import repeat

class ResNet(object):
	"""docstring for ResNet"""
	def __init__(self, arg):
		self.arg = arg

	def __call__(self, input):
		with tf.variable_scope('ResNet', reuse=tf.AUTO_REUSE) as scope:
			conv = []
			out = Conv2d('initial_transform', input, kernel_size=5, strides=2)
			conv.append(out)
		
			for x in range( 8 ):
				out = Residual_block('res_block'+str(x), conv[-1])
				conv.append(out)

		return(out)


class Regularization_block(object):
	"""docstring for Regularization"""
	def __init__(self, arg):
		self.arg = arg
	
	def __call__(self, cost_volume):
		
		with tf.variable_scope('Regularization', reuse=tf.AUTO_REUSE) as scope:
			# downsample
			conv = []
			out = Conv3d_block('downsample_0',cost_volume, 32) #0
			conv.append(out)
			out = Downsample_block('downsample_1',conv[-1], 64) #1
			conv.append(out)
			out = Downsample_block('downsample_2',conv[-1], 64) #2
			conv.append(out)
			out = Downsample_block('downsample_3',conv[-1], 64) #3
			conv.append(out)
			out = Downsample_block('downsample_4',conv[-1], 128) #4
			conv.append(out)

			# upsample
			out = Upsample_block('upsample_0', conv[-1], conv[3], 64)
			conv.append(out)
			out = Upsample_block('upsample_1', conv[-1], conv[2], 64)
			conv.append(out)
			out = Upsample_block('upsample_2', conv[-1], conv[1], 64)
			conv.append(out)
			out = Upsample_block('upsample_3', conv[-1], conv[0], 32)
			conv.append(out)


			# omit bn and relu!
			# output: batch x disp x height x width x 1
			out = tf.layers.conv3d_transpose(conv[-1], filters=1, kernel_size=3, strides=(2,2,2), 
				name='upsample_final', padding="same", reuse=tf.AUTO_REUSE)
		
		return(out)
