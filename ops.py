import tensorflow as tf
from tensorflow.contrib.layers import repeat, conv2d


def Conv2d(layer_name, input, filters=32, kernel_size=3, strides=1, trainable=True):
	out = tf.layers.conv2d( input , filters=filters, kernel_size=[kernel_size,kernel_size], strides=strides, 
		name=layer_name, padding="same", reuse=tf.AUTO_REUSE, trainable=trainable )
	out = tf.layers.batch_normalization(out, name=layer_name+'_batchnorm')
	outf = tf.nn.relu(out, name=layer_name+'_relu')
	return( outf )


def Conv3d(layer_name, input, filters=32, kernel_size=3, strides=1, trainable=True):
	out = tf.layers.conv3d( input , filters=filters, kernel_size=kernel_size, strides=strides, 
		name=layer_name, padding="same", reuse=tf.AUTO_REUSE, trainable=trainable )
	out = tf.layers.batch_normalization(out, name=layer_name+'_batchnorm')
	outf = tf.nn.relu(out, name=layer_name+'_relu')
	return( outf )


def Conv3d_block(scope_name, input, filters):
	with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
		conv = Conv3d('conv1', input, filters)
		conv = Conv3d('conv2', conv, filters)
	return( conv )


def Downsample_block(scope_name, input, filters):
	with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
		conv = Conv3d( 'downsample', input, filters, strides=2)
		conv = Conv3d_block( 'Downsample_conv' ,conv, filters)
	return( conv )


def Upsample_block(scope_name, input, residual, filters=32):
	
	with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
		out = tf.layers.conv3d_transpose( input , filters=filters, kernel_size=3, strides=2, 
			name='deconv1', padding="same", reuse=tf.AUTO_REUSE )
		out = tf.layers.batch_normalization(out, name='batchnorm')
		outf = tf.nn.relu(out, name='relu')
	return( outf+residual )


def Residual_block(scope_name, input	):

	with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
		out1 = Conv2d('conv1', input) 
		out2 = Conv2d('conv2', out1) 

	return(input+out2)



def CostVolume(inputs, max_disp):
	
	if max_disp % 2 != 0:
		raise ValueError("max_disp must be divisible by 2.")
	
	disp = int(max_disp / 2)

	def inner(inputs, disp):
		left_tensor, right_tensor = inputs
		# extend right features by `disp` columns on the left side
		right_tensor = tf.pad( tensor=right_tensor, paddings=((0,0),(0,0),(disp,0),(0,0)), mode="CONSTANT" )
		shape = left_tensor.get_shape().as_list()

		disparity_costs = []
		for d in range(disp):
			# get slice with `d` 0 on the left -> shifting feature over each other
			left_tensor_slice = left_tensor
			right_tensor_slice = right_tensor[:, :, d: d + shape[2], :]
			cost = tf.concat(
				[left_tensor_slice, right_tensor_slice],
				axis=3,
			)
			disparity_costs.append(cost)
		# stack everything into a 4D tensor
		return tf.stack(disparity_costs, axis=1)

	return inner(inputs, max_disp)
