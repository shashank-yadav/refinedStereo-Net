from tensorflow.contrib.layers import repeat, conv2d
import tensorflow as tf
import numpy as np
import os
import cv2
import random


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


def Residual_block(scope_name, input    ):

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


def Soft_argmin(cost):
	# def inner(cost):
	# there is only one feature left per dimension
	# input: batch x disp x height x width x 1
	# output: batch x disp x height x width
	shape = cost.get_shape().as_list()
	max_disp = shape[1]
	cost = tf.squeeze(cost, axis=-1)

	# calc propability for the disparities
	prob = -cost
	norm_prob = tf.nn.softmax(prob, dim=1)

	# calc disparity
	disp_vec = tf.range(max_disp, dtype=tf.float32)
	disp_map = tf.reshape(disp_vec, (1, max_disp, 1, 1))
	# output = tf.layers.conv2d(
	# 	norm_prob, disp_map, strides=(1, 1),
	# 	data_format='channels_first', padding='valid',
	# )

	output = tf.reduce_sum( disp_map*norm_prob, axis=1 )
	return(output)

	# return Lambda(inner, output_shape=(height, width))


def loss(logits, labels, max_disp):
	# both logits and labels have dimensions of channels x height x width
	mask = tf.cast( tf.logical_and(labels > 0, labels < max_disp) , dtype=tf.bool)
	diff = tf.abs( labels - logits)
	diff = tf.where(mask, diff, tf.zeros_like(labels))
	loss_mean = tf.reduce_sum(diff) / tf.reduce_sum( tf.cast(mask, tf.float32) )
	return(loss_mean)


def get_filelist(folder):
	filelist = os.listdir(folder)
	# random.shuffle(filelist)
	return(filelist)


def get_batch(id, batch_size, filelist):
	left_dir='train/out_colored_0'
	right_dir='train/out_colored_1'
	disp_dir='train/out_disp_occ'

	# filelist = get_filelist(left_dir)

	ids = [ (id*batch_size + x)%len(filelist) for x in range(batch_size) ]
	
	left_batch = [None]*batch_size
	right_batch = [None]*batch_size
	disp_batch = [None]*batch_size
	map_batch = [None]*batch_size

	for x in xrange(0,batch_size):
		left_batch[x] = 1.0*cv2.imread( left_dir+'/'+ filelist[ids[x]] , -1)[np.newaxis, :, : ,:]
		right_batch[x] = 1.0*cv2.imread( right_dir+'/'+ filelist[ids[x]], -1)[np.newaxis, :, : ,:]
		
		y = cv2.imread( disp_dir+'/'+ filelist[ids[x]] , -1)[np.newaxis, :, :]

		disp_batch[x] = (1.0*y.copy())/256.0
		
		y[y>0] = 1
		# map_batch
		map_batch[x] = y
		# map_batch[x] = 1.0*y
		# map_batch[x] = y>0
	return( np.concatenate(left_batch,0), np.concatenate(right_batch,0), np.concatenate(disp_batch,0) )