import better_exceptions

from model import ResNet, Regularization_block
from ops import CostVolume, Soft_argmin, get_filelist, get_batch, loss, loss_unregularized
import tensorflow as tf
import numpy as np
from refine_embeddings2 import EmbeddingsRefiner

batch_size = 2
height = 224
width = 224
max_disp = 192
channels = 3
embedding_dimensions = 32
num_steps = 1000000
learning_rate = 3e-6
max_pts = 1000
checkpoint_path = './checkpoints_right_refined/model'
log_dir = './logs_right_refined'

filelist = get_filelist('train/out_colored_0')

l = len(filelist)
train_filelist = filelist[0:int(0.9*l)]
test_filelist = filelist[int(0.9*l):]
# print filelist


X1 = tf.placeholder( tf.float32, shape=[batch_size, height, width, channels], name='X1' )
X2 = tf.placeholder( tf.float32, shape=[batch_size, height, width, channels], name='X2' )
Y = tf.placeholder( tf.float32, shape=[batch_size, height, width], name='Y' )

emb_net = ResNet({})
reg_net = Regularization_block({})
refiner = EmbeddingsRefiner(embedding_dimensions=embedding_dimensions)

left = emb_net(X1)
right = emb_net(X2)

print('Right: ')
print right.shape

h = height / 2
w = width / 2
left_emb_resized = tf.reshape( left, [batch_size * h, w, -1])		# (2*112, 112, 32)
right_emb_resized = tf.reshape( right, [batch_size * h, w, -1])		# (2*112, 112, 32)
print ('Resized: ')
print right_emb_resized.shape

# refine embeddings
right_emb_list = tf.unstack( right_emb_resized, axis=1 )
left_features_refined, right_features_refined = refiner.refine(left_emb_resized, right_emb_list, w, refine_left=False, refine_right=True)
print ('Refined: ')
print right_features_refined.shape

left_batched = tf.reshape( left_features_refined, [batch_size, h, w, -1] )
right_batched = tf.reshape( right_features_refined, [batch_size, h, w, -1] )
print ('Batched: ')
print left_batched.shape
print right_batched.shape

vol = CostVolume( (left_batched, right_batched) , max_disp/2 )
# vol = CostVolume( (left_batched, right_batched) , max_disp/2 )

print('Vol: ')
print vol.shape

prediction = reg_net(vol)

print prediction.shape

logits = Soft_argmin(prediction)
tf.summary.histogram('logits', logits)

loss_op = loss_unregularized(labels= Y, logits= logits, max_disp=max_disp)
loss_test_op = loss_unregularized(labels= Y, logits= logits, max_disp=max_disp)


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# all_vars = tf.all_variables()
trainable_vars = tf.trainable_variables()
# other_vars = list( set(all_vars) - set(trainable_vars) )
refiner_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'refiner')
load_vars = list(set(trainable_vars) - set(refiner_vars))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
	# Get the gradient pairs (Tensor, Variable)
	grads = optimizer.compute_gradients(loss_op)
	# Update the weights wrt to the gradient
	train_op = optimizer.apply_gradients(grads)
	# Save the grads with tf.summary.histogram
	# for index, grad in enumerate(grads):
	# 	tf.summary.scalar("{}-grad-norm".format(grads[index][1].name), tf.norm(grads[index][0]))
	# 	tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])
	# train_op = optimizer.minimize(loss_op)

tf.summary.scalar('loss', loss_op)
tf.summary.scalar('loss_test', loss_test_op, collections=['test'])

summary_op = tf.summary.merge_all()
summary_test_op = tf.summary.merge_all(key='test')


init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

	sess.run(init)

	# loader = tf.train.Saver(load_vars)
	loader = tf.train.Saver()
	loader.restore(sess, 'checkpoints_right_refined/model205500.ckpt')

	train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)

	saver = tf.train.Saver(max_to_keep=10)
	train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(log_dir + '/test')

	for i in xrange(205501, num_steps):
	
		left_input, right_input, expected_output = get_batch(i, batch_size, train_filelist)
		train_summary, train_loss, _ = sess.run( [summary_op, loss_op, train_op], feed_dict={X1: left_input, X2: right_input, Y:expected_output })

		if i%10 == 0:
			train_writer.add_summary(train_summary, i)

		if i%30 == 0:
			left_input_test, right_input_test, expected_output_test = get_batch(i, batch_size, test_filelist)
			test_summary, test_loss = sess.run( [summary_test_op, loss_test_op], feed_dict={X1: left_input_test, X2: right_input_test, Y:expected_output_test })
			test_writer.add_summary(test_summary, i)

		if i % 500 == 0:
			print('saving_'+str(i))
			saver.save(sess, checkpoint_path+str(i)+'.ckpt')

