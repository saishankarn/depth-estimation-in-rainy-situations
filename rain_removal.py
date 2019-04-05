import os
import re
import random
import numpy as np
import tensorflow as tf
import cv2
from GuidedFilter import guided_filter


data_path      = '/home/saisai/rainy_depth/TrainData/input'    # the path of rainy images
label_path     = '/home/saisai/rainy_depth/TrainData/label'       # the path of ground truth
inference_path = '/home/saisai/rainy_depth/TestData'
log_directory  = '/home/saisai/rainy_depth/log_directory'

data_files  = os.listdir(data_path)
label_files = os.listdir(label_path) 
test_files  = os.listdir(inference_path)

iterations		   = int(2 * 1e4)
batch_size         = 20
patch_size         = 64
N_ROWS             = 256
N_COLS             = 512
NUM_INPUT_CHANNELS = 3


config                          = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess 		                    = tf.Session(config=config)

class Rain_Removal():

	"""
	Rain_Removal class has the Rain removal network and the loss calculation
	"""

	def __init__(self, data, label, is_training):

		self.data   = data
		self.label  = label
		self.base   = guided_filter(self.data, self.data, 15, 1, nhwc=True) # using guided filter for obtaining base layer
		self.detail = self.data - self.base   # detail layer

		#  layer 1
		with tf.variable_scope('conv_layer_1'):

			self.conv1     = tf.layers.conv2d(self.detail, 16, (3, 3), padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer(),  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-10), name = 'conv_1')
			self.bn_1       = tf.layers.batch_normalization(self.conv1, training = is_training, name = 'bn_1')
			self.conv1_out = tf.nn.relu(self.bn_1, name = 'relu_1')

		#  layers 2 to 25

		for i in range(12):

			with tf.variable_scope('conv_layer_%d'%(i * 2 + 2)):	

				self.conv2     = tf.layers.conv2d(self.conv1_out, 16, (3, 3), padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer(),  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-10), name=('conv_%d'%(i * 2 + 2)))
				self.bn_2      = tf.layers.batch_normalization(self.conv2, training = is_training, name = ('bn_%d'%(i * 2 + 2)))	
				self.conv2_out = tf.nn.relu(self.bn_2, name = ('relu_%d'%(i * 2 + 2)))


			with tf.variable_scope('conv_layer_%d'%(i * 2 + 3)): 

				self.conv2     = tf.layers.conv2d(self.conv2_out, 16, (3, 3), padding = 'same', kernel_initializer = tf.contrib.layers.xavier_initializer(),  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-10), name = ('conv_%d'%(i * 2 + 3)))
				self.bn_2      = tf.layers.batch_normalization(self.conv2, training = is_training, name = ('bn_%d'%(i * 2 + 3)))
				self.conv2_out = tf.nn.relu(self.bn_2, name = ('relu_%d'%(i * 2 + 3)))

			self.conv1_out = tf.add(self.conv1_out, self.conv2_out)   # shortcut

		# layer 26

		with tf.variable_scope('conv_layer_26'):

			self.conv26       = tf.layers.conv2d(self.conv1_out, 16, (3, 3), padding = 'same',  kernel_initializer = tf.contrib.layers.xavier_initializer(),  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-10), name = 'conv_26')
			self.neg_residual = tf.layers.batch_normalization(self.conv26, training = is_training, name = 'bn_26')

		self.output = tf.add(self.data, self.neg_residual)

		self.loss   = tf.reduce_mean(tf.square(self.label - self.output))

def read_data(data_path, label_path, data_files, label_files):

	Data  = np.zeros((batch_size, patch_size, patch_size, NUM_INPUT_CHANNELS)) 
	Label = np.zeros((batch_size, patch_size, patch_size, NUM_INPUT_CHANNELS)) 

	for i in range(batch_size):

		r_idx = random.randint(0,len(data_files)-1)
    
        data_image  = cv2.imread(data_path + '/' + data_files[r_idx])
        data_image  = data_image / 255.0

        label_image = cv2.imread(label_path + '/' + label_files[r_idx])
        label_image = label / 255.0

        x = random.randint(0, N_ROWS - patch_size)
        y = random.randint(0, N_COLS - patch_size)

        data_input  = data_image[x : x + patch_size, y : y + patch_size, :]
        label_input = label_image[x : x + patch_size, y : y + patch_size, :]

        data[i, :, :, :]  = rainy_input
        label[i, :, :, :] = label_input

    return data, label

# train function

def train():

	train_saver = tf.train.Saver()
	Total_loss  = 0
	lr = 0.0001
	for i in range(iterations):

		train_data, train_label = read_data(data_path, label_path, data_files, label_files, patch_size)
		_, iteration_loss = sess.run((train_, loss), feed_dict = {data : train_data, label : train_label, is_training : True, learning_rate : lr})

		Total_loss += iteration_loss

		if(i % 100 == 0 and i > 0):

			average_loss = Total_loss / (iterations + 1)

			content = 'iterations : ' + str(i + 1) + 'average loss : ' + str(average_loss) 

			test_image_loc = inference_path
			rand_int = random.randint(0, len(test_image_loc) - 1)

			test_image = cv2.imread(test_image_loc + '/' + test[rand_int])
			test_image = test_image / 255.0
			out_image  = sess.run((output), feed_dict = {data : test_image, is_training : False})
			out_path   = inference_path + '/' + str(i) + '.jpg'
			cv2.imwrite(out_path, out_image)

		if(i % 1000 == 0 and i > 0):

			train_saver.save(sess, log_directory + '/model', global_step = i)

	train_saver.save(sess, log_directory + '/model', global_step = iterations)

# main function

data          = tf.placeholder(tf.float32, shape = (batch_size, patch_size, patch_size, NUM_INPUT_CHANNELS), name = 'rainy_patches')  # data
label         = tf.placeholder(tf.float32, shape = (batch_size, patch_size, patch_size, NUM_INPUT_CHANNELS), name = 'label_patches')  # label
is_training   = tf.placeholder(tf.bool, name = 'is_training')
learning_rate = tf.placeholder(tf.float32, shape = [])

model  = Rain_Removal(data, label, is_training)

# loss calculation

loss   = model.loss
output = model.output

# optimisation

#optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.00001, decay = 0.9, momentum = 0.5, epsilon = 1e-10)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0.9, beta2 = 0.999)
train_= optimizer.minimize(loss)

init_op = tf.global_variables_initializer()		#Initializing the global variables
sess.run(init_op)

train()
#test()









