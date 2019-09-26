#! /usr/bin/python3
###########################################################################
# implement 3D cnn for EEG decode
###########################################################################
from cnn_class import cnn
import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import time
import sys
import math
import scipy.io as sio

np.random.seed(33)

###########################################################################
# set model parameters
###########################################################################
# kernel parameter
kernel_depth_1st	= 128
kernel_height_1st	= 3
kernel_width_1st 	= 3

kernel_depth_2nd	= 3
kernel_height_2nd	= 3
kernel_width_2nd 	= 3

kernel_depth_3rd	= 3
kernel_height_3rd	= 3
kernel_width_3rd 	= 3

kernel_stride		= 1

conv_channel_num	= 4

# pooling parameter
pooling_depth_1st 	= "None"
pooling_height_1st 	= "None"
pooling_width_1st 	= "None"

pooling_depth_2nd 	= "None"
pooling_height_2nd 	= "None"
pooling_width_2nd	= "None"

pooling_depth_3rd 	= "None"
pooling_height_3rd 	= "None"
pooling_width_3rd 	= "None"

pooling_stride		= "None"

# full connected parameter
fc_size 			= 1024

###########################################################################
# set training parameters
###########################################################################
# set learning rate
learning_rate = 1e-4

# set maximum traing epochs
training_epochs = 80

# set batch size
batch_size = 200

# set dropout probability
dropout_prob = 0.5

# set whether use L2 regularization
enable_penalty = True

# set L2 penalty
lambda_loss_amount = 0.5

###########################################################################
# set dataset parameters
###########################################################################
# input channel
input_channel_num = 1

# window size
window_size = 128

# input depth
input_depth = window_size

# input height 
input_height = 9

# input width
input_width = 9

# prediction class
num_labels = 4

# train test split
train_test_split = 0.9
# fold
fold =10 

cnn_suffix      =".mat_win_128_cnn_dataset.pkl"
label_suffix    =".mat_win_128_labels.pkl"

subject = sys.argv[1]
arousal_or_valence = sys.argv[2]
# dataset directory
dataset_dir = "/home/yyl/ijcnn/deap_shuffled_data/yes_"+arousal_or_valence+"/"

# load dataset and label
with open(dataset_dir+subject+cnn_suffix, "rb") as fp:
  	datasets = pickle.load(fp)
  	print(datasets.shape)
with open(dataset_dir+subject+label_suffix, "rb") as fp:
  	labels = pickle.load(fp)
lables_backup = labels
num_labels = len(set(labels))
# reshape dataset
datasets = datasets.reshape(len(datasets), window_size, 9,9, 1)

# shuffle
index = np.array(range(0, len(datasets)))
np.random.shuffle(index)
datasets=datasets[index]
labels=labels[index]

# set label to one hot
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)

###########################################################################
# for output record
###########################################################################

# shape of cnn layer
conv_1_shape = str(kernel_depth_1st)+"*"+str(kernel_height_1st)+"*"+str(kernel_width_1st)+"*"+str(kernel_stride)+"*"+str(conv_channel_num)
pool_1_shape = str(pooling_height_1st)+"*"+str(pooling_width_1st)+"*"+str(pooling_stride)+"*"+str(conv_channel_num)

conv_2_shape = str(kernel_depth_2nd)+"*"+str(kernel_height_2nd)+"*"+str(kernel_width_2nd)+"*"+str(kernel_stride)+"*"+str(conv_channel_num*2)
pool_2_shape = str(pooling_depth_2nd)+"*"+str(pooling_height_2nd)+"*"+str(pooling_width_2nd)+"*"+str(pooling_stride)+"*"+str(conv_channel_num*2)

conv_3_shape = str(kernel_depth_3rd)+"*"+str(kernel_height_3rd)+"*"+str(kernel_width_3rd)+"*"+str(kernel_stride)+"*"+str(conv_channel_num*4)
pool_3_shape = str(pooling_depth_3rd)+"*"+str(pooling_height_3rd)+"*"+str(pooling_width_3rd)+"*"+str(pooling_stride)+"*"+str(conv_channel_num*4)

# regularization method
if enable_penalty:
	regularization_method = 'dropout+l2'
else:
	regularization_method = 'dropout'

# result output
result_dir = "/home/hy01/emotion_first/result/3D_CNN/"+arousal_or_valence+"/"
###########################################################################
# build network
###########################################################################

# instance cnn class
cnn_3d = cnn()

# input placeholder
X = tf.placeholder(tf.float32, shape=[None, input_depth, input_height, input_width, input_channel_num], name = 'X')
Y = tf.placeholder(tf.float32, shape=[None, num_labels], name = 'Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# first CNN layer
conv_1 = cnn_3d.apply_conv3d(X, kernel_depth_1st, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride,'1')
# pool_1 = cnn_3d.apply_max_pooling3d(conv_1, pooling_depth, pooling_height, pooling_width, pooling_stride)
print("conv1 shape:",conv_1.shape)
# second CNN layer
conv_2 = cnn_3d.apply_conv3d(conv_1, kernel_depth_2nd, kernel_height_2nd, kernel_width_2nd, conv_channel_num, conv_channel_num*2, kernel_stride,'2')
print("conv2 shape:",conv_2.shape)
# pool_2 = cnn_3d.apply_max_pooling3d(conv_2, pooling_depth, pooling_height, pooling_width, pooling_stride)
# third CNN layer
conv_3 = cnn_3d.apply_conv3d(conv_2, kernel_depth_3rd, kernel_height_3rd, kernel_width_3rd, conv_channel_num*2, conv_channel_num*4, kernel_stride,'3')
print("conv3 shape:",conv_3.shape)
# pool_3 = cnn_3d.apply_max_pooling3d(conv_3, pooling_depth, pooling_height, pooling_width, pooling_stride)
# fourth CNN layer
'''
conv_3 = cnn_3d.apply_conv3d(conv_3, kernel_depth_4rd, kernel_height_4rd, kernel_width_4rd, conv_channel_num*4, conv_channel_num*8, kernel_stride)
print("conv4 shape:",conv_3.shape)
'''
# pool_3 = cnn_3d.apply_max_pooling3d(conv_3, pooling_depth, pooling_height, pooling_width, pooling_stride)

# flattern the last layer of cnn
shape = conv_3.get_shape().as_list()

conv_3_flat = tf.reshape(conv_3, [-1, shape[1]*shape[2]*shape[3]*shape[4]])
print("shape:",shape)
# fully connected layer
fc = cnn_3d.apply_fully_connect(conv_3_flat, shape[1]*shape[2]*shape[3]*shape[4], fc_size,'4')

## Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing
fc_drop = tf.nn.dropout(fc, keep_prob)

# l2 regularization
l2 = lambda_loss_amount * sum(
	tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)


# readout layer
y_ = cnn_3d.apply_readout(fc_drop, fc_size,num_labels,'5')

# probability prediction 
y_posi = tf.nn.softmax(y_, name = "y_posi")

# class prediction 
y_pred = tf.argmax(y_, 1, name = "y_pred")



if enable_penalty:
	# cross entropy cost function
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y)) + l2
else:
	# cross entropy cost function
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name = 'loss')

# set training SGD optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# get correctly predicted object
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))

# calculate prediction accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

###########################################################################
# train test and save result
###########################################################################
# run with gpu memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

for curr_fold in range(fold):
	fold_size = datasets.shape[0]//fold
	indexes_list = [i for i in range(len(datasets))]
	indexes = np.array(indexes_list)
	split_list = [i for i in range(curr_fold*fold_size,(curr_fold+1)*fold_size)]
	split = np.array(split_list)
	test_x = datasets[split] 
	test_y = labels[split]

	split = np.array(list(set(indexes_list)^set(split_list)))
	train_x = datasets[split] 
	train_y = labels[split]
	print("training examples:", train_y.shape[0])
	print("test examples	:",test_y.shape[0])
	# set train batch number per epoch
	batch_num_per_epoch = math.floor(train_x.shape[0]/batch_size)+ 1

	# set test batch number per epoch
	accuracy_batch_size = batch_size
	train_accuracy_batch_num = batch_num_per_epoch
	test_accuracy_batch_num = math.floor(test_x.shape[0]/batch_size)+ 1

	# print label
	one_hot_labels = np.array(list(pd.get_dummies(lables_backup)))
	print(one_hot_labels)


	with tf.Session(config=config) as session:
		session.run(tf.global_variables_initializer())
		train_accuracy_save = np.zeros(shape=[0], dtype=float)
		test_accuracy_save 	= np.zeros(shape=[0], dtype=float)
		test_loss_save 		= np.zeros(shape=[0], dtype=float)
		train_loss_save 	= np.zeros(shape=[0], dtype=float)
		count=-1
		for epoch in range(training_epochs):
			print("learning_rate:",learning_rate)
			cost_history = np.zeros(shape=[0], dtype=float)
			# training process
			for b in range(batch_num_per_epoch):
				start = b* batch_size
				if (b+1)*batch_size>train_y.shape[0]:
					offset = train_y.shape[0] % batch_size
				else:
					offset = batch_size
				batch_x = train_x[start:(offset + start), :, :, :, :]
				batch_y = train_y[start:(offset + start), :]
	#			print("traing:",start,"->",(start+offset))
				'''
				offset = (b * batch_size) % (train_y.shape[0] - batch_size) 
				batch_x = train_x[offset:(offset + batch_size), :, :, :, :]
				batch_y = train_y[offset:(offset + batch_size), :]
				'''
				_, c = session.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1-dropout_prob})
				cost_history = np.append(cost_history, c)
			# calculate train and test accuracy after each training epoch
			if(epoch%1 == 0):
				train_accuracy 	= np.zeros(shape=[0], dtype=float)
				test_accuracy	= np.zeros(shape=[0], dtype=float)
				test_loss 		= np.zeros(shape=[0], dtype=float)
				train_loss 		= np.zeros(shape=[0], dtype=float)
				for i in range(train_accuracy_batch_num):
					start = i* batch_size
					if (i+1)*batch_size>train_y.shape[0]:
						offset = train_y.shape[0] % batch_size
					else:
						offset = accuracy_batch_size
	#				offset = (i * accuracy_batch_size) % (train_y.shape[0] - accuracy_batch_size) 
					train_batch_x = train_x[start:(offset + start), :, :, :, :]
					train_batch_y = train_y[start:(offset + start), :]
	#				print("testing:",start,"->",start+offset)
					train_a, train_c = session.run([accuracy, cost], feed_dict={X: train_batch_x, Y: train_batch_y, keep_prob: 1.0})
					
					train_loss = np.append(train_loss, train_c)
					train_accuracy = np.append(train_accuracy, train_a)
				print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch+1, " Training Cost: ", np.mean(train_loss), "Training Accuracy: ", np.mean(train_accuracy))
				train_accuracy_save = np.append(train_accuracy_save, np.mean(train_accuracy))
				train_loss_save = np.append(train_loss_save, np.mean(train_loss))
				# calculate test accuracy after each training epoch
				for j in range(test_accuracy_batch_num):
					start = j * batch_size
					if (j+1)*batch_size>test_y.shape[0]:
						offset = test_y.shape[0] % batch_size
					else:
						offset = batch_size
	#				offset = (j * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size) 
					test_batch_x = test_x[start:(start + offset), :, :, :, :]
					test_batch_y = test_y[start:(start + offset), :]
	#				print("testing:",start,"-",start+offset)
					test_a, test_c = session.run([accuracy, cost], feed_dict={X: test_batch_x, Y: test_batch_y,keep_prob: 1.0})
					test_accuracy = np.append(test_accuracy, test_a)
					test_loss = np.append(test_loss, test_c)

				print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch+1, " Test Cost: ", np.mean(test_loss), "Test Accuracy: ", np.mean(test_accuracy),"\n")
				test_accuracy_save 	= np.append(test_accuracy_save, np.mean(test_accuracy))
				test_loss_save 		= np.append(test_loss_save, np.mean(test_loss))
			# reshuffle
			index_2 = np.array(range(0, len(train_y)))
			np.random.shuffle(index_2)
			train_x=train_x[index_2]
			train_y=train_y[index_2]

			index_3 = np.array(range(0, len(test_y)))
			np.random.shuffle(index_3)
			test_x=test_x[index_3]
			test_y=test_y[index_3]

			# learning_rate decay
			if(np.mean(train_accuracy)<0.9):
				learning_rate=1e-4
			elif(0.9<np.mean(train_accuracy)<0.95):
				learning_rate=5e-5
			elif(0.99<np.mean(train_accuracy)):
				learning_rate=5e-6

	###########################################################################
	# save result and model after training 
	###########################################################################
		test_accuracy 	= np.zeros(shape=[0], dtype=float)
		test_loss 		= np.zeros(shape=[0], dtype=float)
		test_pred		= np.zeros(shape=[0], dtype=float)
		test_true		= np.zeros(shape=[0,num_labels], dtype=float)
		test_posi		= np.zeros(shape=[0,num_labels], dtype=float)
		for k in range(test_accuracy_batch_num):

			start = k * batch_size
			if (k+1)*batch_size>test_y.shape[0]:
				offset = test_y.shape[0] % batch_size
			else:
				offset = batch_size
	#		offset = (k * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size) 
			test_batch_x = test_x[start:(offset + start), :, :, :, :]
			test_batch_y = test_y[start:(offset + start), :]
	#		print("final testing:",start,"->",start+offset)
			test_a, test_c, test_p, test_r = session.run([accuracy, cost, y_pred, y_posi], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0})
			test_t = test_batch_y

			test_accuracy 	= np.append(test_accuracy, test_a)
			test_loss 		= np.append(test_loss, test_c)
			test_pred 		= np.append(test_pred, test_p)
			test_true 		= np.vstack([test_true, test_t])
			test_posi		= np.vstack([test_posi, test_r])
		test_pred_1_hot = np.asarray(pd.get_dummies(test_pred), dtype = np.int8)
		test_true_list	= tf.argmax(test_true, 1).eval()
	#	print("test_accuracy:",test_accuracy)
		# recall
		test_recall = recall_score(test_true, test_pred_1_hot, average=None)
		# precision
		test_precision = precision_score(test_true, test_pred_1_hot, average=None)
		# f1 score
		test_f1 = f1_score(test_true, test_pred_1_hot, average=None)
		# confusion matrix
		# confusion_matrix = confusion_matrix(test_true_list, test_pred)
		# print("confusion_matrix:",confusion_matrix)

		print("("+time.asctime(time.localtime(time.time()))+") Final Test Cost: ", np.mean(test_loss), "Final Test Accuracy: ", np.mean(test_accuracy))
		# save result
		result 	= pd.DataFrame({'epoch':range(1,epoch+2), "train_accuracy":train_accuracy_save, "test_accuracy":test_accuracy_save,"train_loss":train_loss_save,"test_loss":test_loss_save})
		ins 	= pd.DataFrame({'conv_1':conv_1_shape, 'pool_1':pool_1_shape, 'conv_2':conv_2_shape, 'pool_2':pool_2_shape, 'conv_3':conv_3_shape, 'pool_3':pool_3_shape,'fc':fc_size,'accuracy':np.mean(test_accuracy), 'keep_prob': 1-dropout_prob, 'sliding_window':window_size, "epoch":epoch+1, "learning_rate":learning_rate, "regularization":regularization_method}, index=[0])
		summary = pd.DataFrame({'class':one_hot_labels, 'recall':test_recall, 'precision':test_precision, 'f1_score':test_f1})

		writer = pd.ExcelWriter(result_dir+subject+"_"+str(curr_fold)+".xlsx")
		# save model implementation paralmeters
		ins.to_excel(writer, 'condition', index=False)
		# save train/test accuracy and loss for each epoch
		result.to_excel(writer, 'result', index=False)
		# save recall/precision/f1 for each class
		summary.to_excel(writer, 'summary', index=False)
		# fpr, tpr, auc
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		i = 0
		writer.save()
		
		# save model
		parameter_count=0

		model_dict= {}
		for variable in tf.trainable_variables():
			print(variable.name,"-->",variable.get_shape())
			count = 1
			for dim in variable.get_shape():
				count = count*dim
			parameter_count = parameter_count+count
			model_dict[variable.name]=session.run(variable)
		sio.savemat("./result/3D_model_"+str(parameter_count)+".mat",model_dict)
		print("----------------------------------------------------------------")
		print("------------------total parameters",parameter_count,"-----------------------")
		print("----------------------------------------------------------------")
		session.close()
		'''
		saver = tf.train.Saver()
		saver.save(session, result_dir+arousal_or_valence+"_checkpoint/model")
		'''
print("**********("+time.asctime(time.localtime(time.time()))+") Train and Test NN End **********\n")