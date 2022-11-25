import argparse
from sklearn.model_selection import train_test_split
from utils import *
from layers import *

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import keras_tuner

from layers import NollaFraud

"""
   NollaFraud
   Source: https://github.com/C1rF/NollaFraud
"""

import os
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
os.environ['TF_NUM_INTEROP_THREADS'] = '16'
os.environ['TF_NUM_INTRAOP_THREADS'] = '16'

parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument('--data', type=str, default='amazon', help='The dataset name. [Amazon_demo, Yelp_demo, amazon,yelp]')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size 1024 for yelp, 256 for amazon.')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate. [0.1 for amazon and 0.001 for yelp]')
parser.add_argument('--lambda_1', type=float, default=1e-4, help='Weight decay (L2 loss weight).')
parser.add_argument('--embed_dim', type=int, default=64, help='Node embedding size at the first layer.')
parser.add_argument('--num_epochs', type=int, default=21, help='Number of epochs.')
parser.add_argument('--test_epochs', type=int, default=10, help='Epoch interval to run test set.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
# parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')

args = parser.parse_args()

print(f'run on {args.data}')

# load topology, feature, and label
homo, relation1, relation2, relation3, feat_data, labels = load_data(args.data)

feat_data = normalize(feat_data) 
adj_lists = [relation1, relation2, relation3]

np.random.seed(args.seed)
random.seed(args.seed)
tf.random.set_seed(args.seed)

# train_test split
if args.data == 'yelp':
	index = list(range(len(labels)))
	x_train, x_val, y_train, y_val = train_test_split(index, labels, stratify = labels, test_size = 0.80,
															random_state = 2, shuffle = True)
elif args.data == 'amazon':
	# 0-3304 are unlabeled nodes
	index = list(range(3305, len(labels)))
	x_train, x_val, y_train, y_val = train_test_split(index, labels[3305:], 
															test_size = 0.90, random_state = 2, shuffle = False)
else:
	exit("Dataset not supported")

num_1 = len(np.where(y_train == 1)[0])
num_2 = len(np.where(y_train == 0)[0])
p0 = (num_1 / (num_1 + num_2))
p1 = 1 - p0
prior = np.array([p1, p0])



def build():

	# TODO: Replace the dummy model with the actual gnn model

	# used for hyperparameter tuning, e.g. the best hiddenLayerDim that minimize loss
	# hiddenLayerDim = hp.Choice("hiddenLayerDim", [64, 128])

	# inputs = keras.Input(shape=(1,), name="node_index")
	# x1 = layers.Dense(hiddenLayerDim, activation="relu")(inputs)
	# x2 = layers.Dense(hiddenLayerDim, activation="relu")(x1)
	# outputs = layers.Dense(1, name="predictions")(x2)
	
	embed_dim = 96

	print(">>>>>>>>>>>>>>>>>\n\n\n\n\n")
	print("adj_ndarray", adjlist_to_ndarray(adj_lists[0]))

	adj_arrs = [adjlist_to_ndarray(adj_list) for adj_list in adj_lists]

	# print("test: ", adj_arrs[0][3305])

	model = NollaFraud(feat_data, adj_arrs, prior, embed_dim)
	# model.build((32, ))

	return model



def fit(model, x, y, validation_data):

	batch_size = 128
	# Prepare the training dataset.
	train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
	train_dataset = train_dataset.batch(batch_size)

	# Prepare the validation dataset.
	val_dataset = tf.data.Dataset.from_tensor_slices(validation_data)
	val_dataset = val_dataset.batch(batch_size)

	# Instantiate an optimizer.
	learningRate = 0.05
	optimizer = keras.optimizers.Adam(learning_rate=learningRate)
	
	
	# Instantiate a loss function.

	# TODO: Replace the dummy loss object with the actual loss function
	dummy_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	loss_fn = dummy_loss

	# Prepare the metrics.
	train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
	val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
	# val_auc_metric = keras.metrics.AUC(from_logits=True)
	# val_precision_metric = keras.metrics.Precision()
	# val_recall_metric = keras.metrics.Recall()
	# TODO: Find a method to calculate f1scores
	# val_f1_metric = tfa.metrics.F1Score(num_classes=2)

	val_epoch_loss_avg = keras.metrics.Mean()


	@tf.function
	def run_train_step(x_batch_train, y_batch_train):
		with tf.GradientTape() as tape:
			logits = model(x_batch_train, training=True)
			# print_with_color("SCORE:")
			# print_with_color(logits)
			print_with_color("PREDICTION:")
			# print_with_color(tf.math.sigmoid(logits).numpy().argmax(axis=1))
			print_with_color(tf.math.argmax(tf.math.sigmoid(logits), axis=1))
			print_with_color("LABELS:")
			print_with_color(tf.cast(y_batch_train, tf.int32))
			loss_value = loss_fn(y_batch_train, logits)
		grads = tape.gradient(loss_value, model.trainable_weights)
		optimizer.apply_gradients(zip(grads, model.trainable_weights))

		# Update training metric.
		train_acc_metric.update_state(y_batch_train, logits)

		return loss_value

	# print_with_color("TRAINABLE WEIGHTS")
	# print_with_color(model.trainable_weights)
	# print_with_color(model.summary())
	# Assign the model to the callbacks.

	@tf.function
	def run_val_step(x_batch_val, y_batch_val):
		val_logits = model(x_batch_val, training=False)
		loss_value = loss_fn(y_batch_val, val_logits)
		# Update val metrics
		val_acc_metric.update_state(y_batch_val, val_logits)
		# val_auc_metric.update_state(y_batch_val, val_logits)
		# print("val_logits: ", val_logits)
		# sigmoid_results = tf.keras.layers.Dense(1, activation="sigmoid")(val_logits)
		# print("val_logits sigmoid: ", sigmoid_results)
		# val_precision_metric.update_state(y_batch_val, sigmoid_results)
		# val_recall_metric.update_state(y_batch_val, sigmoid_results)
		# val_f1_metric.update_state(y_batch_val, val_logits)
		val_epoch_loss_avg.update_state(loss_value)

		return loss_value


	# Record the best validation loss value
	best_epoch_loss = float("inf")

	epochs = 61
	for epoch in range(epochs):
		# Iterate over the batches of the dataset.
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
			loss_value = run_train_step(x_batch_train, y_batch_train)
			# Log every 200 batches.
			# if step % 200 == 0:
			print(
				"Training loss (for one batch) at step %d: %.4f"
				% (step, float(loss_value))
			)
			# print("Seen so far: %d samples" % ((step + 1) * batch_size))
			# model.print_stats()
			# for layer in model.layers: print(layer.get_config(), layer.get_weights())
		
			# Display metrics at the end of each epoch.
			train_acc = train_acc_metric.result()
			print("Training acc over epoch: %.4f" % (float(train_acc),))

			# Reset training metrics at the end of each epoch
			train_acc_metric.reset_states()


		# Run a validation loop at the end of each epoch.
		for x_batch_val, y_batch_val in val_dataset:
			val_loss_value = run_val_step(x_batch_val, y_batch_val)
			# print("Validation loss: %.4f" % (float(val_loss_value),))

		val_acc = val_acc_metric.result()
		# val_auc = val_auc_metric.result()
		# val_precision = val_precision_metric.result()
		# val_recall = val_recall_metric.result()
		# val_f1 = val_f1_metric.result()
		val_loss = float(val_epoch_loss_avg.result().numpy())


		val_acc_metric.reset_states()
		# val_auc_metric.reset_states()
		# val_precision_metric.reset_states()
		# val_recall_metric.reset_states()
		# val_f1_metric.reset_states()
		val_epoch_loss_avg.reset_states()
		print("Validation acc: %.4f" % (float(val_acc),))
		# print("Validation auc: %.4f" % (float(val_auc),))
		# print("Validation precision: %.4f" % (float(val_precision),))
		# print("Validation recall: %.4f" % (float(val_recall),))
		# print("Validation f1: %.4f" % (float(val_f1),))
		print("Validation loss: %.4f" % (float(val_loss),))

		best_epoch_loss = float(min(best_epoch_loss, val_loss))

	return best_epoch_loss




model = build()
fit(model, x=x_train, y=y_train, validation_data=(x_val, y_val))