import argparse
from sklearn.model_selection import train_test_split
from utils import *

import tensorflow as tf
from tensorflow import keras
from keras import layers


"""
   NollaFraud
   Source: https://github.com/C1rF/NollaFraud
"""

parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument('--data', type=str, default='amazon', help='The dataset name. [Amazon_demo, Yelp_demo, amazon,yelp]')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size 1024 for yelp, 256 for amazon.')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate. [0.1 for amazon and 0.001 for yelp]')
parser.add_argument('--lambda_1', type=float, default=1e-4, help='Weight decay (L2 loss weight).')
parser.add_argument('--embed_dim', type=int, default=64, help='Node embedding size at the first layer.')
parser.add_argument('--num_epochs', type=int, default=61, help='Number of epochs.')
parser.add_argument('--test_epochs', type=int, default=10, help='Epoch interval to run test set.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')

args = parser.parse_args()

print(f'run on {args.data}')

# load topology, feature, and label
homo, relation1, relation2, relation3, feat_data, labels = load_data(args.data)

np.random.seed(args.seed)
random.seed(args.seed)
tf.random.set_seed(args.seed)

# train_test split
if args.data == 'yelp':
	index = list(range(len(labels)))
	x_train, x_test, y_train, y_test = train_test_split(index, labels, stratify = labels, test_size = 0.80,
															random_state = 2, shuffle = True)
elif args.data == 'amazon':
	# 0-3304 are unlabeled nodes
	index = list(range(3305, len(labels)))
	x_train, x_test, y_train, y_test = train_test_split(index, labels[3305:], stratify = labels[3305:],
															test_size = 0.90, random_state = 2, shuffle = True)
else:
	exit("Dataset not supported")

# Construct tensorflow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(args.batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(args.batch_size)

# TODO: Replace the dummy model with the actual gnn model
dummy_model = keras.Sequential([
    layers.Dense(1, input_shape=[1]),
    layers.Dense(1, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model = dummy_model

# TODO: Replace the dummy loss object with the actual loss function
dummy_loss = keras.losses.BinaryCrossentropy(from_logits=True)
loss = dummy_loss

optimizer = keras.optimizers.Adam(learning_rate=args.lr)

train_loss_results = []
train_accuracy_results = []
test_loss_results = []
test_accuracy_results = []

# Train
for epoch in range(args.num_epochs):

	train_epoch_loss_avg = keras.metrics.Mean()
	train_epoch_accuracy = keras.metrics.BinaryAccuracy()

	for _, (batch_nodes, batch_label) in enumerate(train_dataset):
		# Optimize the model
		with tf.GradientTape() as tape:
			# Obtain loss and grads
			loss_value = loss(y_true=batch_label, y_pred=model(batch_nodes))
			grads = tape.gradient(loss_value, model.trainable_variables)
			# Optimize
			optimizer.apply_gradients(zip(grads, model.variables))
			# Track status
			train_epoch_loss_avg.update_state(loss_value)  # Add current batch loss
			train_epoch_accuracy.update_state(batch_label, model(batch_nodes))

	# End epoch
	train_loss_results.append(train_epoch_loss_avg.result())
	train_accuracy_results.append(train_epoch_accuracy.result())

	if epoch % 10 == 0:
		print("Train =====> Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                train_epoch_loss_avg.result(),
                                                                train_epoch_accuracy.result()))
# Test
for epoch in range(args.test_epochs):
	test_epoch_loss_avg = keras.metrics.Mean()
	test_epoch_accuracy = keras.metrics.BinaryAccuracy()

	for _, (batch_nodes, batch_label) in enumerate(test_dataset):
		# Optimize the model
		with tf.GradientTape() as tape:
			# Obtain loss
			loss_value = loss(y_true=batch_label, y_pred=model(batch_nodes))
			# Track status
			test_epoch_loss_avg.update_state(loss_value)  # Add current batch loss
			test_epoch_accuracy.update_state(batch_label, model(batch_nodes))

	# End epoch
	test_loss_results.append(test_epoch_loss_avg.result())
	test_accuracy_results.append(test_epoch_accuracy.result())

	if epoch == args.test_epochs - 1:
		print("Test =====> Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                test_epoch_loss_avg.result(),
                                                                test_epoch_accuracy.result()))

# TODO: Evaluate performance