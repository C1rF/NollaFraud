import argparse
from sklearn.model_selection import train_test_split
from utils import *
from layers import *

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

from layers import NollaFraud
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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


parser.add_argument('--data', type=str, default='amazon',
                    help='The dataset name. [Amazon_demo, Yelp_demo, amazon,yelp]')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size 1024 for yelp, 256 for amazon.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate. [0.1 for amazon and 0.001 for yelp]')
parser.add_argument('--lambda_1', type=float, default=1e-4,
                    help='Weight decay (L2 loss weight).')
parser.add_argument('--embed_dim', type=int, default=64,
                    help='Node embedding size at the first layer.')
parser.add_argument('--num_epochs', type=int,
                    default=21, help='Number of epochs.')
parser.add_argument('--test_epochs', type=int, default=10,
                    help='Epoch interval to run test set.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')


args = parser.parse_args()

print(f'run on {args.data}')


homo, relation1, relation2, relation3, feat_data, labels = load_data(args.data)

feat_data = normalize(feat_data)
adj_lists = [relation1, relation2, relation3]
adj_lists = [adjlist_to_ndarray(adj_list) for adj_list in adj_lists]

np.random.seed(args.seed)
random.seed(args.seed)
tf.random.set_seed(args.seed)


if args.data == 'yelp':
    index = list(range(len(labels)))
    x_train, x_val, y_train, y_val = train_test_split(index, labels, stratify=labels, test_size=0.80,
                                                      random_state=2, shuffle=True)
elif args.data == 'amazon':

    index = list(range(3305, len(labels)))
    x_train, x_val, y_train, y_val = train_test_split(index, labels[3305:],
                                                      test_size=0.90, random_state=2, shuffle=False)
else:
    exit("Dataset not supported")

num_1 = len(np.where(y_train == 1)[0])
num_2 = len(np.where(y_train == 0)[0])
p0 = (num_1 / (num_1 + num_2))
p1 = 1 - p0
prior = np.array([p1, p0])


def build():

    embed_dim = 96

    adj_arrs = [adjlist_to_ndarray(adj_list) for adj_list in adj_lists]

    model = NollaFraud(feat_data, adj_arrs, prior, embed_dim)

    return model


def fit(model, x, y, validation_data):

    batch_size = 128

    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = train_dataset.batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(validation_data)
    val_dataset = val_dataset.batch(batch_size)

    learningRate = 0.05
    optimizer = keras.optimizers.Adam(learning_rate=learningRate)

    dummy_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn = dummy_loss

    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    val_epoch_loss_avg = keras.metrics.Mean()

    @tf.function
    def run_train_step(x_batch_train, y_batch_train):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)

            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_acc_metric.update_state(y_batch_train, logits)

        return loss_value

    @tf.function
    def run_val_step(x_batch_val, y_batch_val):
        val_logits = model(x_batch_val, training=False)
        loss_value = loss_fn(y_batch_val, val_logits)

        val_acc_metric.update_state(y_batch_val, val_logits)

        val_epoch_loss_avg.update_state(loss_value)

        return loss_value

    best_epoch_loss = float("inf")

    epochs = 61
    for epoch in range(epochs):

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = run_train_step(x_batch_train, y_batch_train)

            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )

            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            train_acc_metric.reset_states()

        for x_batch_val, y_batch_val in val_dataset:
            val_loss_value = run_val_step(x_batch_val, y_batch_val)

        val_acc = val_acc_metric.result()

        val_loss = float(val_epoch_loss_avg.result().numpy())

        val_acc_metric.reset_states()

        val_epoch_loss_avg.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))

        print("Validation loss: %.4f" % (float(val_loss),))

        best_epoch_loss = float(min(best_epoch_loss, val_loss))

    return best_epoch_loss


model = build()
fit(model, x=x_train, y=y_train, validation_data=(x_val, y_val))
