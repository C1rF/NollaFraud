import argparse
from sklearn.model_selection import train_test_split
from utils import *
from layers import *


import tensorflow as tf
from tensorflow import keras

import time
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""
   NollaFraud
   Source: https://github.com/C1rF/NollaFraud
"""

parser = argparse.ArgumentParser()


parser.add_argument('--data', type=str, default='amazon',
                    help='The dataset name. [Amazon_demo, Yelp_demo, amazon,yelp]')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size 1024 for yelp, 256 for amazon.')
parser.add_argument('--lr', type=float, default=0.05,
                    help='Initial learning rate. [0.1 for amazon and 0.001 for yelp]')
parser.add_argument('--lambda_1', type=float, default=1e-4,
                    help='Weight decay (L2 loss weight).')
parser.add_argument('--embed_dim', type=int, default=96,
                    help='Node embedding size at the first layer.')
parser.add_argument('--num_epochs', type=int,
                    default=20, help='Number of epochs.')
parser.add_argument('--test_epochs', type=int, default=30,
                    help='Epoch interval to run test set.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--no_cuda', action='store_true',
                    default=False, help='Disables CUDA training.')

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
    x_train, x_test, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=0.80,
                                                        random_state=2, shuffle=True)
elif args.data == 'amazon':
    index = list(range(3305, len(labels)))
    x_train, x_test, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                        test_size=0.90, random_state=2, shuffle=True)
else:
    exit("Dataset not supported")


num_1 = len(np.where(y_train == 1)[0])
num_2 = len(np.where(y_train == 0)[0])
p0 = (num_1 / (num_1 + num_2))
p1 = 1 - p0
prior = np.array([p1, p0])

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(args.batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(args.batch_size)

optimizer = keras.optimizers.Adam(learning_rate=args.lr)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model = NollaFraud(feat_data, adj_lists, prior, args.embed_dim)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


time_callback = TimeHistory()

train_history = model.fit(train_dataset, batch_size=args.batch_size,
                          epochs=args.num_epochs, callbacks=[time_callback])
print('Train history: ', train_history.history)

overall_time = 0.0
for t in time_callback.times:
    overall_time += t
time_per_epoch = overall_time / args.num_epochs
print('Time per epoch: ', time_per_epoch)
