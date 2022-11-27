import tensorflow as tf
from tensorflow import keras
from utils import *

class Linear(keras.layers.Layer):
    def __init__(self, units):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="GlorotUniform",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="Zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class NollaFraud(tf.keras.Model):
    def __init__(self, feat_data, adj_lists, prior, embed_dim) -> None:

        super(NollaFraud, self).__init__()

        self.embed_dim = embed_dim
        self.feat_data = feat_data
        self.adj_lists = adj_lists
        self.prior = prior

        self.mlp = MLP(feat_data, feat_data.shape[1], self.embed_dim)
        self.inter_agg1 = InterAgg(self.embed_dim, self.mlp, self.adj_lists)
        self.inter_agg2 = InterAgg(
            self.embed_dim * 2, self.inter_agg1, self.adj_lists)

        self.linear = Linear(2)

    def call(self, inputs):
        x = self.inter_agg2(inputs)

        x = self.linear(x)
        x = tf.cast(x, tf.float64) + \
            tf.cast(tf.math.log(self.prior), tf.float64)
        return x

    def print_stats(self):
        pass

    def save_weights(self, path):
        pass


class MLP(tf.keras.layers.Layer):
    def __init__(self, feat_data, input_dim, output_dim) -> None:
        super(MLP, self).__init__()
        self.feat_data = feat_data

        self.linear = Linear(output_dim)

    def call(self, nodes):
        return self.linear(tf.gather(self.feat_data, nodes))


class IntraAgg(tf.keras.layers.Layer):

    def __init__(self) -> None:
        super().__init__()

    def call(self, embedding, nodes, neighbor_lists, unique_nodes_new_index, self_feats):
        """
        Code partially from https://github.com/williamleif/graphsage-simple/
        :param nodes: list of nodes in a batch
        :param embedding: embedding of all nodes in a batch
        :param neighbor_lists: neighbor node id list for each batch node in one relation 
        :param unique_nodes_new_index
        """

        neighbor_lists = tf.cast(neighbor_lists, tf.int32)

        flatten_neighbor_lists = tf.reshape(neighbor_lists, [-1])
        positive_neighbor_lists = tf.boolean_mask(
            neighbor_lists, neighbor_lists >= 0)

        unique_neighbor_lists, idx = tf.unique(flatten_neighbor_lists)
        unique_nodes_list = tf.boolean_mask(
            unique_neighbor_lists, unique_neighbor_lists >= 0)

        unique_nodes = unique_nodes_list

        mask = tf.zeros([tf.shape(neighbor_lists)[0], tf.shape(
            unique_nodes_list)[0]], dtype=tf.int32)

        masked_neighbor_lists = tf.boolean_mask(
            flatten_neighbor_lists, flatten_neighbor_lists >= 0)

        column_indices = tf.map_fn(
            fn=lambda node: tf.cast(
                tf.where(node == unique_nodes_list), tf.int32),
            elems=masked_neighbor_lists, parallel_iterations=16
        )

        column_indices = tf.reshape(column_indices, [-1])

        column_indices = tf.cast(column_indices, tf.int32)

        row_length = tf.map_fn(lambda x: tf.shape(tf.boolean_mask(x, x >= 0))[0], elems=neighbor_lists,
                               parallel_iterations=16)

        line_nums = tf.range(tf.shape(neighbor_lists)[0])

        row_indices = tf.repeat(line_nums, row_length)

        ones = tf.ones(tf.size(row_indices), dtype=tf.int32)

        indices = tf.stack([row_indices, column_indices], axis=1)

        mask = tf.tensor_scatter_nd_update(mask, indices, ones)

        num_neighbors = tf.reduce_sum(mask, axis=1, keepdims=True)

        mask = tf.math.divide(mask, num_neighbors)

        neighbors_new_index_tensor = tf.map_fn(
            fn=lambda node: tf.cast(
                tf.where(node == unique_nodes_new_index), tf.int32),
            elems=unique_nodes_list, parallel_iterations=16
        )

        neighbors_new_index_tensor = tf.reshape(
            neighbors_new_index_tensor, [-1])

        embed_matrix = tf.gather(embedding, neighbors_new_index_tensor)

        embed_matrix = tf.cast(embed_matrix, tf.float64)
        _feats_1 = tf.matmul(mask, embed_matrix)
        _feats_1 = tf.cast(_feats_1, tf.float32)

        _feats_2 = self_feats - _feats_1

        return tf.concat((_feats_1, _feats_2), 1)


def weight_inter_agg(num_relations, neighbor_features, embed_dim, alpha, batch_size):
    """
    Weight inter-relation aggregator
    :param num_relations: number of relations in the graph ( 3 )
    :param neighbor_features: Concatenated intra-aggregation results ( size = ( 3*batch_size, 2^(k+1) ) )
    :param embed_dim: the dimension of output embedding ( 2*64/2*128 )
    :param alpha: weight paramter for each relation (size = (2^(k+1), 3))
    :param n: number of nodes in a batch (256)
    """

    neighbor_features_T = tf.transpose(neighbor_features)

    W = tf.nn.softmax(alpha, axis=1)

    weighted_sum = tf.zeros(
        shape=(embed_dim, batch_size), dtype=tf.dtypes.float32)
    for r in range(num_relations):
        temp = tf.repeat(tf.reshape(
            W[:, r], (embed_dim, 1)), repeats=batch_size, axis=1)
        weighted_sum += tf.math.multiply(
            temp, neighbor_features_T[:, r*batch_size:(r+1)*batch_size])

    return tf.transpose(weighted_sum)


class InterAgg(tf.keras.layers.Layer):

    def __init__(self, embed_dim, previous_layer, adj_lists):
        """
        Initialize the inter-relation aggregator
        """
        super().__init__(trainable=True)

        self.embed_dim = embed_dim
        self.previous_layer = previous_layer
        self.adj_lists = tf.cast(adj_lists, tf.int32)

        initializer = tf.keras.initializers.GlorotUniform()
        self.alpha = tf.Variable(initial_value=initializer(
            shape=(self.embed_dim*2, 3), dtype='float32'), trainable=True)

        self.intraAgg1 = IntraAgg()
        self.intraAgg2 = IntraAgg()
        self.intraAgg3 = IntraAgg()

    def call(self, nodes):
        """
        :param nodes: a list of batch node indices (global)
        :param features: features of all nodes 
        :param adj_lists: a list of adjacency lists for each single-relation graph = [adj_lists_1, adj_lists_2, adj_lists_3]
        """

        nodes = tf.cast(nodes, tf.int32)

        neighbors_for_batch_nodes = tf.map_fn(
            fn=lambda adj_list: tf.map_fn(fn=lambda node: tf.gather(
                adj_list, node), elems=nodes, parallel_iterations=16),
            elems=self.adj_lists, parallel_iterations=16)

        neighbors_for_batch_nodes = tf.cast(
            neighbors_for_batch_nodes, tf.int32)

        combined_tensor = tf.concat(neighbors_for_batch_nodes, axis=0)
        combined_tensor = tf.reshape(combined_tensor, [-1])
        combined_tensor = tf.concat([combined_tensor, nodes], axis=0)
        unique_nodes_in_combined_tensor, idx = tf.unique(combined_tensor)

        unique_nodes_in_combined_tensor = tf.boolean_mask(
            unique_nodes_in_combined_tensor, unique_nodes_in_combined_tensor >= 0)

        unique_nodes_new_index_tensor = unique_nodes_in_combined_tensor

        combined_set_features = self.previous_layer(
            unique_nodes_in_combined_tensor)

        r1_list_tensor = neighbors_for_batch_nodes[0]
        r2_list_tensor = neighbors_for_batch_nodes[1]
        r3_list_tensor = neighbors_for_batch_nodes[2]

        unique_nodes_new_index_tensor = tf.cast(
            unique_nodes_new_index_tensor, tf.int32)

        batch_nodes_new_index_tensor = tf.map_fn(
            fn=lambda node: tf.cast(
                tf.where(unique_nodes_new_index_tensor == node), tf.int32),
            elems=nodes, parallel_iterations=16
        )

        batch_nodes_new_index_tensor = tf.reshape(
            batch_nodes_new_index_tensor, [-1])

        batch_nodes_features = tf.gather(
            combined_set_features, batch_nodes_new_index_tensor)

        r1_new_embedding_features = self.intraAgg1(
            combined_set_features[:, -self.embed_dim:], nodes, r1_list_tensor, unique_nodes_new_index_tensor, batch_nodes_features[:, -self.embed_dim:])
        r2_new_embedding_features = self.intraAgg2(
            combined_set_features[:, -self.embed_dim:], nodes, r2_list_tensor, unique_nodes_new_index_tensor, batch_nodes_features[:, -self.embed_dim:])
        r3_new_embedding_features = self.intraAgg3(
            combined_set_features[:, -self.embed_dim:], nodes, r3_list_tensor, unique_nodes_new_index_tensor, batch_nodes_features[:, -self.embed_dim:])

        neighbors_features_all_relations_concat = tf.concat((r1_new_embedding_features,
                                                             r2_new_embedding_features,
                                                             r3_new_embedding_features),
                                                            axis=0)

        batch_size = len(nodes)

        inter_layer_outputs = weight_inter_agg(len(
            self.adj_lists), neighbors_features_all_relations_concat, self.embed_dim * 2, self.alpha, batch_size)

        result = tf.concat((batch_nodes_features, inter_layer_outputs), axis=1)

        return result
