import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, initializers
from utils import *
import math

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
        self.b = self.add_weight(shape=(self.units,), initializer="Zeros", trainable=True)

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
        self.inter_agg2 = InterAgg(self.embed_dim * 2, self.inter_agg1, self.adj_lists)

        self.linear = Linear(2)

        # initializer_w = tf.keras.initializers.GlorotUniform()
        # initializer_b = tf.keras.initializers.Zeros()
        # self.linear_weights = tf.Variable(initial_value = initializer_w(shape=((int(math.pow(2, 3)-1) * self.embed_dim), 2), dtype='float32'), trainable=True)
        # self.linear_bias = tf.Variable(initial_value = initializer_b(shape=((int(math.pow(2, 3)-1) * self.embed_dim),), dtype='float32'), trainable=True)

    
    def call(self, inputs):
        x = self.inter_agg2(inputs)
        # x = tf.linalg.matmul(x, self.linear_weights) + self.linear_bias
        x = self.linear(x)
        x = tf.cast(x, tf.float64) + tf.cast(tf.math.log(self.prior), tf.float64)
        return x

    def print_stats(self):
        pass
        # print(self.linear.get_config(), self.linear.get_weights())

    def save_weights(self, path):
        pass
        # print("Saving weights to: ", path)


class MLP(tf.keras.layers.Layer):
    def __init__(self, feat_data, input_dim, output_dim) -> None:
        super(MLP, self).__init__()
        self.feat_data = feat_data
        # initializer = tf.keras.initializers.GlorotUniform()
        # self.mlp_weights = tf.Variable(initial_value = initializer(shape=(input_dim, output_dim), dtype='float32'), trainable=True)
        self.linear = Linear(output_dim)

    def call(self, nodes):
        return self.linear(tf.gather(self.feat_data, nodes))

# class IntraAgg_(tf.keras.layers.Layer):
#     """Intra-aggregation layer"""
#     def __init__(self) -> None:
#         super().__init__(trainable=False)

#     def call(
#         self,
#         in_embeddings,
#         adj_lists,
#         batch_indices
#         ) -> tf.Tensor:
#         for idx in batch_indices:
#             neighbor_embeddings = tf.gather(in_embeddings, adj_lists[idx])
#             mean_embedding = tf.reduce_mean(neighbor_embeddings, 0)


class IntraAgg(tf.keras.layers.Layer):

    def __init__(self) -> None:
        super().__init__()
    # [{1, 2, 3}, {2, 3, 4, 6, 1}, {3, 4, 5}]
    # 
    def call(self, embedding, nodes, neighbor_lists, unique_nodes_new_index, self_feats):
        """
        Code partially from https://github.com/williamleif/graphsage-simple/
        :param nodes: list of nodes in a batch
        :param embedding: embedding of all nodes in a batch
        :param neighbor_lists: neighbor node id list for each batch node in one relation # [[list],[list],[list]]
        :param unique_nodes_new_index
        """

        # find unique nodes
        # unique_nodes_list = list(set.union(*neighbor_lists))
        # print("neighbor_lists: ", neighbor_lists)

        neighbor_lists = tf.cast(neighbor_lists, tf.int32)

        
        # neighbor_lists = tf.reshape(neighbor_lists, [-1])
        flatten_neighbor_lists = tf.reshape(neighbor_lists, [-1])
        positive_neighbor_lists = tf.boolean_mask(neighbor_lists, neighbor_lists >= 0)


        unique_neighbor_lists, idx = tf.unique(flatten_neighbor_lists)
        unique_nodes_list = tf.boolean_mask(unique_neighbor_lists, unique_neighbor_lists >= 0)


        # print("unique_nodes_list: ", unique_nodes_list)

        # id mapping
        # CirF: Match node ID to index in unique_nodes_list
        # unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        unique_nodes = unique_nodes_list

        # CirF: Both source and destination nodes are in the block
        # mask = np.zeros((len(neighbor_lists), len(unique_nodes)))
        # print("neighbor list shape: ", tf.shape(neighbor_lists))
        # print("unique node shape: ", tf.shape(unique_nodes_list))
        mask = tf.zeros([tf.shape(neighbor_lists)[0], tf.shape(unique_nodes_list)[0]], dtype=tf.int32)

        # column_indices = [unique_nodes[n]
        #                   for neighbor_list in neighbor_lists for n in neighbor_list]

        masked_neighbor_lists = tf.boolean_mask(flatten_neighbor_lists, flatten_neighbor_lists >= 0)


        # print("masked_neighbor_lists: ", masked_neighbor_lists)


        column_indices = tf.map_fn(
            fn=lambda node: tf.cast(tf.where(node == unique_nodes_list), tf.int32), 
            elems=masked_neighbor_lists, parallel_iterations=16
            )

            
        # print("tf.where0: \n", tf.where(masked_neighbor_lists[0] == unique_nodes_list))
        # print("column indices1: ", column_indices)
        column_indices = tf.reshape(column_indices, [-1])
        # print("column indices2: ", column_indices)
        column_indices = tf.cast(column_indices, tf.int32)


        # CirF: Equivalent to
        # for neighbor_list in neighbor_lists:
        #     for n in neighbor_list:
        #         column_indices.append(unique_nodes[n])
        # -CirF
        # row_indices = [i for i in range(len(neighbor_lists))
        #                for _ in range(len(neighbor_lists[i]))]
        # neighbor_lists = tf.constant([[1,2,3,4], [2,3,4,5]])
        row_length = tf.map_fn(lambda x: tf.shape(tf.boolean_mask(x, x >= 0))[0], elems=neighbor_lists, 
            parallel_iterations=16)
        # print("row_length: ", row_length)
        # line_nums = tf.constant(list(range(tf.shape(neighbor_lists)[0]))) 
        line_nums = tf.range(tf.shape(neighbor_lists)[0])
        # print("line_nums: ", line_nums)
        row_indices = tf.repeat(line_nums, row_length)

        # row_indices = tf.constant(list(range(tf.size(masked_neighbor_lists))), dtype=tf.int64)
        # print("row indices: ", row_indices, )

        ones = tf.ones(tf.size(row_indices), dtype=tf.int32)
        # print("ones: ", ones)
        indices = tf.stack([row_indices, column_indices], axis = 1)
        # print("indices: ", indices)
        # indices = tf.reshape(indices, [-1, 2])
        # tf.print("indices: ", indices, summarize=-1)

        # mask[row_indices, column_indices] = 1
        mask = tf.tensor_scatter_nd_update(mask, indices, ones)
        # print("mask: ", mask)

        # num_neighbors = mask.sum(1, keepdims=True)
        num_neighbors = tf.reduce_sum(mask, axis=1, keepdims=True)
        # print("num_neighbors: ", num_neighbors)

        #mask = torch.true_divide(mask, num_neigh)
        mask = tf.math.divide(mask, num_neighbors)
        # print("divided mask: ", mask)

        # print("unique_nodes_new_index: ", unique_nodes_new_index)

        # neighbors_new_index = [unique_nodes_new_index[n]
        #                        for n in unique_nodes_list]


        neighbors_new_index_tensor = tf.map_fn(
            fn=lambda node: tf.cast(tf.where(node == unique_nodes_new_index), tf.int32),
            elems=unique_nodes_list, parallel_iterations=16
        )

        neighbors_new_index_tensor = tf.reshape(neighbors_new_index_tensor, [-1])

        # print("neighbors_new_index_tensor: ", neighbors_new_index_tensor)

        embed_matrix = tf.gather(embedding, neighbors_new_index_tensor)

        embed_matrix = tf.cast(embed_matrix, tf.float64)
        _feats_1 = tf.matmul(mask, embed_matrix)
        _feats_1 = tf.cast(_feats_1, tf.float32)

        # difference
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
    
    ## transpose of neighbor_features
    neighbor_features_T = tf.transpose(neighbor_features)
    
    ## apply softmax function on trainable parameter alpha
    W = tf.nn.softmax(alpha, axis = 1)
    
    ## results to be returned
    weighted_sum = tf.zeros(shape=(embed_dim, batch_size), dtype=tf.dtypes.float32)
    for r in range(num_relations):
        temp = tf.repeat(tf.reshape(W[:, r], (embed_dim, 1)), repeats=batch_size, axis = 1)
        weighted_sum += tf.math.multiply(temp, neighbor_features_T[:, r*batch_size:(r+1)*batch_size])
        
    return tf.transpose(weighted_sum)



class InterAgg(tf.keras.layers.Layer):

    def __init__(self, embed_dim, previous_layer, adj_lists):
        """
        Initialize the inter-relation aggregator
        """
        super().__init__(trainable=True)
        
        ## Set up the InterAgg variable
        self.embed_dim = embed_dim
        self.previous_layer = previous_layer
        self.adj_lists = tf.cast(adj_lists, tf.int32)
        ## Glorot uniform initializer = Xavier uniform initializer
        initializer = tf.keras.initializers.GlorotUniform()
        self.alpha = tf.Variable(initial_value = initializer(shape=(self.embed_dim*2, 3), dtype='float32') , trainable=True)
        ## Initialize 3 IntraAgg objects for 3 relations
        self.intraAgg1 = IntraAgg()
        self.intraAgg2 = IntraAgg()
        self.intraAgg3 = IntraAgg()
        

    def call(self, nodes):
        """
        :param nodes: a list of batch node indices (global)
        :param features: features of all nodes 
        :param adj_lists: a list of adjacency lists for each single-relation graph = [adj_lists_1, adj_lists_2, adj_lists_3]
        """
        ## sum of neighbors of nodes in a full batch

        # if not isinstance(nodes, list):
        #     nodes = nodes.numpy().tolist()
        
        # print("embed_dim: ", self.embed_dim)
        # print("nodes: ", nodes)
        nodes = tf.cast(nodes, tf.int32)

        # neighbors_for_batch_nodes = tf.zeros([3, 64, 10], dtype=tf.int32)
        # i = 0
        # for adj_list in self.adj_lists:
        #     # neighbors_for_batch_nodes.append(   [  set(adj_list[int(node)]) for node in nodes   ]   )
        #     # neighbors_for_batch_nodes.append()
        #     nodeNeighborTensor = tf.map_fn(fn=lambda node: tf.gather(adj_list, node), elems=nodes)
        #     # neighbors_for_batch_nodes.append(nodeNeighborTensor)
        #     neighbors_for_batch_nodes = neighbors_for_batch_nodes[i,:,:].assign(nodeNeighborTensor)
        #     i += 1

        neighbors_for_batch_nodes = tf.map_fn(
            fn=lambda adj_list: tf.map_fn(fn=lambda node: tf.gather(adj_list, node), elems=nodes, parallel_iterations=16),
             elems=self.adj_lists, parallel_iterations=16)

        # print("nodes[0]", nodes[0])
        # print("gather: ", tf.gather(self.adj_lists[0], nodes[0]))
        neighbors_for_batch_nodes = tf.cast(neighbors_for_batch_nodes, tf.int32)
        # print("neighbors_for_batch_nodes: ", neighbors_for_batch_nodes)
        
        combined_tensor = tf.concat(neighbors_for_batch_nodes, axis=0)
        combined_tensor = tf.reshape(combined_tensor, [-1])
        combined_tensor = tf.concat([combined_tensor, nodes], axis=0)
        unique_nodes_in_combined_tensor, idx = tf.unique(combined_tensor)

        # extract non-negative values in unique_nodes_in_combined_tensor
        unique_nodes_in_combined_tensor = tf.boolean_mask(unique_nodes_in_combined_tensor, unique_nodes_in_combined_tensor >= 0)

        # print("unique_nodes_in_combined_tensor: ", unique_nodes_in_combined_tensor)

        ## a set of global indices containing all the batch nodes and their neighbors
        # unique_nodes_in_combined_set =  set.union(    set.union(*neighbors_for_batch_nodes[0])  ,   set.union(*neighbors_for_batch_nodes[1])  ,  set.union(*neighbors_for_batch_nodes[2], set(nodes))  )
        
        
        # TODO: Modify all sets, dicts, lists and iterations to tensors from here.

        ## an index mapping: from global index n to local index i w.r.t combined_set
        # unique_nodes_new_index_dictionary = {n: i for i, n in enumerate(list(unique_nodes_in_combined_tensor))}
        
        unique_nodes_new_index_tensor = unique_nodes_in_combined_tensor

        # print("unique_nodes_new_index_tensor: ", unique_nodes_new_index_tensor)
        
        ## extract features of nodes in combined_set from all features
        # combined_set_features = self.previous_layer(tf.constant(list(unique_nodes_in_combined_tensor)))
        combined_set_features = self.previous_layer(unique_nodes_in_combined_tensor)
        

        ## get lists of neighbors' indices for each relation
        # r1_list = [set(neighbors_for_single_node) for neighbors_for_single_node in neighbors_for_batch_nodes[0]] # [set,...,set] 
        # r2_list = [set(neighbors_for_single_node) for neighbors_for_single_node in neighbors_for_batch_nodes[1]] # [set,...,set] 
        # r3_list = [set(neighbors_for_single_node) for neighbors_for_single_node in neighbors_for_batch_nodes[2]] # [set,...,set]
        
        r1_list_tensor = neighbors_for_batch_nodes[0]
        r2_list_tensor = neighbors_for_batch_nodes[1]
        r3_list_tensor = neighbors_for_batch_nodes[2]

        # print("r1_list_tensor: ", r1_list_tensor)


        ## get the local index of all batch nodes
        # batch_nodes_new_index = [unique_nodes_new_index_dictionary[int(n)] for n in nodes]

        unique_nodes_new_index_tensor = tf.cast(unique_nodes_new_index_tensor, tf.int32)
        
        batch_nodes_new_index_tensor = tf.map_fn(
            fn=lambda node: tf.cast(tf.where(unique_nodes_new_index_tensor == node), tf.int32),
            elems=nodes, parallel_iterations=16
            )

        batch_nodes_new_index_tensor = tf.reshape(batch_nodes_new_index_tensor, [-1])

        
        # print("batch_nodes_new_index_tensor: ", batch_nodes_new_index_tensor)


        ## get the features of all batch nodes (it is part of combined_set_features by excluding the neighbors' rows)
        ## batch_nodes_features = combined_set_features[batch_nodes_new_index]
        # batch_nodes_features = tf.gather(combined_set_features, batch_nodes_new_index)
        batch_nodes_features = tf.gather(combined_set_features, batch_nodes_new_index_tensor)
        

        r1_new_embedding_features = self.intraAgg1(combined_set_features[:, -self.embed_dim:], nodes, r1_list_tensor, unique_nodes_new_index_tensor, batch_nodes_features[:, -self.embed_dim:])
        r2_new_embedding_features = self.intraAgg2(combined_set_features[:, -self.embed_dim:], nodes, r2_list_tensor, unique_nodes_new_index_tensor, batch_nodes_features[:, -self.embed_dim:])
        r3_new_embedding_features = self.intraAgg3(combined_set_features[:, -self.embed_dim:], nodes, r3_list_tensor, unique_nodes_new_index_tensor, batch_nodes_features[:, -self.embed_dim:])
        
        neighbors_features_all_relations_concat = tf.concat((r1_new_embedding_features,
                                                             r2_new_embedding_features, 
                                                             r3_new_embedding_features), 
                                                            axis = 0)
        
  
        ## get the batch size
        batch_size = len(nodes)
        
        ## compute the weighted sum 
        inter_layer_outputs = weight_inter_agg( len(self.adj_lists) , neighbors_features_all_relations_concat, self.embed_dim * 2, self.alpha, batch_size)
        
        result = tf.concat((batch_nodes_features, inter_layer_outputs), axis = 1)
        
        return result
 



        
  
        
        
    