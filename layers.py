import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, initializers
from utils import *

class NollaFraud(tf.keras.Model):
    def __init__(self, feat_data, adj_lists, prior) -> None:
        super().__init__()
        self.mlp = MLP(feat_data, 64)
        self.feat_data = feat_data
        self.adj_lists = adj_lists
        self.prior = prior
        self.inter_agg1 = InterAgg(64, self.mlp, self.adj_lists)
        self.inter_agg2 = InterAgg(128, self.inter_agg1, self.adj_lists)
    
    def call(self, inputs):
        # x = self.mlp(inputs)
        # x = self.inter_agg1(inputs, self.mlp, self.adj_list)
        print('Input to the model: ', inputs)
        x = self.inter_agg2(inputs)
        # print_with_color('AFTER_AGG: ')
        # l = 0
        # for element in x:
        #     if l > 10: break
        #     print(element)
        #     l += 1
        x = layers.Dense(2)(x)
        x = layers.LeakyReLU(alpha=0.3)(x)
        # print_with_color("AGG RES:")
        # print_with_color(x)
        x = layers.Dense(1, activation="sigmoid")(tf.cast(x, tf.float64) + tf.cast(tf.math.log(self.prior), tf.float64))
        # x = layers.Softmax()(x)
        # x = tf.math.argmax(x, 1)
        # print("SCORE: ", x)

        return x


class MLP(tf.keras.layers.Layer):
    def __init__(self, feat_data, output_dim) -> None:
        super().__init__()
        self.feat_data = feat_data
        self.output_dim = output_dim

    def call(self, nodes):
        result = layers.Dense(self.output_dim, 
            activation="relu"
            # kernel_initializer=initializers.Ones(),
            # bias_initializer=initializers.Ones(),
            )(tf.gather(self.feat_data, nodes))
        print_with_color("MLP result:")
        print_with_color(result)
        return result

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
        super().__init__(trainable=False)

    def call(self, embedding, nodes, neighbor_lists, unique_nodes_new_index, self_feats):
        """
        Code partially from https://github.com/williamleif/graphsage-simple/
        :param nodes: list of nodes in a batch
        :param embedding: embedding of all nodes in a batch
        :param neighbor_lists: neighbor node id list for each batch node in one relation # [[list],[list],[list]]
        :param unique_nodes_new_index
        """

        # find unique nodes
        unique_nodes_list = list(set.union(*neighbor_lists))

        # id mapping
        # CirF: Match node ID to index in unique_nodes_list
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        # CirF: Both source and destination nodes are in the block
        # mask = tf.zeros(len(neighbor_lists), len(unique_nodes))
        mask = np.zeros((len(neighbor_lists), len(unique_nodes)))

        column_indices = [unique_nodes[n]
                          for neighbor_list in neighbor_lists for n in neighbor_list]
        # CirF: Equivalent to
        # for neighbor_list in neighbor_lists:
        #     for n in neighbor_list:
        #         column_indices.append(unique_nodes[n])
        # -CirF
        row_indices = [i for i in range(len(neighbor_lists))
                       for _ in range(len(neighbor_lists[i]))]

        mask[row_indices, column_indices] = 1

        num_neighbors = mask.sum(1, keepdims=True)
        #mask = torch.true_divide(mask, num_neigh)
        mask = mask / num_neighbors
        # print("MASK: ", mask)

        neighbors_new_index = [unique_nodes_new_index[n]
                               for n in unique_nodes_list]

        embed_matrix = tf.gather(embedding, neighbors_new_index)

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
    # print("neighbor shape: ", neighbor_features.shape)
    
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
        self.adj_lists = adj_lists
        ## Glorot uniform initializer = Xavier uniform initializer
        initializer = tf.keras.initializers.GlorotUniform()
        self.alpha = tf.Variable(initial_value = initializer( shape=(self.embed_dim*2, 3), dtype='float32' ) , trainable=True)
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

        if not isinstance(nodes, list):
            nodes = nodes.numpy().tolist()
        
        # print("Length of nodes: ", len(nodes))
        neighbors_for_batch_nodes = []
        for adj_list in self.adj_lists:
            neighbors_for_batch_nodes.append(   [  set(adj_list[int(node)]) for node in nodes   ]   )
        
        ## a set of global indices containing all the batch nodes and their neighbors
        unique_nodes_in_combined_set =  set.union(    set.union(*neighbors_for_batch_nodes[0])  ,   set.union(*neighbors_for_batch_nodes[1])  ,  set.union(*neighbors_for_batch_nodes[2], set(nodes))  )
        
        ## an index mapping: from global index n to local index i w.r.t combined_set
        unique_nodes_new_index_dictionary = {n: i for i, n in enumerate(list(unique_nodes_in_combined_set))}
        
        ## extract features of nodes in combined_set from all features
        # print("Inputs to the previous layer in InterAgg: ",list(unique_nodes_in_combined_set))
        combined_set_features = self.previous_layer(tf.constant(list(unique_nodes_in_combined_set)))
        # print("Previous Layer Output Shape: ", combined_set_features.shape)
        
        ## get lists of neighbors' indices for each relation
        r1_list = [set(neighbors_for_single_node) for neighbors_for_single_node in neighbors_for_batch_nodes[0]] # [set,...,set] 
        r2_list = [set(neighbors_for_single_node) for neighbors_for_single_node in neighbors_for_batch_nodes[1]] # [set,...,set] 
        r3_list = [set(neighbors_for_single_node) for neighbors_for_single_node in neighbors_for_batch_nodes[2]] # [set,...,set]
        
        ## get the local index of all batch nodes
        batch_nodes_new_index = [unique_nodes_new_index_dictionary[int(n)] for n in nodes]
        
        # print("Length of batch_nodes_new_index: ", len(batch_nodes_new_index))
        # print("New Index:", batch_nodes_new_index)
        ## get the features of all batch nodes (it is part of combined_set_features by excluding the neighbors' rows)
        ## batch_nodes_features = combined_set_features[batch_nodes_new_index]
        batch_nodes_features = tf.gather(combined_set_features, batch_nodes_new_index)
        
        r1_new_embedding_features = self.intraAgg1(combined_set_features[:, -self.embed_dim:], nodes, r1_list, unique_nodes_new_index_dictionary, batch_nodes_features[:, -self.embed_dim:])
        r2_new_embedding_features = self.intraAgg2(combined_set_features[:, -self.embed_dim:], nodes, r2_list, unique_nodes_new_index_dictionary, batch_nodes_features[:, -self.embed_dim:])
        r3_new_embedding_features = self.intraAgg3(combined_set_features[:, -self.embed_dim:], nodes, r3_list, unique_nodes_new_index_dictionary, batch_nodes_features[:, -self.embed_dim:])
        
        neighbors_features_all_relations_concat = tf.concat((r1_new_embedding_features,
                                                             r2_new_embedding_features, 
                                                             r3_new_embedding_features), 
                                                            axis = 0)
  
        ## get the batch size
        batch_size = len(nodes)
        # print("Length of adj_lists: ", len(self.adj_lists))
        # print("Shape of neighbors_features_all_relations_concat: ", neighbors_features_all_relations_concat.shape)
        # print("Embedding Dimension: ", self.embed_dim)
        # print("Alpha Shape: ", self.alpha.shape)
        # print("Batch Size: ", batch_size)
        
        ## compute the weighted sum 
        inter_layer_outputs = weight_inter_agg( len(self.adj_lists) , neighbors_features_all_relations_concat, self.embed_dim * 2, self.alpha, batch_size)
        
        result = tf.concat((batch_nodes_features, inter_layer_outputs), axis = 1)
        
        return result
 



        
  
        
        
    