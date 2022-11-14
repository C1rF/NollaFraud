import tensorflow as tf


class IntraAgg(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__(trainable=False)

    def call(self, ):
        pass



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
        temp = tf.repeat( tf.reshape(W[:,1],(embed_dim,1)) ,repeats=batch_size, axis = 1)
        weighted_sum += tf.math.multiply(temp, neighbor_features_T[:, r*batch_size:(r+1)*batch_size])
        
    return tf.transpose(weighted_sum)



class InterAgg(tf.keras.layers.Layer):

    def __init__(self, embed_dim):
        """
        Initialize the inter-relation aggregator
        """
        super().__init__(trainable=False)
        
        ## Set up the InterAgg variable
        self.embed_dim = embed_dim
        ## Glorot uniform initializer = Xavier uniform initializer
        initializer = tf.keras.initializers.GlorotUniform()
        self.alpha = tf.Variable(initial_value = initializer( shape=(self.embed_dim*2, 3), dtype='float32' ) , trainable=True)
        ## Initialize 3 IntraAgg objects for 3 relations
        self.intraAgg1 = IntraAgg()
        self.intraAgg2 = IntraAgg()
        self.intraAgg3 = IntraAgg()
        

    def call(self, nodes, features, adj_lists):
        """
        :param nodes: a list of batch node indices (global)
        :param features: features of all nodes 
        :param adj_lists: a list of adjacency lists for each single-relation graph = [adj_lists_1, adj_lists_2, adj_lists_3]
        """
        ## sum of neighbors of nodes in a full batch
        neighbors_for_batch_nodes = []
        for adj_list in adj_lists:
            neighbors_for_batch_nodes.append(   [  set(adj_list[int(node)]) for node in nodes   ]   )
        
        ## a set of global indices containing all the batch nodes and their neighbors
        unique_nodes_in_combined_set =  set.union(    set.union(*neighbors_for_batch_nodes[0])  ,   set.union(*neighbors_for_batch_nodes[1])  ,  set.union(*neighbors_for_batch_nodes[2], set(nodes))  )
        
        ## an index mapping: from global index n to local index i w.r.t combined_set
        unique_nodes_new_index_dictionary = {n: i for i, n in enumerate(list(unique_nodes_in_combined_set))}
        
        ## extract features of nodes in combined_set from all features
        combined_set_features = features(list(unique_nodes_in_combined_set))
        
        ## get lists of neighbors' indices for each relation
        r1_list = [set(neighbors_for_single_node) for neighbors_for_single_node in neighbors_for_batch_nodes[0]] # [set,...,set] 
        r2_list = [set(neighbors_for_single_node) for neighbors_for_single_node in neighbors_for_batch_nodes[1]] # [set,...,set] 
        r3_list = [set(neighbors_for_single_node) for neighbors_for_single_node in neighbors_for_batch_nodes[2]] # [set,...,set]
        
        ## get the local index of all batch nodes
        batch_nodes_new_index = [unique_nodes_new_index_dictionary[int(n)] for n in nodes]
        
        ## get the features of all batch nodes (it is part of combined_set_features by excluding the neighbors' rows)
        batch_nodes_features = combined_set_features[batch_nodes_new_index]
        
        r1_new_embedding_features = self.intraAgg1(combined_set_features[:, -self.embed_dim:], nodes, r1_list, unique_nodes_new_index_dictionary, batch_nodes_features[:, -self.embed_dim:])
        r2_new_embedding_features = self.intraAgg2(combined_set_features[:, -self.embed_dim:], nodes, r2_list, unique_nodes_new_index_dictionary, batch_nodes_features[:, -self.embed_dim:])
        r3_new_embedding_features = self.intraAgg3(combined_set_features[:, -self.embed_dim:], nodes, r3_list, unique_nodes_new_index_dictionary, batch_nodes_features[:, -self.embed_dim:])
        
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
 



        
  
        
        
    
    
    
    