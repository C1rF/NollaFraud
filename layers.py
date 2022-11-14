import numpy as np
import tensorflow as tf


class IntraAgg_(tf.keras.layers.Layer):
    """Intra-aggregation layer"""
    def __init__(self) -> None:
        super().__init__(trainable=False)

    def call(
        self,
        in_embeddings: tf.Tensor,
        adj_lists: list[list[int]],
        batch_indices: list[int]
        ) -> tf.Tensor:
        for idx in batch_indices:
            neighbor_embeddings = tf.gather(in_embeddings, adj_lists[idx])
            mean_embedding = tf.reduce_mean(neighbor_embeddings, 0)


class IntraAgg(tf.keras.layers.Layer):

    def __init__(self) -> None:
        super().__init__(trainable=False)

    def forward(self, embedding, nodes, neighbor_lists, unique_nodes_new_index, self_feats):
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
        mask = np.zeros(len(neighbor_lists), len(unique_nodes))

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

        num_neighbors = mask.sum(1, keepdim=True)
        #mask = torch.true_divide(mask, num_neigh)
        mask = mask / num_neighbors

        neighbors_new_index = [unique_nodes_new_index[n]
                               for n in unique_nodes_list]

        embed_matrix = embedding[neighbors_new_index]

        _feats_1 = tf.matmul(mask, embed_matrix)

        # difference
        _feats_2 = self_feats - _feats_1

        return tf.concat((_feats_1, _feats_2), 1)