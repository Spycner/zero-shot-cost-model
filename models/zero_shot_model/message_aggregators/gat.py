import torch
from torch import nn
import dgl.function as fn
from dgl.nn import edge_softmax
from dgl.utils import expand_as_pair


from .aggregator import MessageAggregator


class GATConv(MessageAggregator):
    """
    Graph Attention Network (GAT) Convolution Layer.

    This class implements the GAT convolution layer, a form of message aggregator that utilizes the attention mechanism
    to weigh the importance of neighboring nodes' features when aggregating them. It extends the functionality of the
    base MessageAggregator class to incorporate multi-head attention for more expressive representation learning.

    Parameters:
    - hidden_dim (int): The dimensionality of the hidden feature representations.
    - num_heads (int): The number of attention heads to use for multi-head attention.
    - feat_drop (float): Dropout rate applied to the input features.
    - attn_drop (float): Dropout rate applied to the attention coefficients.
    - negative_slope (float): The negative slope parameter for the LeakyReLU activation function used in computing attention scores.
    - **fc_out_kwargs: Additional keyword arguments for the output fully connected layer.

    Attributes:
    - _num_heads (int): The number of attention heads.
    - _in_src_feats (int): The input feature size for source nodes.
    - _in_dst_feats (int): The input feature size for destination nodes.
    - _out_feats (int): The output feature size.
    - fc (nn.Linear): The fully connected layer applied to input features.
    - attn_l (nn.Parameter): The learnable parameter vector for source node attention.
    - attn_r (nn.Parameter): The learnable parameter vector for destination node attention.
    - feat_drop (nn.Dropout): Dropout layer for input features.
    - attn_drop (nn.Dropout): Dropout layer for attention coefficients.
    - leaky_relu (nn.LeakyReLU): LeakyReLU activation function for computing attention scores.

    The forward pass of the GATConv layer takes a graph and its node features as input and returns the node features
    transformed by the GAT convolution process. This involves computing attention scores based on the node features,
    applying dropout to these scores, and then using them to weight the aggregation of neighboring node features.
    """

    def __init__(
        self,
        hidden_dim=0,
        num_heads=4,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        **fc_out_kwargs,
    ):
        super().__init__(
            input_dim=(num_heads + 1) * hidden_dim,
            output_dim=hidden_dim,
            **fc_out_kwargs,
        )
        in_feats = hidden_dim
        out_feats = hidden_dim

        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats

        self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(
        self,
        graph=None,
        etypes=None,
        in_node_types=None,
        out_node_types=None,
        feat_dict=None,
    ):
        with graph.local_scope():
            # Validate the shapes of tensors in feat_dict
            for node_type, feat in feat_dict.items():
                expected_shape = (self._in_src_feats,)
                if feat.shape[1:] != expected_shape:
                    raise ValueError(
                        f"Feature shape for node type {node_type} is incorrect. Expected shape with {expected_shape}, got {feat.shape[1:]}."
                    )

            # Store the input features in the graph's node data dictionary under the key "h"
            graph.ndata["h"] = feat_dict

            # Compute the source features for each node type by applying a linear transformation followed by a dropout,
            # and then reshaping the result to separate the heads for multi-head attention.
            feat_src = {
                t: self.fc(self.feat_drop(feat_dict[t])).view(
                    -1, self._num_heads, self._out_feats
                )
                for t in in_node_types
            }

            # Similarly, compute the destination features for each node type.
            feat_dst = {
                t: self.fc(self.feat_drop(feat_dict[t])).view(
                    -1, self._num_heads, self._out_feats
                )
                for t in out_node_types
            }

            # Compute the attention scores for the source nodes by multiplying the source features
            # with the attention vector (self.attn_l), summing over the feature dimension, and adding an extra dimension
            # at the end for broadcasting.
            el = {
                t: (feat_src[t] * self.attn_l).sum(dim=-1).unsqueeze(-1)
                for t in in_node_types
            }

            # Compute the attention scores for the destination nodes in a similar manner using self.attn_r.
            er = {
                t: (feat_dst[t] * self.attn_r).sum(dim=-1).unsqueeze(-1)
                for t in out_node_types
            }

            # Ensure etypes, in_node_types, and out_node_types are correctly provided and aligned
            if not (set(etypes) <= set(zip(in_node_types, out_node_types))):
                raise ValueError(
                    "Provided etypes, in_node_types, and out_node_types are not correctly aligned or provided."
                )

            # Update the graph's edge data with the computed source and destination features and attention scores.
            for etype in etypes:
                t_in, edge_t, t_out = etype
                if t_in not in in_node_types or t_out not in out_node_types:
                    continue  # Skip updating if the node types are not in the provided lists
                graph[etype].srcdata.update({"ft": feat_src[t_in], "el": el[t_in]})
                graph[etype].dstdata.update({"er": er[t_out]})

            # Apply the attention mechanism to compute the edge attention scores by summing the source and destination
            # attention scores and applying a leaky ReLU activation.
            for etype in etypes:
                graph.apply_edges(fn.u_add_v("el", "er", "e"), etype=etype)

            # Extract the edge attention scores from the graph's edge data and apply a softmax operation
            # to normalize them. The attention dropout is then applied.
            e_dict = graph.edata.pop("e")
            attention_dict = dict()
            for etype, e_data in e_dict.items():
                e = self.leaky_relu(e_data)
                edge_sm = self.attn_drop(edge_softmax(graph[etype], e))
                attention_dict[etype] = edge_sm

            # Store the normalized attention scores back in the graph's edge data under the key "a".
            graph.edata["a"] = attention_dict

            # Perform message passing using the computed attention scores to aggregate information from neighboring nodes.
            graph.multi_update_all(
                {
                    etype: (fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
                    for etype in etypes
                },
                cross_reducer="sum",
            )

            # Retrieve the updated node features from the graph's node data and reshape them to concatenate the features
            # from all heads.
            feat = graph.ndata["h"]
            rst = {
                t: ndata.view(-1, self._num_heads * self._in_src_feats)
                for t, ndata in graph.dstdata["ft"].items()
            }

            # Combine the original node features with the aggregated messages to produce the final output.
            out_dict = self.combine(feat, out_node_types, rst)

            return out_dict
