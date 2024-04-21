from typing import Dict, List, Optional

import dgl
import dgl.function as fn

from aggregator import MessageAggregator


class MscnConv(MessageAggregator):
    """
    Multi-Scale Convolutional Network (MSCN) Convolution Layer.

    This class implements the MSCN convolution layer, a form of message aggregator that operates on graphs to aggregate
    node features across different types of edges (etypes) and node types. It extends the functionality of the base
    MessageAggregator class to support multi-scale feature aggregation.

    Parameters:
    - hidden_dim (int): The dimensionality of the hidden feature representations.
    - cross_reducer (str): The type of cross-type reducer to use for aggregating messages across different edge types.
    - **kwargs: Additional keyword arguments.

    Attributes:
    - cross_reducer (str): The cross-type reducer used for message aggregation.
    """

    def __init__(self, hidden_dim: int = 0, cross_reducer: str = "sum", **kwargs):
        super().__init__(input_dim=2 * hidden_dim, output_dim=hidden_dim, **kwargs)
        self.cross_reducer: str = cross_reducer

    def forward(
        self,
        graph: Optional[dgl.DGLGraph] = None,
        etypes: Optional[List[str]] = None,
        in_node_types: Optional[List[str]] = None,
        out_node_types: Optional[List[str]] = None,
        feat_dict: Optional[Dict[str, dgl.ndarray]] = None,
    ) -> Dict[str, dgl.ndarray]:
        """
        Forward pass of the MSCN convolution layer.

        Parameters:
        - graph (dgl.DGLGraph): The graph on which message passing is performed.
        - etypes (List[str]): A list of edge types over which message passing is performed.
        - in_node_types (List[str]): A list of input node types for which messages are gathered.
        - out_node_types (List[str]): A list of output node types for which aggregated messages are computed.
        - feat_dict (Dict[str, dgl.ndarray]): A dictionary mapping node types to their feature tensors.

        Returns:
        - Dict[str, dgl.ndarray]: A dictionary mapping node types to their updated feature tensors after message passing.

        Raises:
        - ValueError: If any of the provided parameters are invalid or not present in the graph.

        The method aggregates features across specified edge types (etypes) and node types, supporting multi-scale
        feature aggregation. The feat_dict parameter should contain the initial features for each node type involved
        in message passing. The structure and format of feat_dict are crucial for the correct execution of the method.
        """
        # Parameter validation
        if not isinstance(graph, dgl.DGLGraph):
            raise ValueError("The graph must be an instance of dgl.DGLGraph.")
        if etypes is None or not all(etype in graph.etypes for etype in etypes):
            raise ValueError(
                "All etypes must be valid edge types present in the graph."
            )
        if in_node_types is None or not all(
            ntype in graph.ntypes for ntype in in_node_types
        ):
            raise ValueError(
                "All in_node_types must be valid node types present in the graph."
            )
        if out_node_types is None or not all(
            ntype in graph.ntypes for ntype in out_node_types
        ):
            raise ValueError(
                "All out_node_types must be valid node types present in the graph."
            )
        if feat_dict is None or not isinstance(feat_dict, dict):
            raise ValueError("feat_dict must be a dictionary of node features.")

        # If etypes is empty, the method returns feat_dict immediately.
        if len(etypes) == 0:
            return feat_dict

        with graph.local_scope():
            graph.ndata["h"] = feat_dict

            # message passing
            graph.multi_update_all(
                {etype: (fn.copy_src("h", "m"), fn.sum("m", "ft")) for etype in etypes},
                cross_reducer=self.cross_reducer,
            )

            feat = graph.ndata["h"]
            rst = graph.ndata["ft"]

            out_dict = self.combine(feat, out_node_types, rst)
            return out_dict
