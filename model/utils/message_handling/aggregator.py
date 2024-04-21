from typing import Dict, List, Optional

import torch

from model.utils.nn_building_blocks import FcOutModel


class MessageAggregator(FcOutModel):
    """
    A base class for message aggregation in graph neural networks.

    This class is designed to be extended by specific message aggregation implementations. It provides a framework
    for combining node features and aggregated messages in a graph structure.

    Attributes:
        test (bool): A flag indicating whether the aggregator is in test mode. In test mode, certain operations
                     may be simplified for debugging purposes. This flag also controls the behavior of the combine
                     method, where in test mode, features are simply added instead of being processed through
                     fully connected layers.
    """

    def __init__(self, test: bool = False, **fc_out_kwargs) -> None:
        super().__init__(**fc_out_kwargs)
        self.test = test

    def forward(
        self,
        graph: Optional[torch.Tensor] = None,
        etypes: Optional[List[str]] = None,
        out_node_types: Optional[List[str]] = None,
        feat_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        raise NotImplementedError

    def combine(
        self,
        feat: Dict[str, torch.Tensor],
        out_node_types: List[str],
        rst: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Combines the original node features with the aggregated messages.

        This method takes the original node features and the results of message aggregation, and combines them
        according to the mode of operation (test or not).

        Args:
            feat (dict): A dictionary containing the original features for each node type.
            out_node_types (list): A list of node types for which the combined features are to be computed.
            rst (dict): A dictionary containing the aggregated messages for each node type.

        Returns:
            dict: A dictionary containing the combined features for each node type.
        """
        out_dict: Dict[str, torch.Tensor] = dict()
        common_types = set(feat.keys()) & set(rst.keys()) & set(out_node_types)

        # Pre-allocate memory for concatenated features to improve performance
        concat_features = {
            out_type: torch.empty(
                (
                    feat[out_type].shape[0],
                    feat[out_type].shape[1] + rst[out_type].shape[1],
                ),
                device=feat[out_type].device,
            )
            for out_type in common_types
        }

        for out_type in common_types:
            if feat[out_type].shape[0] != rst[out_type].shape[0]:
                raise ValueError(
                    f"Shape mismatch for node type {out_type}: feat and rst shapes do not match."
                )
            # Pre-compute concatenated features outside the conditional to avoid duplication
            torch.cat(
                [feat[out_type], rst[out_type]], dim=1, out=concat_features[out_type]
            )

            # send through fully connected layers
            if not self.test:
                out_dict[out_type] = self.fcout(concat_features[out_type])
            # simply add in test mode for simplified operation
            else:
                out_dict[out_type] = feat[out_type] + rst[out_type]
        return out_dict
