from typing import Dict, Tuple, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from model.utils.nn_building_blocks import FcOutModel
from model.utils.node_building.node_encoder import NodeEncoder
from model.utils.message_handling import PassDirection, aggregator_classes


class ZeroShotModel(pl.LightningModule, FcOutModel):
    def __init__(
        self,
        device: str = "cpu",
        hidden_dim: Optional[int] = None,
        final_mlp_kwargs: Optional[Dict] = None,
        output_dim: int = 1,
        tree_layer_name: Optional[str] = None,
        tree_layer_kwargs: Optional[Dict] = None,
        test: bool = False,
        skip_message_passing: bool = False,
        node_type_kwargs: Optional[Dict] = None,
        feature_statistics: Optional[Dict] = None,
        add_tree_model_types: Optional[List[str]] = None,
        prepasses: Optional[List[Dict]] = None,
        plan_featurization: Optional[Dict] = None,
        encoders: Optional[List[Tuple[str, Dict]]] = None,
        label_norm: Optional[Dict] = None,
    ):
        """
        Initializes the ZeroShotModel, a PyTorch Lightning module for zero-shot learning.

        Args:
            device (str): The device to run the model on, defaults to "cpu".
            hidden_dim (Optional[int]): The dimensionality of the hidden layers.
            final_mlp_kwargs (Optional[Dict]): Additional keyword arguments for the final MLP layer.
            output_dim (int): The dimensionality of the output layer.
            tree_layer_name (Optional[str]): The name of the tree layer to use.
            tree_layer_kwargs (Optional[Dict]): Additional keyword arguments for the tree layer.
            test (bool): Flag indicating whether the model is in test mode.
            skip_message_passing (bool): Flag to skip the message passing step.
            node_type_kwargs (Optional[Dict]): Additional keyword arguments for node type encoders.
            feature_statistics (Optional[Dict]): Statistics of the features for normalization.
            add_tree_model_types (Optional[List[str]]): Additional tree model types to include.
            prepasses (Optional[List[Dict]]): Prepass configurations for message passing.
            plan_featurization (Optional[Dict]): Configuration for plan featurization.
            encoders (Optional[List[Tuple[str, Dict]]]): Encoders for different node types.
            label_norm (Optional[Dict]): Configuration for label normalization.
        """
        super().__init__(
            output_dim=output_dim,
            input_dim=hidden_dim,
            final_out_layer=True,
            **final_mlp_kwargs,
        )

        self.label_norm = label_norm

        self.test = test
        self.skip_message_passing = skip_message_passing
        self.device = device
        self.hidden_dim = hidden_dim

        # Initialize models for different tree edge types
        tree_model_types = add_tree_model_types + [
            "to_plan",
            "intra_plan",
            "intra_pred",
        ]
        self.tree_models = nn.ModuleDict(
            {
                node_type: aggregator_classes[tree_layer_name](
                    hidden_dim=self.hidden_dim, **tree_layer_kwargs
                )
                for node_type in tree_model_types
            }
        )

        self.prepasses = prepasses

        if plan_featurization is not None:
            self.plan_featurization = plan_featurization
            node_type_kwargs.update(output_dim=hidden_dim)
            self.node_type_encoders = nn.ModuleDict(
                {
                    enc_name: NodeEncoder(
                        features, feature_statistics, **node_type_kwargs
                    )
                    for enc_name, features in encoders
                }
            )

    def encode_node_types(
        self, g, features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes node types using the appropriate encoders.

        Args:
            g: The graph.
            features (Dict[str, torch.Tensor]): The features to encode, keyed by node type.

        Returns:
            Dict[str, torch.Tensor]: The encoded features, keyed by node type.
        """
        hidden_dict = dict()
        for node_type, input_features in features.items():
            if node_type not in self.node_type_encoders:
                assert node_type.startswith("plan") or node_type.startswith(
                    "logical_pred"
                )

                node_type_m = self.node_type_encoders.get(
                    "logical_pred" if node_type.startswith("logical_pred") else "plan"
                )
            else:
                node_type_m = self.node_type_encoders[node_type]
            hidden_dict[node_type] = node_type_m(input_features)

        return hidden_dict

    def forward(self, input: Tuple) -> torch.Tensor:
        """
        Forward pass of the ZeroShotModel.

        Args:
            input (Tuple): A tuple containing the graph and features.

        Returns:
            torch.Tensor: The output of the model.
        """
        graph, features = input
        features = self.encode_node_types(graph, features)
        out = self.message_passing(graph, features)

        return out

    def message_passing(self, g, feat_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Performs bottom-up message passing on the graph encoding of the queries in the batch.

        Args:
            g: The graph.
            feat_dict (Dict[str, torch.Tensor]): The feature dictionary.

        Returns:
            torch.Tensor: The hidden states of the root nodes.
        """
        if not self.skip_message_passing:
            pass_directions = [
                PassDirection(g=g, **prepass_kwargs)
                for prepass_kwargs in self.prepasses
            ]

            if g.max_pred_depth is not None:
                for d in reversed(range(g.max_pred_depth)):
                    pd = PassDirection(
                        model_name="intra_pred",
                        g=g,
                        e_name="intra_predicate",
                        n_dest=f"logical_pred_{d}",
                    )
                    pass_directions.append(pd)

            pass_directions.append(
                PassDirection(model_name="to_plan", g=g, e_name="to_plan")
            )

            for d in reversed(range(g.max_depth)):
                pd = PassDirection(
                    model_name="intra_plan", g=g, e_name="intra_plan", n_dest=f"plan{d}"
                )
                pass_directions.append(pd)

            combined_e_types = set()
            for pd in pass_directions:
                combined_e_types.update(pd.etypes)
            if combined_e_types != set(g.canonical_etypes):
                raise ValueError(
                    "The combined edge types do not match the graph's canonical edge types."
                )

            for pd in pass_directions:
                if len(pd.etypes) > 0:
                    out_dict = self.tree_models[pd.model_name](
                        g,
                        etypes=pd.etypes,
                        in_node_types=pd.in_types,
                        out_node_types=pd.out_types,
                        feat_dict=feat_dict,
                    )
                    for out_type, hidden_out in out_dict.items():
                        feat_dict[out_type] = hidden_out

        out = feat_dict["plan0"]

        if not self.test:
            out = self.fcout(out)

        return out
