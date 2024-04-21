import pytorch_lightning as pl
import torch.nn as nn

from models.zero_shot_model.utils.nn_building_blocks import FcOutModel
from models.zero_shot_model.utils.node_encoder import NodeEncoder

import message_aggregators


class ZeroShotModel(pl.LightningModule, FcOutModel):
    def __init__(
        self,
        device="cpu",
        hidden_dim=None,
        final_mlp_kwargs=None,
        output_dim=1,
        tree_layer_name=None,
        tree_layer_kwargs=None,
        test=False,
        skip_message_passing=False,
        node_type_kwargs=None,
        feature_statistics=None,
        add_tree_model_types=None,
        prepasses=None,
        plan_featurization=None,
        encoders=None,
        label_norm=None,
    ):
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

        # use different models per edge type
        tree_model_types = add_tree_model_types + [
            "to_plan",
            "intra_plan",
            "intra_pred",
        ]
        self.tree_models = nn.ModuleDict(
            {
                node_type: message_aggregators.__dict__[tree_layer_name](
                    hidden_dim=self.hidden_dim, **tree_layer_kwargs
                )
                for node_type in tree_model_types
            }
        )

        # these message passing steps are performed in the beginning (dependent on the concrete database system at hand)
        self.prepasses = prepasses

        if plan_featurization is not None:
            self.plan_featurization = plan_featurization
            # different models to encode plans, tables, columns, filter_columns and output_columns
            node_type_kwargs.update(output_dim=hidden_dim)
            self.node_type_encoders = nn.ModuleDict(
                {
                    enc_name: NodeEncoder(
                        features, feature_statistics, **node_type_kwargs
                    )
                    for enc_name, features in encoders
                }
            )
