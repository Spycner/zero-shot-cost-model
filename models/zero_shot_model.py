import pytorch_lightning as pl
import torch.nn as nn

from models.utils.nn_building_blocks import FcOutModel
from models.utils.node_building.node_encoder import NodeEncoder
from models.utils.message_handling import PassDirection, aggregator_classes


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
                node_type: aggregator_classes[tree_layer_name](
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

    def encode_node_types(self, g, features):
        # initialize hidden state per node type
        hidden_dict = dict()
        for node_type, input_features in features.items():
            # encode all plans with same model
            if node_type not in self.node_type_encoders:
                assert node_type.startswith("plan") or node_type.startswith(
                    "logical_pred"
                )

                if node_type.startswith("logical_pred"):
                    node_type_m = self.node_type_encoders["logical_pred"]
                else:
                    node_type_m = self.node_type_encoders["plan"]
            else:
                node_type_m = self.node_type_encoders[node_type]
            hidden_dict[node_type] = node_type_m(input_features)

        return hidden_dict

    def forward(self, input):
        graph, features = input
        features = self.encode_node_types(graph, features)
        out = self.message_passing(graph, features)

        return out

    def message_passing(self, g, feat_dict):
        """
        Bottom-up message passing on the graph encoding of the queries in the batch. Returns the hidden states of the
        root nodes.
        """

        # also allow skipping this for testing
        if not self.skip_message_passing:
            # all passes before predicates, to plan and intra_plan passes
            pass_directions = [
                PassDirection(g=g, **prepass_kwargs)
                for prepass_kwargs in self.prepasses
            ]

            if g.max_pred_depth is not None:
                # intra_pred from deepest node to top node
                for d in reversed(range(g.max_pred_depth)):
                    pd = PassDirection(
                        model_name="intra_pred",
                        g=g,
                        e_name="intra_predicate",
                        n_dest=f"logical_pred_{d}",
                    )
                    pass_directions.append(pd)

            # filter_columns & output_columns to plan
            pass_directions.append(
                PassDirection(model_name="to_plan", g=g, e_name="to_plan")
            )

            # intra_plan from deepest node to top node
            for d in reversed(range(g.max_depth)):
                pd = PassDirection(
                    model_name="intra_plan", g=g, e_name="intra_plan", n_dest=f"plan{d}"
                )
                pass_directions.append(pd)

            # make sure all edge types are considered in the message passing
            combined_e_types = set()
            for pd in pass_directions:
                combined_e_types.update(pd.etypes)
            assert combined_e_types == set(g.canonical_etypes)

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

        # compute top nodes of dags
        out = feat_dict["plan0"]

        # feed them into final feed forward network
        if not self.test:
            out = self.fcout(out)

        return out
