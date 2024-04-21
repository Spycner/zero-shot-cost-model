import numpy as np
import torch
from torch import nn

from nn_building_blocks import FcOutModel

from feature_statistics import FeatureType


class EmbeddingInitializer(nn.Module):
    """
    Initializes embeddings with options for dimension minimization, dropout, and one-hot encoding.

    Attributes:
        p_dropout (float): The probability of an element to be zeroed.
        drop_whole_embeddings (bool): If True, applies dropout to the whole embeddings before embedding operation.
                                      This is a form of regularization that can improve generalization by preventing
                                      the model from becoming too reliant on any single embedding.
        emb_dim (int): The dimension of the embeddings.
        embed (nn.Embedding): The embedding layer.
        do (nn.Dropout): The dropout layer. Applied to the output of the embedding layer unless
                         drop_whole_embeddings is True, in which case dropout is applied before the embedding operation.

    Parameters:
        num_embeddings (int): The size of the dictionary of embeddings.
        max_emb_dim (int): The maximum dimension of the embeddings.
        p_dropout (float): The probability of an element to be zeroed.
        minimize_emb_dim (bool, optional): If True, minimizes the embedding dimension. Defaults to True.
        drop_whole_embeddings (bool, optional): If True, applies dropout to the whole embeddings before embedding operation. Defaults to False.
        one_hot (bool, optional): If True, uses one-hot encoding for embeddings. Defaults to False.
        weight_clamp_min (float, optional): The minimum value for clamping the weights. Defaults to -2.
        weight_clamp_max (float, optional): The maximum value for clamping the weights. Defaults to 2.
    """

    def __init__(
        self,
        num_embeddings,
        max_emb_dim,
        p_dropout,
        minimize_emb_dim=True,
        drop_whole_embeddings=False,
        one_hot=False,
        weight_clamp_min=-2,
        weight_clamp_max=2,
    ):
        super().__init__()

        # Validate inputs
        if not isinstance(num_embeddings, int) or num_embeddings <= 0:
            raise ValueError("num_embeddings must be a positive integer")
        if not isinstance(max_emb_dim, int) or max_emb_dim <= 0:
            raise ValueError("max_emb_dim must be a positive integer")
        if not isinstance(p_dropout, float) or not (0 <= p_dropout <= 1):
            raise ValueError("p_dropout must be a float between 0 and 1")
        if one_hot and num_embeddings > max_emb_dim:
            raise ValueError(
                "For one-hot encoding, num_embeddings must not exceed max_emb_dim"
            )

        self.p_dropout = p_dropout
        self.drop_whole_embeddings = drop_whole_embeddings
        if minimize_emb_dim:
            self.emb_dim = min(max_emb_dim, num_embeddings)
        else:
            self.emb_dim = max_emb_dim
        self.embed = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=self.emb_dim
        )
        # Note: In-place operations like clamp_ can be efficient but may lead to unexpected behavior if not carefully managed.
        self.embed.weight.data.clamp_(
            weight_clamp_min, weight_clamp_max
        )  # Initialize weights with truncated normal distribution
        if one_hot:
            self.embed.weight.requires_grad = (
                False  # Freeze weights for one-hot encoding
            )
            if num_embeddings <= max_emb_dim:
                self.embed.weight.data = torch.eye(
                    self.emb_dim
                )  # Use identity matrix for one-hot encoding
        self.do = nn.Dropout(p=p_dropout)

    def forward(self, input):
        """
        Forward pass for embedding initialization.

        Parameters:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying embedding and dropout (if applicable).
        """
        if self.drop_whole_embeddings and self.training:
            mask = torch.zeros_like(input).bernoulli_(
                1 - self.p_dropout
            )  # Generate dropout mask
            input = input * mask  # Apply dropout mask
        out = self.embed(input)  # Apply embedding
        if not self.drop_whole_embeddings:
            out = self.do(out)  # Apply dropout to embedding output
        return out


class NodeEncoder(FcOutModel):
    """
    A class for encoding nodes with features into a unified representation suitable for further processing.

    This encoder handles both numeric and categorical features, applying embeddings to the latter
    and concatenating them into a single tensor that can be fed into neural network models.

    Attributes:
        features (list): A list of feature names to be encoded.
        feature_statistics (dict): A dictionary containing statistics about features necessary for encoding.
        max_emb_dim (int): The maximum dimension for embeddings.
        drop_whole_embeddings (bool): Whether to apply dropout to the whole embeddings.
        one_hot_embeddings (bool): Whether to use one-hot encoding for embeddings.
        input_dim (int): The total dimension of the encoded input.
        embeddings (nn.ModuleDict): A dictionary of embedding layers for categorical features.

    Parameters:
        features (list): A list of feature names to be encoded.
        feature_statistics (dict): A dictionary containing statistics about features necessary for encoding.
        max_emb_dim (int, optional): The maximum dimension for embeddings. Defaults to 32.
        drop_whole_embeddings (bool, optional): Whether to apply dropout to the whole embeddings. Defaults to False.
        one_hot_embeddings (bool, optional): Whether to use one-hot encoding for embeddings. Defaults to True.
        **kwargs: Additional keyword arguments for the parent class.
    """

    def __init__(
        self,
        features,
        feature_statistics,
        max_emb_dim=32,
        drop_whole_embeddings=False,
        one_hot_embeddings=True,
        **kwargs,
    ):
        if not isinstance(features, list):
            raise TypeError("features must be a list")
        if not isinstance(feature_statistics, dict):
            raise TypeError("feature_statistics must be a dictionary")

        for feat in features:
            if feat not in feature_statistics:
                raise ValueError(f"Feature {feat} not in feature statistics")
            if not isinstance(feature_statistics[feat], dict):
                raise TypeError(f"Statistics for feature {feat} must be a dictionary")
            if (
                "type" not in feature_statistics[feat]
                or "no_vals" not in feature_statistics[feat]
            ):
                raise ValueError(
                    f"Statistics for feature {feat} must include 'type' and 'no_vals' keys"
                )

        self.features = features
        self.feature_types = [
            FeatureType[feature_statistics[feat]["type"]] for feat in features
        ]
        self.feature_indices = []

        ### Initialize embeddings and input dimensions ###

        self.input_dim = 0
        self.input_feature_index = 0
        embeddings = {}

        for _, (feat, feature_type) in enumerate(
            zip(self.features, self.feature_types)
        ):
            if feature_type == FeatureType.numeric:
                self.feature_indices.append(
                    np.arange(self.input_feature_index, self.input_feature_index + 1)
                )
                self.input_feature_index += 1

                self.input_dim += 1
            elif feature_type == FeatureType.categorical:
                self.feature_indices.append(
                    np.arange(self.input_feature_index, self.input_feature_index + 1)
                )
                self.input_feature_index += 1

                try:
                    embedding = EmbeddingInitializer(
                        feature_statistics[feat]["no_vals"],
                        max_emb_dim,
                        kwargs.get(
                            "p_dropout", 0.0
                        ),  # Safer access to p_dropout with a default value of 0.0
                        drop_whole_embeddings=drop_whole_embeddings,
                        one_hot=one_hot_embeddings,
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to initialize embedding for feature {feat}: {str(e)}"
                    ) from e
                embeddings[feat] = embedding

                self.input_dim += embedding.emb_dim
            else:
                raise NotImplementedError(
                    f"Feature type {feature_type} not implemented"
                )

        super().__init__(self.input_dim, **kwargs)
        self.embeddings = nn.ModuleDict(embeddings)

    def forward(self, input):
        """
        Encodes input features into a unified representation.

        This method processes both numeric and categorical features, applying embeddings to the latter
        and concatenating them into a single tensor. For categorical features, the input tensor is reshaped
        to ensure compatibility with the embedding layer. This operation assumes the input for categorical
        features is 1-dimensional per feature. If your input does not meet this requirement, please reshape
        it accordingly before passing it to this method.

        Parameters:
            input (Tensor): The input tensor containing feature values.

        Returns:
            Tensor: The encoded input tensor, ready for further processing.
        """
        if self.no_input_required:
            return self.replacement_param.repeat(input.shape[0], 1)

        if input.shape[1] != self.input_feature_index:
            raise ValueError(
                f"Expected input tensor width to match the total number of features ({self.input_feature_index}), but got {input.shape[1]}"
            )

        encoded_input = []
        for feat, feat_type, feat_index in zip(
            self.features, self.feature_types, self.feature_indices
        ):
            if feat_index.size == 0:
                continue  # Skip if feature index is empty, indicating an issue with initialization

            feat_data = input[:, feat_index]

            if feat_type == FeatureType.numeric:
                encoded_input.append(feat_data)
            elif feat_type == FeatureType.categorical:
                try:
                    feat_data = torch.reshape(feat_data, (-1,))
                    embd_data = self.embeddings[feat](feat_data.long())
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to encode categorical feature {feat}: {str(e)}"
                    ) from e
                encoded_input.append(embd_data)
            else:
                raise NotImplementedError(f"Feature type {feat_type} not implemented")

        # Determine the correct dimension for concatenation based on the input shape
        concat_dim = 1 if input.dim() > 1 else 0
        try:
            input_enc = torch.cat(encoded_input, dim=concat_dim)
        except Exception as e:
            raise RuntimeError(f"Failed to concatenate encoded inputs: {str(e)}") from e

        return self.fcout(input_enc)
