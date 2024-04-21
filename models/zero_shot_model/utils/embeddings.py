import torch
from torch import nn


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
