import logging

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


logger = logging.getLogger(__name__)


class DynamicLayer(nn.Module):
    """
    Defines a flexible layer for neural networks, supporting customizable activation and normalization.

    This class is designed to simplify the construction of neural network layers by offering
    customizable options for dropout, activation, and normalization. It allows for the dynamic
    selection and instantiation of activation and normalization mechanisms via class names and
    specific keyword arguments.

    Parameters:
    - p_dropout (float, optional): Specifies the dropout probability, with a default of None indicating no dropout.
    - activation_class_name (str, optional): Specifies the class name of the activation function to be used. Defaults to None, indicating no activation. Example values include 'ReLU', 'LeakyReLU', 'Sigmoid', etc.
    - activation_class_kwargs (dict, optional): Provides keyword arguments for initializing the activation class. Defaults to an empty dictionary.
    - norm_class_name (str, optional): Specifies the class name of the normalization function to be used. Defaults to None, indicating no normalization. Example values include 'BatchNorm1d', 'LayerNorm', 'InstanceNorm1d', etc.
    - norm_class_kwargs (dict, optional): Provides keyword arguments for initializing the normalization class. Defaults to an empty dictionary.
    - inplace (bool, optional): Determines whether operations (where applicable) should be performed in-place to save memory. Defaults to False. It is recommended to use this with caution as in-place operations can lead to unexpected behavior during backward operations. It should only be used when memory optimization is critical and the implications are fully understood.

    Attributes:
    - inplace (bool): Indicates whether in-place operations are enabled.
    - p_dropout (float): The probability of an element being zeroed during dropout.
    - act_class (class): The class of the activation function used.
    - act_class_kwargs (dict): Keyword arguments for the activation class instantiation.
    - norm_class (class): The class of the normalization function used.
    - norm_class_kwargs (dict): Keyword arguments for the normalization class instantiation.
    """

    def __init__(
        self,
        p_dropout=None,
        activation_class_name=None,
        activation_class_kwargs=None,
        norm_class_name=None,
        norm_class_kwargs=None,
        inplace=False,  # Defaulting inplace to False as a safer default option.
        **kwargs,
    ):
        super().__init__()
        self.inplace = inplace
        self.p_dropout = p_dropout if p_dropout is not None else 0.0
        if activation_class_name in nn.__dict__:
            self.act_class = nn.__dict__[activation_class_name]
        else:
            raise ValueError(
                f"Activation class '{activation_class_name}' not found. Please ensure it is a valid torch.nn module."
            )

        self.act_class_kwargs = activation_class_kwargs or {}

        if norm_class_name in nn.__dict__:
            self.norm_class = nn.__dict__[norm_class_name]
        else:
            raise ValueError(
                f"Normalization class '{norm_class_name}' not found. Please ensure it is a valid torch.nn module."
            )
        self.norm_class_kwargs = norm_class_kwargs or {}

    def get_act(self):
        return self.act_class(inplace=self.inplace, **self.act_class_kwargs)

    def get_norm(self, num_feats):
        return self.norm_class(
            num_feats, inplace=self.inplace, **self.norm_class_kwargs
        )


class ResidualBlock(DynamicLayer):
    """
    A ResidualBlock class that inherits from DynamicLayer to create a residual block for a neural network.

    This class constructs a residual block with optional normalization, activation, and dropout layers
    between two linear transformations. The input and output dimensions must be equal. This block can be
    used as a building block for deeper neural networks by stacking multiple such blocks.

    Parameters:
    - dim_in (int): The dimensionality of the input.
    - dim_out (int): The dimensionality of the output. Must be equal to dim_in.
    - norm (bool, optional): If True, normalization is applied after each linear transformation. Defaults to False.
    - activation (bool, optional): If True, an activation function is applied after each linear transformation
      and normalization (if normalization is applied). Defaults to False.
    - dropout (bool, optional): If True, dropout is applied after the second activation function (if activation is applied).
      This is conditionally based on the `final_layer` flag to prevent dropout in the final layer of the network,
      where it might not be desirable to randomly zero elements. Defaults to False.
    - final_layer (bool, optional): Indicates if this block is the final layer of the network. If True, dropout is not applied
      regardless of the dropout parameter to ensure the integrity of the final output. This parameter is reserved for future
      extensions and to provide flexibility in network design. Defaults to False.
    - **kwargs: Additional keyword arguments inherited from DynamicLayer.

    Attributes:
    - layers (nn.ModuleList): A list of layers constituting the residual block.

    Methods:
    - forward(x): Defines the forward pass of the ResidualBlock.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        norm=False,
        activation=False,
        dropout=False,
        final_layer=False,
        **kwargs,
    ):
        if dim_in != dim_out:
            raise ValueError(
                "In a ResidualBlock, the input dimension (dim_in) must be equal to the output dimension (dim_out) to ensure the addition operation in the residual connection is valid."
            )
        super().__init__(**kwargs)

        layers = []
        for _ in range(2):
            layers.append(nn.Linear(dim_in, dim_out))
            if norm:
                layers.append(self.get_norm(dim_out))
            if activation:
                layers.append(self.get_act())
            if (
                _ == 1 and dropout and not final_layer
            ):  # Add dropout only after the second linear layer and if not the final layer
                layers.append(nn.Dropout(p=self.p_dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        x = self.layers(x)
        return x + residual


class FcLayer(DynamicLayer):
    """
    A fully connected layer that can optionally include normalization, activation, and dropout.

    This class extends DynamicLayer to create a customizable fully connected layer for neural networks. It allows
    the inclusion of normalization, activation functions, and dropout based on the provided flags. This layer can
    be used as a building block in more complex architectures.

    Parameters:
    - dim_in (int): The dimensionality of the input.
    - dim_out (int): The dimensionality of the output.
    - norm (bool, optional): If True, normalization is applied to the layer. Defaults to False.
    - activation (bool, optional): If True, an activation function is applied to the layer. Defaults to False.
    - dropout (bool, optional): If True, dropout is applied to the layer. This is ignored if `final_layer` is True.
      Defaults to False.
    - final_layer (bool, optional): Indicates if this layer is the final layer of the network. If True, dropout is not
      applied regardless of the dropout parameter to ensure the integrity of the final output. Defaults to False.
    - **kwargs: Additional keyword arguments inherited from DynamicLayer.

    Attributes:
    - layers (nn.Sequential): A sequential container of modules that will be applied to the input in order.

    Methods:
    - forward(x): Defines the forward pass of the FcLayer.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        norm=False,
        activation=False,
        dropout=False,
        final_layer=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        layers = []
        layers.append(nn.Linear(dim_in, dim_out))
        if norm:
            layers.append(self.get_norm(dim_out))
        if activation:
            layers.append(self.get_act())
        if dropout and not final_layer:
            layers.append(nn.Dropout(p=self.p_dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FcOutModel(DynamicLayer):
    """
    This class extends DynamicLayer to construct a fully connected output model with customizable layer dimensions,
    optional residual connections, and a specified loss function. It supports special handling for cases where the
    input dimension is zero, allowing for flexible model configurations.

    Parameters:
    - output_dim (int, optional): The dimensionality of the output. Defaults to None.
    - input_dim (int, optional): The dimensionality of the input. A value of 0 indicates a special handling case.
      Defaults to None.
    - n_layers (int, optional): The number of layers in the model. Defaults to None.
    - width_factor (float, optional): A factor that determines the width (dimensionality) of each subsequent layer
      by multiplying the dimension of the previous layer. Defaults to None.
    - residual (bool, optional): If True, residual connections are used wherever possible. Defaults to True.
    - loss_class_name (str, optional): The name of the loss class to be used. If not specified, a default loss
      (CrossEntropyLoss) is used. Defaults to None.
    - loss_class_kwargs (dict, optional): Keyword arguments for the loss class constructor. Defaults to None.
    - task (str, optional): Specifies the task for which the model is being used. This parameter does not affect
      the model's structure but can be used for logging or conditional logic in extended implementations. Defaults to None.
    - **kwargs: Additional keyword arguments inherited from DynamicLayer.

    Attributes:
    - replacement_param (Parameter, optional): A special parameter initialized only when input_dim is 0. It can serve
      various purposes, such as acting as a bypass mechanism or holding a specific value for specialized computations.
      Defaults to None.
    - loss_func (nn.Module): The loss function used by the model. Initialized based on loss_class_name and
      loss_class_kwargs.
    - fcout (nn.Sequential): A sequential container of the fully connected layers, optionally including residual blocks,
      that make up the model.

    Methods:
    - forward(x): Defines the forward pass of the FcOutModel.
    """

    def __init__(
        self,
        output_dim=None,
        input_dim=None,
        n_layers=None,
        width_factor=None,
        residual=True,
        loss_class_name=None,
        loss_class_kwargs=None,
        task=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # When input_dim is 0, it indicates a special case where the model might be used in a different
        # context or as a placeholder. In such cases, instead of proceeding with the usual layer construction,
        # a single replacement parameter is initialized. This parameter can serve various purposes, such as
        # acting as a bypass mechanism or holding a specific value for specialized computations.
        if input_dim == 0:
            self.replacement_param = Parameter(torch.Tensor(output_dim))
            # The bound for uniform initialization is determined based on the output dimension,
            # following the principle of initializing weights in a range that considers the scale of the dimension.
            bound = 1 / np.sqrt(output_dim)
            init.uniform_(self.replacement_param, -bound, bound)
            return  # Early return to bypass the usual layer construction process

        if loss_class_name is not None:
            self.loss_func = nn.__dict__.get(loss_class_name)(**loss_class_kwargs)
        else:
            self.loss_func = nn.CrossEntropyLoss(**loss_class_kwargs)

        # The dimensions of the layers are calculated to gradually decrease from the input dimension to the output dimension.
        # This is achieved by creating a sequence of dimensions where each subsequent layer's dimension is a fraction of the previous one,
        # determined by the width_factor. This approach ensures a smooth transition of information through the network.
        layer_dims = [input_dim]
        for _ in range(1, n_layers):
            next_dim = int(layer_dims[-1] * width_factor)
            layer_dims.append(next_dim)
        layer_dims.append(output_dim)

        layers = []
        for layer_in, layer_out in zip(layer_dims, layer_dims[1:]):
            if not residual or layer_in != layer_out:
                layers.append(FcLayer(layer_in, layer_out, **kwargs))
            else:
                layers.append(ResidualBlock(layer_in, layer_out, **kwargs))
        self.fcout = nn.Sequential(*layers)
