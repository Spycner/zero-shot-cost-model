import torch.nn as nn


class DynamicLayer(nn.Module):
    """
    Defines a flexible layer for neural networks, supporting customizable activation and normalization.

    This class is designed to simplify the construction of neural network layers by offering
    customizable options for dropout, activation, and normalization. It allows for the dynamic
    selection and instantiation of activation and normalization mechanisms via class names and
    specific keyword arguments.

    Parameters:
    - p_dropout (float, optional): Specifies the dropout probability, with a default of None indicating no dropout.
    - activation_class_name (str, optional): Specifies the class name of the activation function to be used. Defaults to None, indicating no activation.
    - activation_class_kwargs (dict, optional): Provides keyword arguments for initializing the activation class. Defaults to an empty dictionary.
    - norm_class_name (str, optional): Specifies the class name of the normalization function to be used. Defaults to None, indicating no normalization.
    - norm_class_kwargs (dict, optional): Provides keyword arguments for initializing the normalization class. Defaults to an empty dictionary.
    - inplace (bool, optional): Determines whether operations (where applicable) should be performed in-place to save memory. Defaults to False.

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
        inplace=False,
        **kwargs,
    ):
        super().__init__()
        # initialize base NN
        self.inplace = inplace
        self.p_dropout = p_dropout if p_dropout is not None else 0.0
        self.activation = self._get_activation(
            activation_class_name, activation_class_kwargs or {}
        )
        self.norm = self._get_norm(norm_class_name, norm_class_kwargs or {})

    def _get_activation(self, class_name: str, class_kwargs: dict) -> nn.Module:
        """Dynamically instantiates and returns the activation class with the provided arguments."""
        if class_name is None:
            return None
        act_class = nn.__dict__.get(class_name)
        if act_class:
            return act_class(**class_kwargs)
        else:
            raise ValueError(f"Activation class {class_name} not found in torch.nn")

    def _get_norm(self, class_name: str, class_kwargs: dict) -> nn.Module:
        """Dynamically instantiates and returns the normalization class with the provided arguments."""
        if class_name is None:
            return None
        norm_class = nn.__dict__.get(class_name)
        if norm_class:
            return norm_class(**class_kwargs)
        else:
            raise ValueError(f"Normalization class {class_name} not found in torch.nn")
