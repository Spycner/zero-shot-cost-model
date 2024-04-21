from .gat import GATConv
from .mscn import MscnConv
from .passing import MessagePassing  # noqa F401

# Mapping of string names to classes
aggregator_classes = {
    "GATConv": GATConv,
    "MscnConv": MscnConv,
}
