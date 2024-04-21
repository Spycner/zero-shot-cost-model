from .gat import GATConv
from .mscn import MscnConv

# Mapping of string names to classes
aggregator_classes = {
    "GATConv": GATConv,
    "MscnConv": MscnConv,
}
