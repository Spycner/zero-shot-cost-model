from enum import Enum


class FeatureType(Enum):
    numeric = "numeric"
    categorical = "categorical"

    def __str__(self):
        return self.value
