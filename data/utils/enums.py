from enum import Enum


class FeatureType(Enum):
    numeric = "numeric"
    categorical = "categorical"

    def __str__(self):
        return self.value


class Operator(Enum):
    NEQ = "!="
    EQ = "="
    LEQ = "<="
    GEQ = ">="
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"
    IS_NOT_NULL = "IS NOT NULL"
    IS_NULL = "IS NULL"
    IN = "IN"
    BETWEEN = "BETWEEN"

    def __str__(self):
        return self.value


class DatabaseSystem(Enum):
    POSTGRES = "postgres"

    def __str__(self):
        return self.value
