import enum


class Activation(enum.Enum):
    SIGMOID = 1
    TANH = 2
    RELU = 3
    LRELU = 4


class NormType(enum.Enum):
    BATCH = 1
    GROUP = 2
    INSTANCE = 3
    LAYER = 4


class OptimizerType(enum.Enum):
    ADAM = 1
    ADADELTA = 2