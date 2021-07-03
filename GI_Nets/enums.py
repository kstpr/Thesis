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
    NONE = 5


class OptimizerType(enum.Enum):
    ADAM = 1
    ADADELTA = 2

class OutputType(enum.Enum):
    DIRECT_IMAGE = 1 # interpreted as an image in [0, 1]
    MASK = 2 # interpreted as a mask in log space (in [-1, 1]) transformed with (x + 1)/ 2 to [0, 1]
    ALBEDO_MULT_W_MASK = 3 # output is a mask in log space, carried to exp space and then multiplied with albedo, resulting to [0, 1] image