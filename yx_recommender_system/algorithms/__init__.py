"""
The :mod:`prediction_algorithms` package includes the prediction algorithms
available for recommendation.

The available prediction algorithms are:

.. autosummary::
    :nosignatures:

    random_pred.NormalPredictor
    baseline_only.BaselineOnly

"""


from .random_pred import RandomPredictor
from .my_algorithm import MyAlgor
from .FM import FM

__all__ = ['RandomPredictor','MyAlgor','FM']
