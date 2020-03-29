
__author__ = "Markus Löning"
__all__ = ["NotEvaluatedError"]


class NotEvaluatedError(ValueError, AttributeError):
    """Exception class to raise if evaluator is used before having
    evaluated any metric.
    """