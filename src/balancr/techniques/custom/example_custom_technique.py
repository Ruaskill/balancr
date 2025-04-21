from typing import Tuple
import numpy as np
from balancr.base import BaseBalancer


class ExampleCustomBalancer(BaseBalancer):
    """
    A dummy balancing technique that simply returns the original data unchanged.
    This class serves as a minimal example of implementing the BaseBalancer interface.
    """

    def __init__(self):
        """Initialize the balancer."""
        super().__init__()

    def balance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        A dummy implementation that returns the data unchanged.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            The original X and y unchanged
        """
        return X, y
