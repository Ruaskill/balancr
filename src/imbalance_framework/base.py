from abc import ABC, abstractmethod
import numpy as np


class BaseBalancer(ABC):
    """Base class for all balancing techniques"""

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def balance(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Balance the dataset"""
        pass
