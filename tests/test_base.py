import pytest
import numpy as np
from balancr import BaseBalancer


def test_cannot_instantiate_abstract_base():
    """Test that BaseBalancer cannot be instantiated directly"""
    with pytest.raises(TypeError):
        BaseBalancer()


def test_concrete_implementation():
    """Test that a concrete implementation must implement balance method"""

    class IncompleteBalancer(BaseBalancer):
        pass

    with pytest.raises(TypeError):
        IncompleteBalancer()


def test_valid_implementation():
    """Test that a valid implementation can be instantiated and used"""

    class ValidBalancer(BaseBalancer):
        def balance(self, X, y):
            return X, y

    # Should not raise any exceptions
    balancer = ValidBalancer()
    assert isinstance(balancer, BaseBalancer)
    assert balancer.name == "ValidBalancer"

    # Test balance method with sample data
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    X_balanced, y_balanced = balancer.balance(X, y)
    assert np.array_equal(X_balanced, X)
    assert np.array_equal(y_balanced, y)


def test_name_attribute():
    """Test that the name attribute is set correctly"""

    class CustomBalancer(BaseBalancer):
        def balance(self, X, y):
            return X, y

    balancer = CustomBalancer()
    assert balancer.name == "CustomBalancer"
