import pytest
from sklearn.metrics import accuracy_score

def test_accuracy_score_perfect():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]
    assert accuracy_score(y_true, y_pred) == 1.0

def test_accuracy_score_incorrect():
    y_true = [1, 0, 1, 0]
    y_pred = [0, 1, 0, 1]
    assert accuracy_score(y_true, y_pred) == 0.0

