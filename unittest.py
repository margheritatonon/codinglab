#all tests need to start with the word test

import pytest

def division(x, y):
    return x / y

def test_division():
    assert division(1, 2) == 0.5
    print("Great!")

def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        division(1, 0)

@pytest.mark.parametrize(
    "a, b, expected",
    [(10, 2, 5), (9, 3, 3)]
)

def test_divide(a, b, expected):
    assert division(a, b) == expected
