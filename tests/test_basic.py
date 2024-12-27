import nanobind_example as m

def test_add():
    assert m.add(1, 2) == 3
    assert m.add(1, 3) == 4

def test_mul():
    assert m.mul(2,3) == 6
