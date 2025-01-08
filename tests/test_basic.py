import nanobind_example as m
import numpy as np


def test_add():
    assert m.add(1, 2) == 3
    assert m.add(1, 3) == 4

def test_mul():
    assert m.mul(2,3) == 6

def test_partial_add():
    add_2 = m.partial_add(2)
    assert add_2(3) == 5
    assert add_2(4) == 6

def test_randn():
    batches = 128
    out = np.zeros((batches, 1024), 'f')
    m.randn(42, out.reshape(-1))
    np.testing.assert_allclose(out.mean(axis=1), 1/out.shape[1], 0.1, 0.2)
    np.testing.assert_allclose(out.std(axis=1), 1+1/out.shape[1], 0.1, 0.2)

def test_randn_perf(benchmark):
    out = np.zeros((32, 1024), 'f')
    benchmark(m.randn, 42, out.reshape(-1))

def test_np_randn_perf(benchmark):
    benchmark(np.random.randn, 32, 1024)

def test_mm():
    A, B, C = np.random.randn(3, 8, 8).astype('f')
    C = (A @ B).astype('f')
    C_ref = C*0
    C_fast = C*0
    m.mm8_fast(A, B, C_fast)
    m.mm8_ref(A, B, C_ref)
    np.testing.assert_allclose(C_ref, C, 1e-6, 1e-6)
    np.testing.assert_allclose(C_fast, C, 1e-6, 1e-6)