import nanobind_example as m
import numpy as np
import scipy.sparse


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
    out = np.zeros((batches, 1024), 'f')
    m.randnpc(42, out.reshape(-1))
    np.testing.assert_allclose(out.mean(axis=1), 1/out.shape[1], 0.1, 0.2)
    np.testing.assert_allclose(out.std(axis=1), 1+1/out.shape[1], 0.1, 0.2)

# def test_randn_perf(benchmark):
#     out = np.zeros((32, 1024), 'f')
#     benchmark(m.randn, 42, out.reshape(-1))

def test_randnpc_perf(benchmark):
    out = np.zeros((32, 1024), 'f')
    benchmark(m.randnpc, 42, out.reshape(-1))

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



def rand_weights(seed=43, sparsity=0.4, cv=1.0, dt=0.1, horizon=256, num_node=90):
    np.random.seed(seed)
    horizonm1 = horizon - 1
    assert ((horizon) & (horizonm1)) == 0
    weights, lengths = np.random.rand(2, num_node, num_node).astype('f')
    lengths[:] *= 0.8
    lengths *= (horizon*dt*0.8)
    zero_mask = weights < (1-sparsity)
    weights[zero_mask] = 0

    # prep cx_j mode
    import scipy.sparse
    # cx_j requires csc format, cx_i requires csr
    spw = scipy.sparse.csc_matrix(weights)
    # since numpy is row major, we need to transpose zero mask then ravel
    nnz_ravel = np.argwhere(~zero_mask.T.copy().ravel())[:, 0]
    # then we also need to extract them in the transposed order
    idelays = (lengths.T.copy().ravel()[
        nnz_ravel] / cv / dt).astype('i') + 2
    return weights, lengths, (
        spw.data.astype(np.float32), spw.indices.astype(np.uint32), spw.indptr.astype(np.uint32), idelays)


def base_setup():  # mode: tvb_kernels.CxMode = tvb_kernels.CxMode.CX_J):
    cv = 1.0
    dt = 0.1
    num_node = 90
    horizon = 256
    weights, lengths, spw = rand_weights(num_node=num_node, horizon=horizon, dt=dt, cv=cv)
    cx = m.Cx(num_node, horizon)
    conn = m.Conn(num_node, spw[0].size, *spw)  #, mode=mode)

    assert cx.buf.shape == (num_node, horizon)
    # then we can test
    cx.buf[:] = np.r_[:1.0:1j*num_node *
                      horizon].reshape(num_node, horizon).astype('f')*4.0

    # impl simple numpy version
    def make_cfun_np():
        zero_mask = weights == 0
        csr_weights = scipy.sparse.csr_matrix(weights)
        idelays = (lengths[~zero_mask]/cv/dt).astype('i')+2
        idelays2 = -horizon + np.c_[idelays, idelays-1].T
        assert idelays2.shape == (2, csr_weights.nnz)

        def cfun_np(t):
            _ = cx.buf[csr_weights.indices, (t-idelays2) % horizon]
            _ *= csr_weights.data
            _ = np.add.reduceat(_, csr_weights.indptr[:-1], axis=1)
            return _  # (2, num_node)
        return cfun_np

    return conn, cx, make_cfun_np()


# def test_conn_kernels():
#     connj, cxj, cfun_np = base_setup()  # tvb_kernels.CxMode.CX_J)
#     # conni, cxi, _ = base_setup(tvb_kernels.CxMode.CX_I)
#     for t in range(1024):
#         cx = cfun_np(t)
#         m.cx_j(cxj, connj, t)
#         # conni(t)
#         np.testing.assert_allclose(cx[0], cxj.cx1, 1e-4, 1e-6)
#         np.testing.assert_allclose(cx[1], cxj.cx2, 1e-4, 1e-6)
#         # np.testing.assert_allclose(cx[0], cxi.cx1, 1e-4, 1e-6)
#         # np.testing.assert_allclose(cx[1], cxi.cx2, 1e-4, 1e-6)
