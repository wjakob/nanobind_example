import nanobind_example as m
import numpy as np
import scipy.sparse
import pytest

@pytest.mark.benchmark(group='randn')
def test_popcount_randn_perf(benchmark):
    out = np.zeros((32, 1024), 'f')
    benchmark(m.randn, 42, out.reshape(-1))

@pytest.mark.benchmark(group='randn')
def test_np_randn_perf(benchmark):
    benchmark(np.random.randn, 32, 1024)

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
    conn = m.Conn(num_node, spw[0].size)  #, mode=mode)
    conn.weights[:] = spw[0]
    conn.indices[:] = spw[1]
    conn.indptr[:] = spw[2]
    conn.idelays[:] = spw[3]

    assert cx.buf.shape == (num_node, horizon)
    # then we can test
    buf_val = np.r_[:1.0:1j*num_node *
                      horizon].reshape(num_node, horizon).astype('f')*4.0
    cx.buf[:] = buf_val
    np.testing.assert_equal(buf_val, cx.buf)
    cx.cx1[:] = cx.cx2[:] = 0.0

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

def test_conn_kernels():
    connj, cxj, cfun_np = base_setup()  # tvb_kernels.CxMode.CX_J)
    for t in range(1024):
        cx = cfun_np(t)
        m.cx_j(cxj, connj, t)
        np.testing.assert_allclose(cx[0], cxj.cx1, 1e-4, 1e-6)
        np.testing.assert_allclose(cx[1], cxj.cx2, 1e-4, 1e-6)

@pytest.mark.benchmark(group='conn_kernels')
def test_conn_cpp(benchmark):
    connj, cxj, cfun_np = base_setup()
    def run():
        for t in range(128):
            m.cx_j(cxj, connj, t)
    benchmark(run)

@pytest.mark.benchmark(group='conn_kernels')
def test_conn_numpy(benchmark):
    connj, cxj, cfun_np = base_setup()
    def run():
        for t in range(128):
            cx = cfun_np(t)
    benchmark(run)


def test_randn():
    batches = 128
    out = np.zeros((batches, 1024), 'f')
    m.randn(42, out.reshape(-1))
    np.testing.assert_allclose(out.mean(axis=1), 1/out.shape[1], 0.1, 0.2)
    np.testing.assert_allclose(out.std(axis=1), 1+1/out.shape[1], 0.1, 0.2)


def base_setup_simd():
    cv = 1.0
    dt = 0.1
    num_node = 90
    horizon = 256
    weights, lengths, spw_j = rand_weights(num_node=num_node, horizon=horizon, dt=dt, cv=cv)
    s_w = scipy.sparse.csr_matrix(weights)
    cx = m.Cx8(num_node, horizon)
    conn = m.Conn(num_node, s_w.data.size)  #, mode=mode)
    conn.weights[:] = s_w.data.astype(np.float32)
    conn.indices[:] = s_w.indices.astype(np.uint32)
    conn.indptr[:] = s_w.indptr.astype(np.uint32)
    conn.idelays[:] = (lengths[weights != 0]/cv/dt).astype(np.uint32)+2
    spw_i = (conn.weights.copy(), conn.indices.copy(), conn.indptr.copy(), conn.idelays.copy())

    assert cx.buf.shape == (num_node, horizon, 8)
    # then we can test
    buf_val = np.r_[:1.0:1j*num_node *
                      horizon * 8].reshape(num_node, horizon, 8).astype('f')*4.0
    cx.buf[:] = buf_val
    np.testing.assert_equal(buf_val, cx.buf)
    cx.cx1[:] = cx.cx2[:] = 0.0

    # impl simple numpy version
    def make_cfun_np():
        zero_mask = weights == 0
        csr_weights = scipy.sparse.csr_matrix(weights)
        idelays = (lengths[~zero_mask]/cv/dt).astype('i')+2
        idelays2 = -horizon + np.c_[idelays, idelays-1].T
        assert idelays2.shape == (2, csr_weights.nnz)

        def cfun_np(t):
            _ = cx.buf[csr_weights.indices, (t-idelays2) % horizon]
            _ *= csr_weights.data.reshape(-1, 1)
            _ = np.add.reduceat(_, csr_weights.indptr[:-1], axis=1)
            return _  # (2, num_node, width)
        return cfun_np

    return conn, cx, make_cfun_np()

def test_basic_simd():
    connpp, cxpp, cfun_np = base_setup_simd()
    for t in range(1024):
        cx = cfun_np(t)
        m.cx_j8(cxpp, connpp, t)
        np.testing.assert_allclose(cx[0], cxpp.cx1, 1e-4, 1e-6)
        np.testing.assert_allclose(cx[1], cxpp.cx2, 1e-4, 1e-6)

@pytest.mark.benchmark(group='conn_simd')
def test_conn_simd_cpp(benchmark):
    connpp, cxpp, cfun_np = base_setup_simd()
    def run():
        for t in range(128):
            m.cx_j8(cxpp, connpp, t)
    benchmark(run)

@pytest.mark.benchmark(group='conn_simd')
def test_conn_simd_numpy(benchmark):
    connpp, cxpp, cfun_np = base_setup_simd()
    def run():
        for t in range(128):
            cx = cfun_np(t)
    benchmark(run)

# now batches of simd
def base_setup_simd_batch():
    cv = 1.0
    dt = 0.1
    num_node = 90
    horizon = 256
    weights, lengths, spw_j = rand_weights(num_node=num_node, horizon=horizon, dt=dt, cv=cv)
    s_w = scipy.sparse.csr_matrix(weights)
    num_batch = 4
    cx = m.Cx8s(num_node, horizon, num_batch)
    conn = m.Conn(num_node, s_w.data.size)  #, mode=mode)
    conn.weights[:] = s_w.data.astype(np.float32)
    conn.indices[:] = s_w.indices.astype(np.uint32)
    conn.indptr[:] = s_w.indptr.astype(np.uint32)
    conn.idelays[:] = (lengths[weights != 0]/cv/dt).astype(np.uint32)+2

    assert cx.buf.shape == (num_batch, num_node, horizon, 8)
    # then we can test
    buf_val = np.r_[:1.0:1j*num_batch*num_node *
                      horizon * 8].reshape(num_batch, num_node, horizon, 8).astype('f')*4.0
    cx.buf[:] = buf_val
    np.testing.assert_equal(buf_val, cx.buf)
    cx.cx1[:] = cx.cx2[:] = 0.0

    # impl simple numpy version
    def make_cfun_np():
        zero_mask = weights == 0
        csr_weights = scipy.sparse.csr_matrix(weights)
        idelays = (lengths[~zero_mask]/cv/dt).astype('i')+2
        idelays2 = -horizon + np.c_[idelays, idelays-1].T
        assert idelays2.shape == (2, csr_weights.nnz)

        def cfun_np(t):
            _ = cx.buf[:, csr_weights.indices, (t-idelays2) % horizon]
            _ *= csr_weights.data.reshape(-1, 1)
            _ = np.add.reduceat(_, csr_weights.indptr[:-1], axis=2)
            assert _.shape == (num_batch, 2, num_node, 8)
            return _  # (2, num_node, width)
        return cfun_np

    return conn, cx, make_cfun_np(), locals()


def test_basic_simd_batch():
    connpp, cxpp, cfun_np, ls = base_setup_simd_batch()
    for t in range(1024):
        cx = cfun_np(t)
        m.cxs8_j(cxpp, connpp, t)
        np.testing.assert_allclose(cx[:,0], cxpp.cx1, 1e-4, 1e-6)
        np.testing.assert_allclose(cx[:,1], cxpp.cx2, 1e-4, 1e-6)

@pytest.mark.benchmark(group='conn_simd_batch')
def test_conn_simd_batch_cpp(benchmark):
    connpp, cxpp, cfun_np, ls = base_setup_simd_batch()
    def run():
        for t in range(128):
            m.cxs8_j(cxpp, connpp, t)
    benchmark(run)

@pytest.mark.benchmark(group='conn_simd_batch')
def test_conn_simd_batch_numpy(benchmark):
    connpp, cxpp, cfun_np, ls = base_setup_simd_batch()
    def run():
        for t in range(128):
            cx = cfun_np(t)
    benchmark(run)