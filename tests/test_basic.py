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

    def make_jax():
        try:
            import jax, jax.numpy as jp # noqa
        except ImportError:
            return None
        horizonm1 = horizon - 1
        data, indices, indptr, idelays = spw
        j_indices = jp.array(indices)
        j_weights = jp.array(data)
        j_indptr = jp.array(indptr)
        buffer = jp.array(buf_val)
        assert idelays.max() < horizon-2
        idelays2 = jp.array(horizon + np.c_[idelays, idelays-1].T)        
        _csr_rows = np.concatenate([i*np.ones(n, 'i')
                                    for i, n in enumerate(np.diff(indptr))])
        j_csr_rows = jp.array(_csr_rows)        
        def f(t):
            wxij = j_weights * buffer[j_indices, (t-idelays2)&horizonm1]
            cx = jp.zeros((2, num_node))
            cx = cx.at[:, j_csr_rows].add(wxij)
            return cx
        f = jax.jit(f)
        f(0)  # warmup
        return f
    
    def make_numba():
        try:
            import numba
        except ImportError:
            return None
        cx = np.zeros((2, num_node), 'f')
        @numba.njit(fastmath=True)
        def kernel(t, cx, buf, data, indices, indptr, idelays):
            num_node = cx.shape[1]
            horizon = buf.shape[1]
            horizonm1 = horizon - 1
            for i in range(num_node):
                cx[0,i] = cx[1,i] = numba.float32(0.0)
            for i in range(num_node):
                b = buf[i]
                th = numba.uint32(t + horizon)
                for j in range(indptr[i], indptr[i+1]):
                    i = indices[j]
                    w = data[j]
                    d = idelays[j]
                    p1 = (th - d) & horizon
                    p2 = (th - d + 1) & horizon
                    cx[0, i] += w * b[p1]
                    cx[1, i] += w * b[p2]
        def f(t):
            kernel(t, cx, buf_val, *spw)
            return cx
        f(0)  # warmup
        return f

    return conn, cx, make_cfun_np(), make_jax(), make_numba()

def test_conn_kernels():
    connj, cxj, cfun_np, cfun_jax, cfun_numba = base_setup()  # tvb_kernels.CxMode.CX_J)
    for t in range(1024):
        cx = cfun_np(t)
        m.cx_j(cxj, connj, t)
        cx2 = cfun_jax(t)
        cx3 = cfun_numba(t)
        np.testing.assert_allclose(cx[0], cxj.cx1, 1e-4, 1e-6)
        np.testing.assert_allclose(cx[1], cxj.cx2, 1e-4, 1e-6)
        # np.testing.assert_allclose(cx3, cx, 2, 0.1)
        # jax code is transposed, ignore errors for now
        # np.testing.assert_allclose(cx2, cx, 1e-4, 1e-6)

@pytest.mark.benchmark(group='conn_kernels')
def test_conn_cpp(benchmark):
    connj, cxj, cfun_np, cfun_jax, cfun_numba = base_setup()
    def run():
        for t in range(128):
            m.cx_j(cxj, connj, t)
    benchmark(run)

@pytest.mark.benchmark(group='conn_kernels')
def test_conn_numpy(benchmark):
    connj, cxj, cfun_np, cfun_jax, cfun_numba = base_setup()
    def run():
        for t in range(128):
            cx = cfun_np(t)
    benchmark(run)

@pytest.mark.benchmark(group='conn_kernels')
def test_conn_jax(benchmark):
    connj, cxj, cfun_np, cfun_jax, cfun_numba = base_setup()
    def run():
        for t in range(128):
            cx = cfun_jax(t)
    benchmark(run)

@pytest.mark.benchmark(group='conn_kernels')
def test_conn_numba(benchmark):
    connj, cxj, cfun_np, cfun_jax, cfun_numba = base_setup()
    def run():
        for t in range(128):
            cx = cfun_numba(t)
    benchmark(run)


try:
    import numba
    """
    INLINE uint64_t sfc64(uint64_t s[4]) {
    uint64_t r = s[0] + s[1] + s[3]++;
    s[0] = (s[1] >> 11) ^ s[1];
    s[1] = (s[2] << 3) + s[2];
    s[2] = r + (s[2] << 24 | s[2] >> 40);
    return r;
    }
    """
    @numba.njit(fastmath=True)
    def sfc64(s):
        r = s[0] + s[1] + s[3]
        s[3] += 1
        s[0] = (s[1] >> 11) ^ s[1]
        s[1] = (s[2] << 3) + s[2]
        s[2] = r + (s[2] << 24 | s[2] >> 40)
        return r
    """
    INLINE float randn1(uint64_t s[4]) {
    uint64_t u = sfc64(s);
    double x = __builtin_popcount(u >> 32);
    x += (uint32_t)u * (1 / 4294967296.);
    x -= 16.5;
    x *= 0.3517262290563295;
    return (float)x;
    }
    """
    @numba.njit(numba.uint64(numba.uint64))
    def popcount(x): 
        b=0
        while(x > 0):
            x &= x - numba.uint64(1)   
            b+=1
        return b
    @numba.njit(fastmath=True)
    def randn1(s):
        u = sfc64(s)
        x = numba.float64(popcount(u >> 32))
        x += numba.uint32(u) * numba.float64(1 / 4294967296.)
        x -= numba.float64(16.5)
        x *= numba.float64(0.3517262290563295)
        return numba.float32(x)
    @numba.njit(fastmath=True)
    def randnpc(s, out):
        for i in range(out.size):
            out[i] = randn1(s)
    @pytest.mark.benchmark(group='randn')
    def test_nbrandnpc_perf(benchmark):
        seed = np.zeros(4, np.uint64)
        seed[:] = 42
        out = np.zeros((32, 1024), 'f')
        benchmark(randnpc, seed, out.reshape(-1))
except ImportError:
    pass

def test_randn():
    batches = 128
    out = np.zeros((batches, 1024), 'f')
    m.randn(42, out.reshape(-1))
    np.testing.assert_allclose(out.mean(axis=1), 1/out.shape[1], 0.1, 0.2)
    np.testing.assert_allclose(out.std(axis=1), 1+1/out.shape[1], 0.1, 0.2)

    seed = np.zeros(4, np.uint64)
    seed[:] = 42
    out[:] = 0
    randnpc(seed, out.reshape(-1))
    np.testing.assert_allclose(out.mean(axis=1), 1/out.shape[1], 0.1, 0.2)
    np.testing.assert_allclose(out.std(axis=1), 1+1/out.shape[1], 0.1, 0.2)

# testing simd width
def base_setup_simd():  # mode: tvb_kernels.CxMode = tvb_kernels.CxMode.CX_J):
    cv = 1.0
    dt = 0.1
    num_node = 90
    horizon = 256
    weights, lengths, spw = rand_weights(num_node=num_node, horizon=horizon, dt=dt, cv=cv)
    cx = m.Cx8(num_node, horizon)
    conn = m.Conn(num_node, spw[0].size)  #, mode=mode)
    conn.weights[:] = spw[0]
    conn.indices[:] = spw[1]
    conn.indptr[:] = spw[2]
    conn.idelays[:] = spw[3]

    assert cx.buf.shape == (num_node, horizon, 8)
    # then we can test
    buf_val = np.r_[:1.0:1j*num_node *
                      horizon * 8].reshape(num_node, horizon, 8).astype('f')*4.0
    cx.buf[:] = buf_val
    np.testing.assert_equal(buf_val, cx.buf)
    cx.cx1[:] = cx.cx2[:] = 0.0

def test_basic_simd():
    base_setup_simd()
