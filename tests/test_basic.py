import tvbk as m
import numpy as np
import scipy.sparse
import pytest
import collections

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
    connj, cxj, cfun_np,  = base_setup()
    def run():
        for t in range(128):
            m.cx_j(cxj, connj, t)
    benchmark(run)

@pytest.mark.benchmark(group='conn_kernels')
def test_conn_numpy(benchmark):
    connj, cxj, cfun_np,  = base_setup()
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


# from vbjax
JRTheta = collections.namedtuple(
    typename='JRTheta',
    field_names='A B a b v0 nu_max r J a_1 a_2 a_3 a_4 mu I'.split(' '))

jr_default_theta = JRTheta(
    A=3.25, B=22.0, a=0.1, b=0.05, v0=5.52, nu_max=0.0025, 
    r=0.56, J=135.0, a_1=1.0, a_2=0.8, a_3=0.25, a_4=0.25, mu=0.22, I=0.0)
    
def dfun_jr_np(ys, cs, p):
    y0, y1, y2, y3, y4, y5 = ys
    c, = cs

    sigm_y1_y2 = 2.0 * p.nu_max / (1.0 + np.exp(p.r * (p.v0 - (y1 - y2))))
    sigm_y0_1  = 2.0 * p.nu_max / (1.0 + np.exp(p.r * (p.v0 - (p.a_1 * p.J * y0))))
    sigm_y0_3  = 2.0 * p.nu_max / (1.0 + np.exp(p.r * (p.v0 - (p.a_3 * p.J * y0))))

    return np.array([y3,
        y4,
        y5,
        p.A * p.a * sigm_y1_y2 - 2.0 * p.a * y3 - p.a ** 2 * y0,
        p.A * p.a * (p.mu + p.a_2 * p.J * sigm_y0_1 + c)
            - 2.0 * p.a * y4 - p.a ** 2 * y1,
        p.B * p.b * (p.a_4 * p.J * sigm_y0_3) - 2.0 * p.b * y5 - p.b ** 2 * y2,
                     ])

def test_jr8():
    for i in range(1024):
        dx = np.zeros((6, 8), 'f')
        x = np.random.randn(*dx.shape).astype('f')+10
        c = np.random.randn(1,8).astype('f') + 10
        p = np.tile(np.array(jr_default_theta).astype('f'), (8, 1)).T.copy()
        assert p.shape == (14, 8)
        m.dfun_jr8(dx, x, c, p)
        dx_np = dfun_jr_np(x, c, JRTheta(*p))
        np.testing.assert_allclose(dx, dx_np, 0.15, 0.1)


# Montbrio-Pazo-Roxin
MPRTheta = collections.namedtuple(
    typename='MPRTheta',
    field_names='tau I Delta J eta cr'.split(' '))

mpr_default_theta = MPRTheta(
    tau=1.0,
    I=0.0,
    Delta=1.0,
    J=15.0,
    eta=-5.0,
    cr=1.0
)

def mpr_dfun(ys, c, p):
    r, V = ys
    r = r * (r > 0)
    I_c = p.cr * c
    # with np.printoptions(precision=3):
    #     print(f'I_c: ', I_c)
    return np.array([
        (1 / p.tau) * (p.Delta / (np.pi * p.tau) + 2 * r * V),
        (1 / p.tau) * (V ** 2 + p.eta + p.J * p.tau *
         r + p.I + I_c - (np.pi ** 2) * (r ** 2) * (p.tau ** 2))
    ])

def test_mpr8():
    for i in range(1024):
        dx = np.zeros((2, 8), 'f')
        x = np.random.randn(*dx.shape).astype('f')/5+np.c_[0.5,-2.0].T
        c = np.random.randn(1,8).astype('f')/2
        p = np.tile(np.array(mpr_default_theta).astype('f'), (8, 1)).T.copy()
        assert p.shape == (6, 8)
        m.dfun_mpr8(dx, x, c, p)
        dx_np = mpr_dfun(x, c, MPRTheta(*p))
        np.testing.assert_allclose(dx, dx_np, 0.01, 0.01)

# ref impl for sim w/ numpy
def run_sim_np(dfun, num_svar, buf_init,
    csr_weights: scipy.sparse.csr_matrix,
    idelays: np.ndarray,
    sim_params: np.ndarray,
    horizon: int,
    z_scale: np.ndarray=None,
    rng_seed=43, num_item=8, num_node=90, num_time=1000, dt=0.1,
    num_skip=5
):
    trace_shape = num_time // num_skip + 1, num_svar, num_node, num_item
    trace = np.zeros(trace_shape, 'f')
    assert idelays.max() < horizon-2
    idelays2 = horizon - np.c_[idelays, idelays-1].T
    assert idelays2.shape == (2, csr_weights.nnz)
    buffer = np.zeros((num_node, horizon, num_item))
    buffer[:] = buf_init


    def cfun(t):
        cx = buffer[csr_weights.indices, (t+idelays2) % horizon]
        cx *= csr_weights.data.reshape(-1, 1)
        cx = np.add.reduceat(cx, csr_weights.indptr[:-1], axis=1)
        # with np.printoptions(precision=3):
        #     print(f'cx1(t={t}):', cx[0,4])
        return cx  # (2, num_node, num_item)

    def heun(x, cx):
        z = np.random.randn(2, num_node, num_item)*z_scale.reshape((2, 1, 1)) if z_scale is not None else 0
        dx1 = dfun(x, cx[0], sim_params)
        xi = x + dt*dx1 + z
        xi[0] = xi[0]*(xi[0] > 0)
        dx2 = dfun(xi, cx[1], sim_params)
        nx = x + dt/2*(dx1 + dx2) + z
        nx[0] = nx[0]*(nx[0] > 0)
        return nx

    x = np.zeros((num_svar, num_node, num_item), 'f')

    for t in range(trace.shape[0]):
        for tt in range(num_skip):
            ttt = t*num_skip + tt
            cx = cfun(ttt)
            print()
            x = heun(x, cx)
            buffer[:, ttt % horizon] = x[0]
        trace[t] = x

    return trace


def test_step_mpr():
    cv = 1.0
    dt = 0.01
    num_node = 8
    num_skip = 1
    num_time = 5
    horizon = 256
    num_batch = 1
    sparsity = 0.9 # nnz=0.5*num_node**2

    weights, lengths, spw_j = rand_weights(
        seed=46,
        sparsity=sparsity, num_node=num_node, horizon=horizon,
        dt=dt, cv=cv)
    weights = np.ones_like(weights)
    s_w = scipy.sparse.csr_matrix(weights)
    idelays = (lengths[weights != 0]/cv/dt).astype(np.uint32)+2
    idelays = idelays//25 + 2
    assert idelays.max() < horizon
    assert idelays.min() >= 2
    cx = m.Cx8s(num_node, horizon, num_batch)
    conn = m.Conn(num_node, s_w.data.size)  #, mode=mode)
    conn.weights[:] = s_w.data.astype(np.float32)
    conn.indptr[:] = s_w.indptr.astype(np.uint32)
    conn.indices[:] = s_w.indices.astype(np.uint32)
    conn.idelays[:] = idelays

    assert cx.buf.shape == (num_batch, num_node, horizon, 8)
    # then we can test
    buf_val = np.r_[:1.0:1j*num_batch*num_node *
                      horizon * 8].reshape(num_batch, num_node, horizon, 8).astype('f')*4.0
    cx.buf[:] = buf_val
    np.testing.assert_equal(buf_val, cx.buf)
    cx.cx1[:] = cx.cx2[:] = 0.0

    buf_init_np = buf_val.transpose(1,2,0,3).reshape(num_node, horizon, num_batch*8)

    # cx, c, x, p, t0, nt, dt
    num_svar, num_parm = 2, 6
    x = np.zeros((num_batch, num_svar, num_node, 8), 'f')
    p = np.zeros((num_batch, num_node, num_parm, 8), 'f')
    p[:] = p + np.array(mpr_default_theta
                        ).reshape(1,1,-1,1) + np.random.randn(*p.shape)*0.1
    # set uniform I & cr
    p[...,1,:] += 2.0
    p[...,5,:] /= 10.0
    
    trace_np = run_sim_np(
        mpr_dfun, num_svar, buf_init_np, s_w, idelays,
        MPRTheta(*p.transpose(2,1,0,3).reshape(num_parm,num_node,num_batch*8)),
        # mpr_default_theta,
        horizon, num_item=num_batch*8, num_node=num_node, num_time=num_time,
        dt=dt, num_skip=num_skip
    )
    trace_np = trace_np.reshape(-1, num_svar, num_node, num_batch, 8).transpose( 0, 3, 1, 2, 4 )
    print()
    trace_c = np.zeros_like(trace_np) # (num_time//num_skip, num_batch, num_svar, num_node, 8)
    for t0 in range(trace_c.shape[0]):
        m.step_mpr8(cx, conn, x, p, t0*num_skip, num_skip, dt)
        trace_c[t0] = x
        # for i in range(num_svar):
        #     a, b = trace_c[t0, 0, i], trace_np[t0, 0, i]
        #     for j in range(num_node):
        #         np.testing.assert_allclose(a[j], b[j], 0.01, 0.01)                                                                                                                                                                                                                                                   
    # np.testing.assert_allclose(trace_c, trace_np, 0.01, 0.01)

    import pylab as pl
    # (num_time//num_skip, num_batch, num_svar, num_node, 8)
    
    ae = 0.0
    iae = -1
    for i in range(8):
        for j in range(8):
            s = 1  # V
            pl.subplot(8,8,i*8+j+1);
            _np = trace_np[:,0,s,i,j]
            _c = trace_c[:,0,s,i,j]
            if True:
                _np, _c = np.diff(_np, axis=0), np.diff(_c, axis=0)
            pl.plot(_np, 'r-', alpha=0.4)
            pl.plot(_c, 'k-', alpha=0.4)
            pl.xticks([]); pl.yticks([])
            _ae = np.abs(_np-_c).sum()
            if _ae > ae:
                ae, iae = _ae, (i,j)
    pl.tight_layout()
    pl.savefig('test_mpr.jpg')
    
    pl.figure()
    s, (i,j) = 1, iae
    _np = trace_np[:,0,s,i,j]
    _c = trace_c[:,0,s,i,j]
    pl.subplot(211)
    pl.plot(_np, 'r-', alpha=0.4)
    pl.plot(_c, 'k-', alpha=0.4)
    pl.subplot(212)
    _np, _c = np.diff(_np, axis=0), np.diff(_c, axis=0)
    pl.plot(_np, 'r-', alpha=0.4)
    pl.plot(_c, 'k-', alpha=0.4)
    pl.savefig('test_mpr2.jpg')
