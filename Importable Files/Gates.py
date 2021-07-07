import numpy as np
from scipy.linalg import expm, norm, dft
from openfermion.ops import BosonOperator, QuadOperator
from openfermion.transforms import get_sparse_operator
from strawberryfields.utils import random_interferometer

def cubic_phase(gamma, cutoff, offset=20):
    x3 = QuadOperator('q0 q0 q0')
    U = expm(get_sparse_operator(-1j*gamma*x3, trunc=cutoff+offset, hbar=2).toarray())
    return U[:cutoff, :cutoff]

def cross_kerr(kappa, cutoff):
    n0 = BosonOperator('0^ 0')
    n1 = BosonOperator('1^ 1')
    U = expm(get_sparse_operator(1j*kappa*n0*n1, trunc=cutoff).toarray())
    return U

def random_unitary(size, cutoff):
    U = np.identity(cutoff, dtype=np.complex128)
    U[:size, :size] = random_interferometer(size)
    return U

def DFT(size, cutoff):
    U = np.identity(cutoff, dtype=np.complex128)
    U[:size, :size] = dft(size)/np.sqrt(size)
    return U

def min_cutoff(U, p, gate_cutoff, cutoff):
    min_cutoff = cutoff + 1
    m = get_modes(U, cutoff)
    for n in range(cutoff, gate_cutoff, -1):
        norms = 1 - norm(U[:n**m, :gate_cutoff**m], axis=0)
        eps = max(norms)
        if eps > p:
            min_cutoff = n+1
            break
    else:
        min_cutoff = gate_cutoff + 1
    return min_cutoff

def get_modes(U, cutoff):
    return int(np.log(U.shape[0])/np.log(cutoff))

def unitary_state_fidelity(V, U, cutoff):
    c = cutoff
    m = get_modes(V, c)
    d = np.int(U.shape[1]**(1/m))
    if m == 1:
        state1 = np.sum(V[:, :d], axis=1)/np.sqrt(d)
        state2 = np.sum(U, axis=1)/np.sqrt(d)
    elif m == 2:
        Ut = V.reshape(c, c, c, c)[:, :, :d, :d].reshape(c**2, d**2)
        eq_sup_state = np.full([d**2], 1/d)
        state1 = Ut @ eq_sup_state
        state2 = U @ eq_sup_state
    fidelity = np.abs(np.vdot(state1, state2))**2
    return state1, state2, fidelity

def sample_average_fidelity(V, U, cutoff, samples=10000):
    c = cutoff
    m = get_modes(V, c)
    d = np.int(U.shape[1]**(1/m))
    if m == 1:
        Ut = V[:, :d]
    elif m == 2:
        Ut = V.reshape(c, c, c, c)[:, :, :d, :d].reshape(c**2, d**2)
    fid = []
    Wlist = []
    for i in range(samples):
        W = random_interferometer(d**m)
        Wlist.append(W)
        f = np.abs(W[:, 0].conj().T @ Ut.conj().T @ U @ W[:, 0])**2
        fid.append(f)
    return np.mean(fid)

def process_fidelity(V, U, cutoff):
    c = cutoff
    m = get_modes(V, c)
    d = np.int(U.shape[1]**(1/m))
    if m == 1:
        Ut = V[:d, :d]
        Ul = U[:d, :d]
    elif m == 2:
        Ut = V.reshape(c, c, c, c)[:d, :d, :d, :d].reshape(d**2, d**2)
        Ul = U.reshape(c, c, d, d)[:d, :d, :d, :d].reshape(d**2, d**2)
    I = np.identity(d**m)
    phi = I.flatten()/np.sqrt(d**m)
    psiV = np.kron(I, Ut) @ phi
    psiU = np.kron(I, Ul) @ phi
    return np.abs(np.vdot(psiV, psiU))**2

def average_fidelity(V, U, cutoff):
    c = cutoff
    m = get_modes(V, c)
    d = np.int(U.shape[1]**(1/m))
    Fe = process_fidelity(V, U, cutoff)
    return (Fe*d+1)/(d+1)