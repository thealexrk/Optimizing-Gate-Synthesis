import numpy as np
from scipy.special import factorial as fac

def single_photon(cutoff):
    state = np.zeros([cutoff])
    state[1] = 1
    return state

def ON(N, a, cutoff):
    state = np.zeros([cutoff])
    state[0] = 1
    state[N] = a
    return state/np.linalg.norm(state)

def hex_GKP(mu, d, delta, cutoff, nmax=7):
    n1 = np.arange(-nmax, nmax+1)[:, None]
    n2 = np.arange(-nmax, nmax+1)[None, :]
    n1sq = n1**2
    n2sq = n2**2
    sqrt3 = np.sqrt(3)
    arg1 = -1j*np.pi*n2*(d*n1+mu)/d
    arg2 = -np.pi*(d**2*n1sq+n2sq-d*n1*(n2-2*mu)-n2*mu+mu**2)/(sqrt3*d)
    arg2 *= 1-np.exp(-2*delta**2)
    amplitude = (np.exp(arg1)*np.exp(arg2)).flatten()[:, None]
    alpha = np.sqrt(np.pi/(2*sqrt3*d)) * (sqrt3*(d*n1+mu) - 1j*(d*n1-2*n2+mu))
    alpha *= np.exp(-delta**2)
    alpha = alpha.flatten()[:, None]
    n = np.arange(cutoff)[None, :]
    coherent = np.exp(-0.5*np.abs(alpha)**2)*alpha**n/np.sqrt(fac(n))
    hex_state = np.sum(amplitude*coherent, axis=0)
    return hex_state/np.linalg.norm(hex_state)

def random_state(cutoff):
    state = np.random.randn(cutoff) + 1j*np.random.randn(cutoff)
    return state/np.linalg.norm(state)

def NOON(N, cutoff):
    state = np.zeros([cutoff, cutoff])
    state[0, N] = 1
    state[N, 0] = 1
    return state.flatten()/np.linalg.norm(state)

def correct_global_phase(state):
    maxentry = np.argmax(np.abs(state))
    phase = state[maxentry]/np.abs(state[maxentry])
    return state/phase