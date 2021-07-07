import numpy as np
from numpy.polynomial.hermite import hermval, hermval2d
from scipy.special import factorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Computer Modern Roman']

def plot_cost(cost_progress, color='#3f9b0b'):
    fig, ax = plt.subplots(1, 1)
    ax.plot(cost_progress, color=color, linewidth=2.0)
    ax.set_xlabel('Steps', fontsize=22)
    ax.set_ylabel('Cost function', fontsize=22)
    ax.xaxis.set_tick_params(which='major', labelsize=20)
    ax.yaxis.set_tick_params(which='major', labelsize=20)
    fig.tight_layout()
    return fig, ax

def wigner(rho, xvec, pvec):
    import copy
    Q, P = np.meshgrid(xvec, pvec)
    cutoff = rho.shape[0]
    A = (Q + P * 1.0j) / (2 * np.sqrt(2 / 2))
    Wlist = np.array([np.zeros(np.shape(A), dtype=complex) for k in range(cutoff)])
    Wlist[0] = np.exp(-2.0 * np.abs(A) ** 2) / np.pi
    W = np.real(rho[0, 0]) * np.real(Wlist[0])
    for n in range(1, cutoff):
        Wlist[n] = (2.0 * A * Wlist[n - 1]) / np.sqrt(n)
        W += 2 * np.real(rho[0, n] * Wlist[n])
    for m in range(1, cutoff):
        temp = copy.copy(Wlist[m])
        Wlist[m] = (2 * np.conj(A) * temp - np.sqrt(m)
                    * Wlist[m - 1]) / np.sqrt(m)
        W += np.real(rho[m, m] * Wlist[m])
        for n in range(m + 1, cutoff):
            temp2 = (2 * A * Wlist[n - 1] - np.sqrt(m) * temp) / np.sqrt(n)
            temp = copy.copy(Wlist[n])
            Wlist[n] = temp2
            W += 2 * np.real(rho[m, n] * Wlist[n])
    return Q, P, W / 2

def wigner_contour_plot(ket, l=5., cmap="RdYlGn"):
    x = np.linspace(-l, l, 100)
    p = np.linspace(-l, l, 100)
    rho = np.outer(ket, np.conj(ket))
    X, P, W = wigner(rho, x, p)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(X, P, W, 10, cmap=cmap, linestyles="solid")
    ax.set_axis_off()
    return fig, ax

def wigner_3D_plot(ket, offset=-0.12, l=5., cmap="RdYlGn", vmin=None, vmax=None):
    x = np.linspace(-l, l, 100)
    p = np.linspace(-l, l, 100)
    rho = np.outer(ket, np.conj(ket))
    X, P, W = wigner(rho, x, p)
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, P, W, cmap=cmap, lw=0.5, rstride=1, cstride=1, vmin=vmin, vmax=vmax)
    ax.contour(X, P, W, 10, cmap=cmap, linestyles="solid", offset=offset)
    ax.set_axis_off()
    return fig, ax

def wavefunction(ket, l=4.5, N=10000):
    c = ket.shape[0]
    n = np.arange(c)
    coefficients = np.real(ket) / (np.sqrt(factorial(n) * (2 ** n)))
    x = np.linspace(-l, l, N)
    y = np.exp(-x ** 2 / 2) * hermval(x, coefficients)
    return x, y

def wavefunction_plot(ket, color='#3f9b0b', l=4, N=10000):
    x, y = wavefunction(ket, l=l, N=N)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, color=color)
    ax.set_xlabel(r'$x$', fontsize=22)
    ax.set_ylabel(r'$\psi(x)$', fontsize=22)
    ax.xaxis.set_tick_params(which='major', labelsize=20)
    ax.yaxis.set_tick_params(which='major', labelsize=20)
    fig.tight_layout()
    return fig, ax

def two_mode_wavefunction(ket, l=4.5, N=100):
    c = int(np.sqrt(ket.shape[0]))
    output_state = ket.reshape(c, c)
    n = np.arange(c)[:, None]
    m = np.arange(c)[None, :]
    coefficients = np.real(output_state) / (
            (np.sqrt(factorial(n) * (2 ** n))) * (np.sqrt(factorial(m) * (2 ** m))))
    x = np.linspace(-l, l, N)
    y = np.linspace(-l, l, N)
    X, Y = np.meshgrid(x, y)
    Z = (np.exp(-X ** 2 / 2))*(np.exp(-Y ** 2 / 2))*hermval2d(X, Y, coefficients)
    return X, Y, Z

def two_mode_wavefunction_plot(ket, cmap="RdYlBu", offset=-0.11, l=4.5, N=100):
    X, Y, Z = two_mode_wavefunction(ket, l=l, N=N)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cmap, lw=0.5, rstride=1, cstride=1)
    ax.contour(X, Y, Z, 10, cmap=cmap, linestyles="solid", offset=offset)
    ax.set_axis_off()
    return fig, ax

def one_mode_unitary_plots(target_unitary, learnt_unitary, square=False):
    c = learnt_unitary.shape[0]
    d = learnt_unitary.shape[1]
    if square:
        Ut = target_unitary[:d, :d]
        Ur = learnt_unitary[:d, :d]
    else:
        Ut = target_unitary[:c, :d]
        Ur = learnt_unitary[:c, :d]
    vmax = np.max([Ut.real, Ut.imag, Ur.real, Ur.imag])
    vmin = np.min([Ut.real, Ut.imag, Ur.real, Ur.imag])
    cmax = max(vmax, vmin)
    fig, ax = plt.subplots(1, 4, figsize=(7, 4))
    ax[0].matshow(Ut.real, cmap=plt.get_cmap('Reds'), vmin=-cmax, vmax=cmax)
    ax[1].matshow(Ut.imag, cmap=plt.get_cmap('Greens'), vmin=-cmax, vmax=cmax)
    ax[2].matshow(Ur.real, cmap=plt.get_cmap('Reds'), vmin=-cmax, vmax=cmax)
    ax[3].matshow(Ur.imag, cmap=plt.get_cmap('Greens'), vmin=-cmax, vmax=cmax)
    for a in ax.ravel():
        a.tick_params(bottom=False,labelbottom=False,
                      top=False,labeltop=False,
                      left=False,labelleft=False,
                      right=False,labelright=False)
    ax[0].set_xlabel(r'$\mathrm{Re}(V)$')
    ax[1].set_xlabel(r'$\mathrm{Im}(V)$')
    ax[2].set_xlabel(r'$\mathrm{Re}(U)$')
    ax[3].set_xlabel(r'$\mathrm{Im}(U)$')
    for a in ax.ravel():
        a.tick_params(color='white', labelcolor='white')
        for spine in a.spines.values():
            spine.set_edgecolor('white')
    fig.tight_layout()
    return fig, ax

def two_mode_unitary_plots(target_unitary, learnt_unitary, square=False):
    c = int(np.sqrt(learnt_unitary.shape[0]))
    d = int(np.sqrt(learnt_unitary.shape[1]))
    if square:
        Ut = target_unitary.reshape(c, c, c, c)[:d, :d, :d, :d].reshape(d**2, d**2)
        Ur = learnt_unitary.reshape(c, c, d, d)[:d, :d, :d, :d].reshape(d**2, d**2)
    else:
        Ut = target_unitary.reshape(c, c, c, c)[:, :, :d, :d].reshape(c**2, d**2)
        Ur = learnt_unitary
    vmax = np.max([Ut.real, Ut.imag, Ur.real, Ur.imag])
    vmin = np.min([Ut.real, Ut.imag, Ur.real, Ur.imag])
    cmax = max(vmax, vmin)
    fig, ax = plt.subplots(1, 4)
    ax[0].matshow(Ut.real, cmap=plt.get_cmap('Reds'), vmin=-cmax, vmax=cmax)
    ax[1].matshow(Ut.imag, cmap=plt.get_cmap('Greens'), vmin=-cmax, vmax=cmax)
    ax[2].matshow(Ur.real, cmap=plt.get_cmap('Reds'), vmin=-cmax, vmax=cmax)
    ax[3].matshow(Ur.imag, cmap=plt.get_cmap('Greens'), vmin=-cmax, vmax=cmax)
    for a in ax.ravel():
        a.tick_params(bottom=False,labelbottom=False,
                      top=False,labeltop=False,
                      left=False,labelleft=False,
                      right=False,labelright=False)
    ax[0].set_ylabel(r'$\mathrm{Re}(V)$')
    ax[1].set_ylabel(r'$\mathrm{Im}(V)$')
    ax[2].set_xlabel(r'$\mathrm{Re}(U)$')
    ax[3].set_xlabel(r'$\mathrm{Im}(U)$')
    for a in ax.ravel():
        a.tick_params(color='white', labelcolor='white')
        for spine in a.spines.values():
            spine.set_edgecolor('white')
    fig.tight_layout()
    return fig, ax