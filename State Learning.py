import os
import time
import argparse
import json
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
from learner.circuits import variational_quantum_circuit
from learner.states import single_photon, ON, hex_GKP, random_state, NOON, correct_global_phase
from learner.plots import wigner_3D_plot, wavefunction_plot, two_mode_wavefunction_plot, plot_cost
import matplotlib.pyplot as plt

def cat_state(a, p, cutoff):
    phi = np.pi*p
    temp = np.exp(-0.5 * np.abs(a)**2)
    N = temp / np.sqrt(2*(1 + np.cos(phi) * temp**4))
    k = np.arange(cutoff)
    c1 = (a**k) / np.sqrt(fac(k))
    c2 = ((-a)**k) / np.sqrt(fac(k))
    ket = (c1 + np.exp(1j*phi) * c2) * N
    return ket

HP = {
    'name': 'cat_gif5',
    'out_dir': 'sim_results',
    'target_state_fn': cat_state,
    'state_params': {'a':1.5, 'p':0},
    'cutoff': 15,
    'depth': 25,
    'reps': 2000,
    'penalty_strength': 0,
    'active_sd': 0.1,
    'passive_sd': 0.1
}

def parse_arguments(defaults):
    parser = argparse.ArgumentParser(description='Quantum state preparation learning.')
    parser.add_argument('-n', '--name',
        type=str, default=defaults["name"], help='Simulation name.')
    parser.add_argument('-o', '--out-dir',
        type=str, default=defaults["out_dir"], help='Output directory')
    parser.add_argument('-s', '--dump-reps',
        type=int, default=100, help='Steps at which to save output')
    parser.add_argument('-D', '--debug',
        action='store_true', help="Debug mode")
    parser.add_argument('-r', '--reps',
        type=int, default=defaults["reps"], help='Optimization steps')
    parser.add_argument('-p', '--state-params',
        type=json.loads, default=defaults["state_params"], help='State parameters')
    parser.add_argument('-c', '--cutoff',
        type=int, default=defaults["cutoff"], help='Fock basis truncation')
    parser.add_argument('-d', '--depth',
        type=int, default=defaults["depth"], help='Number of layers')
    parser.add_argument('-P', '--penalty-strength',
        type=int, default=defaults["penalty_strength"], help='Regularisation penalty strength')
    args = parser.parse_args()
    hyperparams = {}
    hyperparams.update(defaults)
    hyperparams.update(vars(args))
    if args.debug:
        hyperparams['depth'] = 1
        hyperparams['reps'] = 5
        hyperparams['name'] += "_debug"
    hyperparams['ID'] = "{}_d{}_c{}_r{}".format(
        hyperparams['name'], hyperparams['depth'], hyperparams['cutoff'], hyperparams['reps'])
    hyperparams['out_dir'] = os.path.join(args.out_dir, hyperparams['ID'], '')
    hyperparams['board_name'] = os.path.join('TensorBoard', hyperparams['ID'], '')
    if not os.path.exists(hyperparams['out_dir']):
        os.makedirs(hyperparams['out_dir'])
    return hyperparams

def state_fidelity(ket, target_state):
    """Calculate the fidelity between the target and output state."""
    fidelity = tf.abs(tf.reduce_sum(tf.conj(ket) * target_state)) ** 2
    return fidelity

def optimize(ket, target_state, parameters, cutoff, reps=1000, penalty_strength=0,
        out_dir='sim_results', ID='state_learning', board_name='TensorBoard',
        dump_reps=100, **kwargs):
    fidelity = state_fidelity(ket, target_state)
    tf.summary.scalar('fidelity', fidelity)
    loss = 1-fidelity #tf.abs(tf.reduce_sum(tf.conj(ket) * target_state) - 1)
    tf.summary.scalar('loss', loss)
    state_norm = tf.abs(tf.reduce_sum(tf.conj(ket) * ket)) ** 2
    tf.summary.scalar('norm', state_norm)
    penalty = penalty_strength * (state_norm - 1)**2
    tf.summary.scalar('penalty', penalty)
    cost = loss + penalty
    tf.summary.scalar('cost', cost)
    optimiser = tf.train.AdamOptimizer()
    minimize_cost = optimiser.minimize(cost)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(board_name)
    merge = tf.summary.merge_all()
    fid_progress = []
    cost_progress = []
    best_state = np.zeros(cutoff)
    best_fid = 0
    start = time.time()
    for i in range(reps):
        _, cost_val, fid_val, ket_val, norm_val, penalty_val, params_val = session.run(
            [minimize_cost, cost, fidelity, ket, state_norm, penalty, parameters])
        cost_progress.append(cost_val)
        fid_progress.append(fid_val)
        fig2, ax2 = wigner_3D_plot(ket_val, offset=-0.155, l=5)
        ax2.set_zlim3d(-0.2, None)
        fig2.savefig(os.path.join(out_dir, '{}.png'.format(i).zfill(4)))
        plt.close(fig2)
        if i % dump_reps == 0:
            print("Rep: {} Cost: {:.4f} Fidelity: {:.4f} Norm: {:.4f}".format(
                i, cost_val, fid_val, norm_val))
            if i > 0:
                np.savez(os.path.join(out_dir, ID+'.npz'),
                    **sim_results)
        if i > 0 and fid_val > best_fid:
            best_fid = fid_val
            min_cost = cost_val
            best_state = correct_global_phase(ket_val)
            end = time.time()
            sim_results = {
                'name': HP['name'],
                'target_state': target_state,
                'state_params': HP['state_params'],
                'cutoff': cutoff,
                'depth': HP['depth'],
                'reps': reps,
                'penalty_strength': penalty_strength,
                'best_runtime': end-start,
                'best_rep': i,
                'min_cost': cost_val,
                'fidelity': best_fid,
                'cost_progress': np.array(cost_progress),
                'fid_progress': np.array(fid_progress),
                'penalty': penalty_val,
                'learnt_state': best_state,
                'params': params_val,
                'd_r': params_val[0],
                'd_phi': params_val[1],
                'r1': params_val[2],
                'sq_r': params_val[3],
                'sq_phi': params_val[4],
                'r2': params_val[5],
                'kappa': params_val[6]
            }
    end = time.time()
    print("Elapsed time is {} seconds".format(np.round(end - start)))
    print("Final cost = ", cost_val)
    print("Minimum cost = ", min_cost)
    print("Optimum fidelity = ", best_fid)
    sim_results['runtime'] = end-start
    sim_results['cost_progress'] = np.array(cost_progress)
    sim_results['fid_progress'] = np.array(fid_progress)
    np.savez(os.path.join(out_dir, ID+'.npz'), **sim_results)
    return sim_results

def save_plots(target_state, best_state, cost_progress, *, modes, offset=-0.11, l=5,
        out_dir='sim_results', ID='state_learner', **kwargs):
    if modes == 1:
        fig1, ax1 = wigner_3D_plot(target_state, offset=offset, l=l)
        fig1.savefig(os.path.join(out_dir, ID+'_targetWigner.png'))
        fig2, ax2 = wigner_3D_plot(best_state, offset=offset, l=l)
        fig2.savefig(os.path.join(out_dir, ID+'_learntWigner.png'))
        figW1, axW1 = wavefunction_plot(target_state, l=l)
        figW1.savefig(os.path.join(out_dir, ID+'_targetWavefunction.png'))
        figW2, axW2 = wavefunction_plot(best_state, l=l)
        figW2.savefig(os.path.join(out_dir, ID+'_learntWavefunction.png'))
    elif modes == 2:
        figW1, axW1 = two_mode_wavefunction_plot(target_state, l=l)
        figW1.savefig(os.path.join(out_dir, ID+'_targetWavefunction.png'))
        figW2, axW2 = two_mode_wavefunction_plot(best_state, l=l)
        figW2.savefig(os.path.join(out_dir, ID+'_learntWavefunction.png'))
    figC, axC = plot_cost(cost_progress)
    figC.savefig(os.path.join(out_dir, ID+'_cost.png'))

if __name__ == "__main__":
    HP = parse_arguments(HP)
    target_state = HP['target_state_fn'](cutoff=HP['cutoff'], **HP['state_params'])
    HP['modes'] = int(np.log(target_state.shape[0])/np.log(HP['cutoff']))
    print('------------------------------------------------------------------------')
    print('Hyperparameters:')
    print('------------------------------------------------------------------------')
    for key, val in HP.items():
        print("{}: {}".format(key, val))
    print('------------------------------------------------------------------------')
    print('Constructing variational quantum circuit...')
    ket, parameters = variational_quantum_circuit(**HP)
    ket = tf.reshape(ket, [-1])
    print('Beginning optimization...')
    res = optimize(ket, target_state, parameters, **HP)
    print('Generating plots...')
    save_plots(res['learnt_state'], target_state, res['cost_progress'], **HP)