import os
import time
from itertools import product
import argparse
import json
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
from learner.circuits import variational_quantum_circuit
from learner.gates import (cubic_phase, DFT, random_unitary, cross_kerr, get_modes,
    unitary_state_fidelity, sample_average_fidelity, process_fidelity, average_fidelity)
from learner.plots import (wigner_3D_plot, wavefunction_plot,
    two_mode_wavefunction_plot, plot_cost, one_mode_unitary_plots, two_mode_unitary_plots)

HP = {
    'name': 'random_gif',
    'out_dir': 'sim_results',
    'target_unitary_fn': random_unitary,
    'target_params': {'size': 4},
    'cutoff': 10,
    'gate_cutoff': 4,
    'depth': 25,
    'reps': 2000,
    'penalty_strength': 0,
    'active_sd': 0.0001,
    'passive_sd': 0.1,
    'maps_outside': False,
}

def parse_arguments(defaults):
    parser = argparse.ArgumentParser(description='Quantum gate synthesis.')
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
    parser.add_argument('-p', '--target-params',
        type=json.loads, default=defaults["target_params"], help='Gate parameters')
    parser.add_argument('-c', '--cutoff',
        type=int, default=defaults["cutoff"], help='Fock basis truncation')
    parser.add_argument('-g', '--gate-cutoff',
        type=int, default=defaults["gate_cutoff"], help='Gate/unitary truncation')
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
    hyperparams['ID'] = "{}_d{}_c{}_g{}_r{}".format(
        hyperparams['name'], hyperparams['depth'], hyperparams['cutoff'], hyperparams['gate_cutoff'], hyperparams['reps'])
    hyperparams['out_dir'] = os.path.join(args.out_dir, hyperparams['ID'], '')
    hyperparams['board_name'] = os.path.join('TensorBoard', hyperparams['ID'], '')
    if not os.path.exists(hyperparams['out_dir']):
        os.makedirs(hyperparams['out_dir'])
    return hyperparams

def real_unitary_overlaps(ket, target_unitary, gate_cutoff, cutoff):
    m = len(ket.shape)-1
    if m == 1:
        in_state = np.arange(gate_cutoff)
        target_kets = np.array([target_unitary[:, i] for i in in_state])
        target_kets = tf.constant(target_kets, dtype=tf.complex64)
        overlaps = tf.real(tf.einsum('bi,bi->b', tf.conj(target_kets), ket))
    elif m == 2:
        fock_states = np.arange(gate_cutoff)
        in_state = np.array(list(product(fock_states, fock_states)))
        target_unitary_sf = np.einsum('ijkl->ikjl', target_unitary.reshape([cutoff]*4))
        target_kets = np.array([target_unitary_sf[:, i, :, j] for i, j in in_state])
        target_kets = tf.constant(target_kets, dtype=tf.complex64)
        overlaps = tf.real(tf.einsum('bij,bij->b', tf.conj(target_kets), ket))
    for idx, state in enumerate(in_state.T):
        tf.summary.scalar('overlap_{}'.format(state), tf.abs(overlaps[idx]))
    return overlaps

def optimize(ket, target_unitary, parameters, cutoff, gate_cutoff, reps=1000, penalty_strength=0,
        out_dir='sim_results', ID='gate_synthesis', board_name='TensorBoard',
        dump_reps=100, **kwargs):
    d = gate_cutoff
    c = cutoff
    m = len(ket.shape)-1
    overlaps = real_unitary_overlaps(ket, target_unitary, gate_cutoff, cutoff)
    mean_overlap = tf.reduce_mean(overlaps)
    tf.summary.scalar("mean_overlap", mean_overlap)
    loss = tf.reduce_sum(tf.abs(overlaps - 1))
    tf.summary.scalar('loss', loss)
    if m == 1:
        state_norms = tf.abs(tf.einsum('bi,bi->b', ket, tf.conj(ket)))
    elif m == 2:
        state_norms = tf.abs(tf.einsum('bij,bij->b', ket, tf.conj(ket)))
    norm_deviation = tf.reduce_sum((state_norms - 1)**2)/gate_cutoff
    penalty = penalty_strength*norm_deviation
    tf.summary.scalar('penalty', penalty)
    cost = loss + penalty
    tf.summary.scalar('cost', cost)
    optimiser = tf.train.AdamOptimizer()
    min_cost_optimize = optimiser.minimize(cost)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(board_name)
    merge = tf.summary.merge_all()
    overlap_progress = []
    cost_progress = []
    best_mean_overlap = 0
    best_min_overlap = 0
    best_max_overlap = 0
    start = time.time()
    for i in range(reps):
        _, cost_val, overlaps_val, ket_val, penalty_val, params_val = session.run(
            [min_cost_optimize, cost, overlaps, ket, penalty, parameters])
        mean_overlap_val = np.mean(overlaps_val)
        min_overlap_val = min(overlaps_val)
        max_overlap_val = max(overlaps_val)
        cost_progress.append(cost_val)
        overlap_progress.append(overlaps_val)
        if m == 1:
            learnt_unitary = ket_val.T
        elif m == 2:
            learnt_unitary = ket_val.reshape(d**2, c**2).T
        c = learnt_unitary.shape[0]
        d = learnt_unitary.shape[1]
        Ur = learnt_unitary[:d, :d]
        vmax = np.max([Ur.real, Ur.imag])
        vmin = np.min([Ur.real, Ur.imag])
        cmax = max(vmax, vmin)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].matshow(Ur.real, cmap=plt.get_cmap('Reds'), vmin=-cmax, vmax=cmax)
        ax[1].matshow(Ur.imag, cmap=plt.get_cmap('Greens'), vmin=-cmax, vmax=cmax)
        for a in ax.ravel():
            a.tick_params(bottom=False,labelbottom=False,
                          top=False,labeltop=False,
                          left=False,labelleft=False,
                          right=False,labelright=False)
        ax[0].set_xlabel(r'$\mathrm{Re}(U)$')
        ax[1].set_xlabel(r'$\mathrm{Im}(U)$')
        for a in ax.ravel():
            a.tick_params(color='white', labelcolor='white')
            for spine in a.spines.values():
                spine.set_edgecolor('white')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, '{}.png'.format(i).zfill(4)))
        plt.close(fig)
        if i % dump_reps == 0:
            print("Rep: {} Cost: {:.4f} Overlaps: Mean = {:.4f}, Min = {:.4f}, Max = {:.4f}".format(
                i, cost_val, mean_overlap_val, min_overlap_val, max_overlap_val))
            summary = session.run(merge)
            writer.add_summary(summary, i)
            if i > 0:
                np.savez(os.path.join(out_dir, ID+'.npz'),
                    **sim_results)
        if i > 0 and mean_overlap_val > best_mean_overlap:
            end = time.time()
            best_mean_overlap = mean_overlap_val
            best_min_overlap = min_overlap_val
            best_max_overlap = max_overlap_val
            min_cost = cost_val
            if m == 1:
                learnt_unitary = ket_val.T
            elif m == 2:
                learnt_unitary = ket_val.reshape(d**2, c**2).T
            eq_state_target, eq_state_learnt, state_fid = unitary_state_fidelity(target_unitary, learnt_unitary, cutoff)
            Fe = process_fidelity(target_unitary, learnt_unitary, cutoff)
            avgF = average_fidelity(target_unitary, learnt_unitary, cutoff)
            sim_results = {
                'name': HP['name'],
                'ID': HP['ID'],
                'target_unitary': target_unitary,
                'target_params': HP['target_params'],
                'cutoff': cutoff,
                'gate_cutoff': gate_cutoff,
                'depth': HP['depth'],
                'reps': HP['reps'],
                'penalty_strength': HP['penalty_strength'],
                'best_runtime': end-start,
                'best_rep': i,
                'mean_overlap': mean_overlap_val,
                'min_overlap': min_overlap_val,
                'max_overlap': max_overlap_val,
                'process_fidelity': Fe,
                'avg_fidelity': avgF,
                'min_cost': cost_val,
                'cost_progress': np.array(cost_progress),
                'mean_overlap_progress': np.mean(np.array(overlap_progress), axis=1),
                'min_overlap_progress': np.min(np.array(overlap_progress), axis=1),
                'max_overlap_progress': np.max(np.array(overlap_progress), axis=1),
                'penalty': penalty_val,
                'learnt_unitary': learnt_unitary,
                'params': params_val,
                'r1': params_val[0],
                'sq_r': params_val[1],
                'sq_phi': params_val[2],
                'r2': params_val[3],
                'disp_r': params_val[4],
                'disp_phi': params_val[5],
                'kappa': params_val[6],
                'eq_state_learnt': eq_state_learnt,
                'eq_state_target': eq_state_target,
                'eq_state_fidelity': state_fid
            }
    end = time.time()
    print("\nElapsed time is {} seconds".format(np.round(end - start)))
    print("Final cost = ", cost_val)
    print("Minimum cost = ", min_cost)
    print("\nMean overlap = {}".format(best_mean_overlap))
    print("Min overlap = {}".format(best_min_overlap))
    print("Max overlap = {}".format(best_max_overlap))
    avgFs = sample_average_fidelity(target_unitary, learnt_unitary, cutoff)
    sim_results['sample_avg_fidelity'] = avgFs
    print("\nProcess fidelity = {}".format(Fe))
    print("Average fidelity = {}".format(avgF))
    print("Sampled average fidelity = {}".format(avgFs))
    print("\nEqual superposition state fidelity = ", state_fid)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sim_results['runtime'] = end-start
    sim_results['cost_progress'] = np.array(cost_progress)
    sim_results['mean_overlap_progress'] = np.mean(np.array(overlap_progress), axis=1)
    sim_results['min_overlap_progress'] = np.min(np.array(overlap_progress), axis=1)
    sim_results['max_overlap_progress'] = np.max(np.array(overlap_progress), axis=1)
    np.savez(os.path.join(out_dir, ID+'.npz'), **sim_results)
    return sim_results

def save_plots(target_unitary, learnt_unitary, eq_state_learnt, eq_state_target,
        cost_progress, *, modes, offset=-0.11, l=5, out_dir='sim_results',
        ID='gate_synthesis', **kwargs):
    square = not kwargs.get('maps_outside', True)
    if modes == 1:
        fig1, ax1 = wigner_3D_plot(eq_state_target, offset=offset, l=l)
        fig1.savefig(os.path.join(out_dir, ID+'_targetWigner.png'))
        fig2, ax2 = wigner_3D_plot(eq_state_learnt, offset=offset, l=l)
        fig2.savefig(os.path.join(out_dir, ID+'_learntWigner.png'))
        figW1, axW1 = one_mode_unitary_plots(target_unitary, learnt_unitary, square=square)
        figW1.savefig(os.path.join(out_dir, ID+'_unitaryPlot.png'))
    elif modes == 2:
        figW1, axW1 = two_mode_wavefunction_plot(eq_state_target, l=l)
        figW1.savefig(os.path.join(out_dir, ID+'_targetWavefunction.png'))
        figW2, axW2 = two_mode_wavefunction_plot(eq_state_learnt, l=l)
        figW2.savefig(os.path.join(out_dir, ID+'_learntWavefunction.png'))
        figM1, axM1 = two_mode_unitary_plots(target_unitary, learnt_unitary, square=square)
        figM1.savefig(os.path.join(out_dir, ID+'_unitaryPlot.png'))
    figC, axC = plot_cost(cost_progress)
    figC.savefig(os.path.join(out_dir, ID+'_cost.png'))
    
if __name__ == "__main__":
    HP = parse_arguments(HP)
    target_unitary = HP['target_unitary_fn'](cutoff=HP['cutoff'], **HP['target_params'])
    HP['modes'] = get_modes(target_unitary, HP['cutoff'])
    HP['batch_size'] = HP['gate_cutoff']**HP['modes']
    print('------------------------------------------------------------------------')
    print('Hyperparameters:')
    print('------------------------------------------------------------------------')
    for key, val in HP.items():
        print("{}: {}".format(key, val))
    print('------------------------------------------------------------------------')
    in_ket = np.zeros([HP['gate_cutoff'], HP['cutoff']])
    np.fill_diagonal(in_ket, 1)
    if HP['modes'] == 2:
        in_ket = np.einsum('ij,kl->ikjl', in_ket, in_ket)
        in_ket = in_ket.reshape(HP['gate_cutoff']**2, HP['cutoff'], HP['cutoff'])
    print('Constructing variational quantum circuit...')
    ket, parameters = variational_quantum_circuit(input_state=in_ket, **HP)
    print('Beginning optimization...')
    res = optimize(ket, target_unitary, parameters, **HP)
    print('Generating plots...')
    save_plots(target_unitary, res['learnt_unitary'], res['eq_state_learnt'],
        res['eq_state_target'], res['cost_progress'], **HP)