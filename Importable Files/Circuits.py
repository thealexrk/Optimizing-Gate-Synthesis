import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *

def one_mode_variational_quantum_circuit(cutoff, input_state=None, batch_size=None,
        depth=25, active_sd=0.0001, passive_sd=0.1, **kwargs):    
    with tf.name_scope('variables'):        
        d_r = tf.Variable(tf.random_normal(shape=[depth], stddev=active_sd))
        d_phi = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        r1 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        sq_r = tf.Variable(tf.random_normal(shape=[depth], stddev=active_sd))
        sq_phi = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        r2 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        kappa = tf.Variable(tf.random_normal(shape=[depth], stddev=active_sd))
    parameters = [d_r, d_phi, r1, sq_r, sq_phi, r2, kappa]

    def layer(i, q, m):
        with tf.name_scope('layer_{}'.format(i)):
            Dgate(d_phi[i]) | q[m]
            Rgate(r1[i]) | q[m]
            Sgate(sq_r[i], sq_phi[i]) | q[m]
            Rgate(r2[i]) | q[m]
            Kgate(kappa[i]) | q[m]
        return q
    
    sf.hbar = 0.5
    prog = sf.Program(1)    
    with prog.context as q:
        if input_state is not None:
            Ket(input_state) | q
        for k in range(depth):
            q = layer(k, q, 0)
    if batch_size is not None:
        eng = sf.Engine("tf", backend_options={"cutoff_dim": cutoff, "batch_size": batch_size})
    else:
        eng = sf.Engine("tf", backend_options={"cutoff_dim": cutoff})
    state = eng.run(prog, run_options={"eval": False}).state
    ket = state.ket()
    return ket, parameters

def two_mode_variational_quantum_circuit(cutoff, input_state=None, batch_size=None,
        depth=25, active_sd=0.0001, passive_sd=0.1, **kwargs):
    with tf.name_scope('variables'):
        theta1 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        phi1 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        r1 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        sq_r = tf.Variable(tf.random_normal(shape=[2, depth], stddev=active_sd))
        sq_phi = tf.Variable(tf.random_normal(shape=[2, depth], stddev=passive_sd))
        theta2 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        phi2 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        r2 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
        d_r = tf.Variable(tf.random_normal(shape=[2, depth], stddev=active_sd))
        d_phi = tf.Variable(tf.random_normal(shape=[2, depth], stddev=passive_sd))
        kappa = tf.Variable(tf.random_normal(shape=[2, depth], stddev=active_sd))
    parameters = [theta1, phi1, r1, sq_r, sq_phi, theta2, phi2, r2, d_r, d_phi, kappa]
    
    def layer(i, q):
        with tf.name_scope('layer_{}'.format(i)):
            BSgate(theta1[k], phi1[k]) | (q[0], q[1])
            Rgate(r1[i]) | q[0]
            for m in range(2):
                Sgate(sq_r[m, i], sq_phi[m, i]) | q[m]
            BSgate(theta2[k], phi2[k]) | (q[0], q[1])
            Rgate(r2[i]) | q[0]
            for m in range(2):
                Dgate(d_r[m, i],  d_phi[m, i]) | q[m]
                Kgate(kappa[m, i]) | q[m]
        return q

    sf.hbar = 2
    prog = sf.Program(2)
    with eng:
        if input_state is not None:
            Ket(input_state) | q
        for k in range(depth):
            q = layer(k, q)
    if batch_size is not None:
        eng = sf.Engine("tf", backend_options={"cutoff_dim": cutoff, "batch_size": batch_size})
    else:
        eng = sf.Engine("tf", backend_options={"cutoff_dim": cutoff})

    state = eng.run(prog, run_options={"eval": False}).state
    ket = state.ket()
    return ket, parameters

def variational_quantum_circuit(*, modes, cutoff, input_state=None, batch_size=None,
        depth=25, active_sd=0.0001, passive_sd=0.1, **kwargs):
    if modes == 2:
        return two_mode_variational_quantum_circuit(cutoff, input_state, batch_size, depth, active_sd, passive_sd, **kwargs)
    return one_mode_variational_quantum_circuit(cutoff, input_state, batch_size, depth, active_sd, passive_sd, **kwargs)