# Learnt State Data

This folder contains the saved NumPy npz files containing hyperparameters, learnt circuit parameters, and other properties and data for the learnt states. These states include 4 single mode states (Single photon state, ON state with a=1, N=9, Hex GKP state with mu=1, delta=0.3, and Random state) alongside one two mode state (The NOON state with N=5).

To access the saved data, the file can be loaded using NumPy:

```python
results = np.load('Single_photon_state.npz')
```

The individual hyperparameters and results can then be accessed via the respective key. For example, to extract the learnt state, as well as a list of the variational circuit layer squeezing magnitudes:

```python
learnt_state = results['ket']
squeezing = results['sq_r']
```

### Available Keys


|       Keys      |   Value type   |                                        Description                                        |
|-----------------|----------------|-------------------------------------------------------------------------------------------|
| `cutoff`        | integer        | The simulation Fock basis truncation.                                                     |
| `depth`         | integer        | Number of layers in the variational quantum circuit.                                      |
| `reps`          | integer        | Number of optimization steps to performed.                                                |
| `cost`          | float          | Minimum value of the cost achieved during the optimization.                               |
| `fidelity`      | float          | Maximum value of the fidelity achieved during the optimization.                           |
| `cost_function` | array[float]   | Value of the cost function for each optimization step.                                    |
| `sq_r`          | array[float]   | Squeezing magnitude of each layer in the variational quantum circuit.                     |
| `sq_phi`        | array[float]   | Squeezing phase of each layer in the variational quantum circuit.                         |
| `disp_r`        | array[float]   | Displacement magnitude of each layer in the variational quantum circuit.                  |
| `disp_phi`      | array[float]   | Displacement phase of each layer in the variational quantum circuit.                      |
| `r`             | array[float]   | Phase space rotation of leach layer.                                                      |
| `kappa`         | array[float]   | Non-linear Kerr interaction strength of each layer.                                       |
| `ket`           | array[complex] | Variational quantum circuit output/learnt state vector corresponding to maximum fidelity. |
