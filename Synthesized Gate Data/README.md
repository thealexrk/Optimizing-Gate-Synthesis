# Synethesized Gate Data

This folder contains the saved NumPy npz files containing hyperparameters, learnt circuit parameters, and other properties and data for the synthesized gates. These gates include 3 single mode gates (Cubic phase gate with gamma=0.1, QFT in the Fock basis, and Random unitary) alongside one two mode gate (Cross-Kerr interaction with kappa=0.1).

To access the saved data, the file can be loaded using NumPy:

```python
results = np.load('Cubic_phase.npz')
```

The individual hyperparameters and results can then be accessed via the respective key. For example, to extract the learnt state, as well as a list of the variational circuit layer squeezing magnitudes:

```python
learnt_state = results['learnt_state']
squeezing = results['sq_r']
```

### Available Keys

|       Keys       |   Value type   |                                           Description                                           |
|------------------|----------------|-------------------------------------------------------------------------------------------------|
| `U_target`       | array[complex] | The target unitary matrix.                                                                      |
| `U_param`        | array[float]   | The target gate parameters.                                                                     |
| `gate_cutoff`    | integer        | The number of input-output relations.                                                           |
| `cutoff`         | integer        | The simulation Fock basis truncation.                                                           |
| `gate_cutoff`    | integer        | The number of input-output relations.                                                           |
| `depth`          | integer        | Number of layers in the variational quantum circuit.                                            |
| `reps`           | integer        | Number of optimization steps to performed.                                                      |
| `min_cost`       | float          | Minimum value of the cost achieved during the optimization.                                     |
| `cost_vs_step`   | array[float]   | Value of the cost function for each optimization step.                                          |
| `sq_r`           | array[float]   | Squeezing magnitude of each layer in the variational quantum circuit.                           |
| `sq_phi`         | array[float]   | Squeezing phase of each layer in the variational quantum circuit.                               |
| `disp_r`         | array[float]   | Displacement magnitude of each layer in the variational quantum circuit.                        |
| `disp_phi`       | array[float]   | Displacement phase of each layer in the variational quantum circuit.                            |
| `r1`             | array[float]   | Initial phase space rotation of leach layer.                                                    |
| `r2`             | array[float]   | Final phase space rotation of leach layer.                                                      |
| `kappa`          | array[float]   | Non-linear Kerr interaction strength of each layer.                                             |
| `U_output`       | array[float]   | The synthesized unitary applied by the variational quantum circuit when the cost was minimized. |
| `state_result`   | array[complex] | The resulting state after applying the synthesized unitary to an equal superposition state.     |
| `state_expected` | array[complex] | The expected state after applying the target unitary to an equal superposition state.           |
| `state_fidelity` | float          | The fidelity between the expected state and the output state.                                   |
