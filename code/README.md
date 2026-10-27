## `DP_solution.py`

Computes the optimal value function and policy for the problem using finite-horizon dynamic programming.

### Configuration

Set the following parameters:

- `actions`: List of available actions and their corresponding leaving probabilities (`p_leave`).
- `Ta`: Switching time (\(\tau\)).
- `H`: Horizon.
- `Ra`: Reward received before the switching time.
- `Rd`: Reward received after the switching time.
- `Rp`: Reward received after leaving the system.

### Output

The script computes and prints the optimal value function (expected rewards) and the corresponding optimal policy.

## `strategies.py`

Implements the exploration strategies and evaluates their performance over multiple independent runs.

### Configuration

The implemented strategies are:

- **Strategy I (Plug-In Switching-Time Estimation):** `EXACT_TS = True`, `INFORMATION_GAIN = False`. 
- **Strategy II (Confidence-Aware Boundary Exploration):** `EXACT_TS = False`, `INFORMATION_GAIN = False`. 
- **Strategy III (Value-of-Information Sensitivity Exploration):** `EXACT_TS = False`, `INFORMATION_GAIN = True`. 

Additional parameters:

- `exit_probs`: List of action-dependent leaving probabilities.
- `Ta`: Switching time (\(\tau\)).
- `H`: Decision horizon.
- `Ra`, `Rd`, `Rp`: Rewards before the switching time, after the switching time, and after leaving the system.
- `num_episodes`: Number of training episodes per run.
- `RUNS`: Number of independent experimental runs.

### Output

The script trains the selected strategy, records the total reward obtained in each episode, tracks the estimated switching-time information, and saves the results to `voi_exploration.json`.

## `ucbvi.py`

Implements the UCBVI algorithm and evaluates its performance over multiple independent runs.

### Configuration

Set the following parameters:

- `Ta`: Switching time (\(\tau\)).
- `H`: Decision horizon.
- `Ra`, `Rd`, `Rp`: Rewards before the switching time, after the switching time, and after leaving the system.
- `exit_probs`: List of action-dependent leaving probabilities.
- `num_episodes`: Number of training episodes per run.
- `RUNS`: Number of independent experimental runs.
- `bonus_type`: Exploration bonus (`'hoeffding'` or `'bernstein'`).

### Output

The script trains the UCBVI agent, records the total reward obtained in each episode for every run, and saves the results to `ucbvi.json`.

## `plot.py`

Computes and compares the cumulative regret of multiple algorithms from experimental reward data.

### Input

The script expects one JSON file per algorithm, each containing a `rewards` array of shape `(n_runs, n_episodes)`.

### Configuration

Set the following parameters:

- `filenames`: List of input JSON files.
- `baseline_reward`: Optimal expected reward used to compute regret (adjust according to the experimental setting).

### Output

The script computes the mean cumulative regret for each algorithm, plots the comparison with 90% CI bands, and saves the figure as `cumulative_regret_comparison.png`.

## plot_exit.py

Computes and compares the mean absorption step across multiple algorithms using experimental results.

### Input

The script expects one JSON file per algorithm in the results/ directory, each containing an exit array of shape (n_runs, n_episodes), representing the absorption steps across experimental runs and episodes.

### Configuration

Set the following parameters:

- `filenames`: List of input JSON files.

### Output

The script computes the mean absorption step across experimental runs for each episode, plots the comparison with 90% CI bands, and saves the figure as `exit_comparison.png`.

## plot_contextual.py

Similar to `plot.py`, but processes experimental reward data from contextual simulations to compute and compare the cumulative regret of multiple algorithms.

## contextual_strategies.py
Implements and evaluates exploration strategies in a contextual environment using a logistic model for action-dependent exit probabilities. The script generates fixed contextual action sets, simulates episodes over multiple runs, and records rewards and exit times for subsequent performance analysis.

### Setup and Dependencies
This implementation relies on libraries from the original `ada-OFU-ecoLogB` repository. To run the script, the following setup steps are required:
- Clone the `ada-OFU-ecoLogB` repository https://github.com/criteo-research/logistic_bandit.
- Replace the original `arm_set.py` file in the cloned repository with the modified version provided.
- Execute `contextual_strategies.py`.

### Configuration
The implemented strategies are:

- **Strategy I (Plug-In Switching-Time Estimation):** `EXACT_TS = True`, `INFORMATION_GAIN = False`. 
- **Strategy II (Confidence-Aware Boundary Exploration):** `EXACT_TS = False`, `INFORMATION_GAIN = False`. 
- **Strategy III (Value-of-Information Sensitivity Exploration):** `EXACT_TS = False`, `INFORMATION_GAIN = True`. 
Additional parameters:

`K`, `P`: Number of contents and contextual features per content, respectively.
`D`: Dimension of the contextual parameter vector, computed as K * P.
`H`: Maximum number of rounds per episode.
`N_EPISODES`: Number of episodes per run.
`RUN`: Number of experimental runs.
`FAILURE_LEVEL`: Confidence parameter used by the learning algorithm.
`TA`: Switching time at which the active reward changes.
`RA, RD, RP`: Rewards before the switching time, after the switching time, and after leaving the system, respectively.
`THETA_SEED`, `ARM_SEED`, `ENV_SEED`: Independent random seeds for generating the true model parameters, contextual action sets, and environment outcomes.
`FILENAME`: Name of the JSON file used to save the experimental results.

### Output
The script generates a fixed set of contextual actions for each episode, initializes the adaptive ECOLog algorithm, and simulates the selected exploration strategy over multiple runs. It records the total reward and exit round for every episode, together with the minimum and maximum true exit probabilities associated with the available actions.
