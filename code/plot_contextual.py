import json
import matplotlib.pyplot as plt
import numpy as np

filenames = ["exact_tau_s", "boundary_confidence", "voi_exploration"]
labels = ["Strategy I", "Strategy II", 
          "Strategy III"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green, Red

stride = 100

fig, ax = plt.subplots(figsize=(10, 6))

H = 400
Ta = 199

def solve_forced_exit_mdp(actions, Ra=1.0, Rd=-1.0, Rp=0.0):
    V = [0.0] * (H + 1)
    policy = [None] * H
    
    V[H] = 0.0
    
    for t in range(H - 1, -1, -1):
        r_active = Ra if t < Ta else Rd
        best_expected_reward = -float('inf')
        best_action = None
        
        for i, a in enumerate(actions):
            p_leave = a['p_leave']
            expected_future = (p_leave * (H - 1 - t) * Rp) + ((1 - p_leave) * V[t+1])
            expected_total = r_active + expected_future

            if expected_total > best_expected_reward:
                best_expected_reward = expected_total
                best_action =  i
        
        V[t] = best_expected_reward
        policy[t] = best_action
            
    return V

for i, filename in enumerate(filenames):
    # Load data
    with open(f"results/{filename}.json", "r") as f:
        data = json.load(f)

    # Shape: (n_runs, n_episodes)
    rewards = np.array(data["rewards"])
    pmin = data["pmin"]
    pmax = data["pmax"]
    
    baseline_reward = []
    for idx in range(len(pmin)):
       actions =  [{'p_leave': pmin[idx]}, {'p_leave': pmax[idx]}]
       V = solve_forced_exit_mdp(actions)
       baseline_reward.append(V[0])

    # Compute Regret
    regrets = np.array(baseline_reward) - np.array(rewards)
    cumulative_regrets = np.cumsum(regrets, axis=1)

    # Statistics across runs
    mean_cumulative_regret = np.mean(cumulative_regrets, axis=0)
    std_cumulative_regret = np.std(cumulative_regrets, axis=0)
    std_cumulative_regret = 1.697 * np.std(cumulative_regrets, axis=0) / np.sqrt(len(regrets))

    episodes = np.arange(1, rewards.shape[1] + 1)

    # Downsampling
    ep_plot = episodes[::stride]
    mean_plot = mean_cumulative_regret[::stride]
    std_plot = std_cumulative_regret[::stride]

    label = labels[i] if labels and i < len(labels) else filename
    color = colors[i % len(colors)]  # Pick color from list

    # Plot mean curve
    ax.plot(ep_plot, mean_plot, label=label, color=color, linewidth=2)

    # Shaded error band using the exact same color
    ax.fill_between(
        ep_plot,
        mean_plot - std_plot,
        mean_plot + std_plot,
        color=color,
        alpha=0.2,
    )

# -------------------------
# Formatting
# -------------------------
ax.set_xlabel("Episode")
ax.set_ylabel("Cumulative Regret")
ax.set_title("Cumulative Regret Comparison")
ax.legend(loc="lower right")
ax.grid(True)
#ax.set_ylim(0, 20000)

plt.tight_layout()
plt.savefig("cumulative_regret_comparison.png", dpi=300, bbox_inches="tight")
plt.show()