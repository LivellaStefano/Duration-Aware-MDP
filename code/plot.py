import json
import matplotlib.pyplot as plt
import numpy as np

# Configuration
filenames = ["exact_tau_s", "boundary_confidence", "voi_exploration", "ucbvi"]
labels = ["Strategy I", "Strategy II", 
          "Strategy III", "UCBVI-BF"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, Orange, Green, Red

# Optimal baseline reward computed using dynamic programming
baseline_reward = 192.22449407937444

stride = 100

fig, ax = plt.subplots(figsize=(10, 6))

for i, filename in enumerate(filenames):
    # Load data
    with open(f"{filename}.json", "r") as f:
        data = json.load(f)
    
    rewards = np.array(data["rewards"])

    # Compute Regret
    regrets = baseline_reward - rewards
    cumulative_regrets = np.cumsum(regrets, axis=1)
    mean_cumulative_regret = np.mean(cumulative_regrets, axis=0)
    std_cumulative_regret = np.std(cumulative_regrets, axis=0)

    episodes = np.arange(1, rewards.shape[1] + 1)

    ep_plot = episodes[::stride]
    mean_plot = mean_cumulative_regret[::stride]
    CI_cumulative_regret = 1.66 * np.std(cumulative_regrets, axis=0) / len(regrets)

    label = labels[i] if labels and i < len(labels) else filename
    color = colors[i % len(colors)]

    # Plot mean curve
    ax.plot(ep_plot, mean_plot, label=label, color=color, linewidth=2)

    # Shaded error band
    ax.fill_between(
        ep_plot,
        mean_plot - CI_cumulative_regret[::stride],
        mean_plot + CI_cumulative_regret[::stride],
        color=color,
        alpha=0.2,
    )

ax.set_xlabel("Episode")
ax.set_ylabel("Cumulative Regret")
ax.set_title("Cumulative Regret Comparison")
ax.legend(loc="upper left")
ax.grid(True)
# For better visualization of the three strategies
# ax.set_ylim(0, 20000)

plt.tight_layout()
plt.savefig("cumulative_regret_comparison.png", dpi=300, bbox_inches="tight")
plt.show()