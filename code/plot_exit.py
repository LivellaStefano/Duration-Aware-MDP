import json
import matplotlib.pyplot as plt
import numpy as np

# Configuration
filenames = ["exact_tau_s", "boundary_confidence", "voi_exploration", "ucbvi"]
labels = ["Strategy I", "Strategy II", 
          "Strategy III", "UCBVI-BF"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, Orange, Green, Red

# Optional downsampling stride
stride = 100

# -------------------------
# Plotting Setup
# -------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for i, filename in enumerate(filenames):
    # Load data
    with open(f"results/{filename}.json", "r") as f:
        data = json.load(f)

    # Shape: (n_runs, n_episodes)
    steps = np.array(data["exit"]) + 1

    # Statistics across runs
    mean_steps = np.mean(steps, axis=0)
    std_cumulative_regret = np.std(steps, axis=0)
    std_cumulative_regret = 1.697 * np.std(steps, axis=0) / np.sqrt(len(steps))

    episodes = np.arange(1, steps.shape[1] + 1)

    # Downsampling
    ep_plot = episodes[::stride]
    mean_plot = mean_steps[::stride]
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
ax.set_xlabel("Episode", fontsize = 20)
ax.set_ylabel("Absorption Step", fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.legend(loc="lower right", fontsize = 16)

ax.grid(True)

plt.tight_layout()
plt.savefig("exit_comparison.png", dpi=300, bbox_inches="tight")
plt.show()