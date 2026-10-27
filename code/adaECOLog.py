import json
import numpy as np

from logbexp.algorithms.algo_factory import create_algo
from logbexp.bandit.arm_set import AdmissibleArmSet, ArmSet
from logbexp.bandit.logistic_oracle import LogisticOracle
from logbexp.utils.utils import sigmoid

# ================================================================
# User settings
# ================================================================

EXACT_TS = False
INFORMATION_GAIN = True
FILENAME = "test_Context_V3_3_seeded.json"

K = P = 6                     # number of contents
D = K * P                     # dimension seen by ECOLog

H = 400                        # maximum number of rounds per episode
N_EPISODES = 2000
RUN = 10

ARM_NORM_UB = 1.0             # upper bound on ||c_t||_2
FAILURE_LEVEL = 0.05

# ---------------------------------------------------------------
# Independent random seeds
# ---------------------------------------------------------------
THETA_SEED = 123
ARM_SEED = 456
ENV_SEED = 789

TA = 199
RA = 1
RP = 0
RD = -1


def make_theta_star(seed):
    """Generate theta_star from its own independent RNG stream."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-1.6, 1.6, size=(K, P))
    if theta.shape != (K, P):
        raise ValueError(
            f"THETA_STAR must have shape ({K}, {P}), got {theta.shape}."
        )
    return theta


def generate_fixed_arm_sets(seed, theta_star_flat):
    """
    Generate all episode arm sets from an isolated NumPy legacy RNG stream.

    ArmSet.generate_arm_list() appears to use NumPy's global RNG internally,
    so we save/restore its global RNG state to prevent arm generation from
    affecting any other random process.
    """
    state = np.random.get_state()
    try:
        np.random.seed(seed)

        fixed_arms = []
        for _ in range(N_EPISODES):
            arm_set = ArmSet(
                AdmissibleArmSet.fxd,
                D,
                K,
                ARM_NORM_UB,
            )
            arm_set.theta_star_flat = theta_star_flat
            arm_set.generate_arm_list()
            # Store an independent copy so a later algorithm cannot mutate it.
            fixed_arms.append(np.asarray(arm_set.arm_list, dtype=float).copy())

        return fixed_arms
    finally:
        np.random.set_state(state)


def make_environment_uniforms(seed, run):
    """
    Pre-generate the environment's Bernoulli uniforms for one run.

    Each run gets a deterministic but independent substream derived from
    ENV_SEED. This removes hidden dependence on the global RNG and makes
    environment randomness fully reproducible.
    """
    seed_sequence = np.random.SeedSequence([seed, run])
    rng = np.random.default_rng(seed_sequence)
    return rng.random((N_EPISODES, H))


# ================================================================
# Problem/reward helpers
# ================================================================


def compute_expected_value(ts, safest, riskiest, step=0):
    V = 0.0
    p_active = 1.0
    for t in range(step, H):
        # reward if active at this time
        if t < TA:
            R_active = RA
        else:
            R_active = RD

        # choose arm depending on switching time
        if t < ts:
            q = 1.0 - safest       # probability to remain active
        else:
            q = 1.0 - riskiest

        # expected reward at this step
        V += p_active * R_active + (1.0 - p_active) * RP

        # update probability of being active next step
        p_active *= q

    return V


def get_min_max(arm_set, theta_star_flat):
    arms = np.asarray(arm_set.arm_list, dtype=float)

    logits = arms @ theta_star_flat
    estimates = np.asarray(sigmoid(logits), dtype=float)

    return min(estimates), max(estimates)


def compute_estimates(algo, arm_set):
    arms = np.asarray(arm_set.arm_list, dtype=float)

    # Current ECOLog point estimate theta_hat_t.
    theta_hat = np.asarray(algo.theta, dtype=float)

    logits = arms @ theta_hat
    estimates = np.asarray(sigmoid(logits), dtype=float)

    return estimates


def compute_ts(max_prob_exit):
    Ta = TA + 1
    Ra_eff = RA - RP
    channel_den = (
        Ra_eff
        - (RD - RP) * (1 - (1 - max_prob_exit) ** (H - Ta))
    )

    numerator_arg = Ra_eff / channel_den

    t_s = np.floor(
        Ta - np.log(numerator_arg) / np.log(1 - max_prob_exit)
    )

    if np.isnan(t_s) or t_s < 0:
        t_s = 0

    return t_s - 1


def upper_lower_ts(algo, arm_set):
    arms = np.asarray(arm_set.arm_list, dtype=float)
    estimates = compute_estimates(algo, arm_set)

    vinv = np.asarray(algo.vtilde_matrix_inv, dtype=float)
    conf_radius = float(algo.conf_radius)

    arm_vinv = arms @ vinv
    norms = np.sqrt(np.maximum(np.sum(arm_vinv * arms, axis=1), 0.0))
    bonuses = conf_radius * norms

    max_U_p = np.max(np.clip(estimates + bonuses, 1e-6, 1 - 1e-6))
    max_L_p = np.max(np.clip(estimates - bonuses, 1e-6, 1 - 1e-6))

    lower_t_s = compute_ts(max_L_p)
    upper_t_s = compute_ts(max_U_p)
    return lower_t_s, upper_t_s


def choose_action(algo, arm_set, t, l_ts, u_ts=None):
    if u_ts is None:
        in_gap = False
        current_l_ts = l_ts
    else:
        in_gap = l_ts < t <= u_ts
        current_l_ts = l_ts

    estimates = compute_estimates(algo, arm_set)
    arms = np.asarray(arm_set.arm_list, dtype=float)

    vinv = np.asarray(algo.vtilde_matrix_inv, dtype=float)
    conf_radius = float(algo.conf_radius)
    arm_vinv = arms @ vinv
    norms = np.sqrt(np.maximum(np.sum(arm_vinv * arms, axis=1), 0.0))

    bonuses = conf_radius * norms

    if INFORMATION_GAIN and in_gap:
        # Confidence intervals on exit probabilities
        U_p = np.clip(estimates + bonuses, 1e-6, 1 - 1e-6)
        L_p = np.clip(estimates - bonuses, 1e-6, 1 - 1e-6)

        safest0 = np.clip(np.min(estimates), 1e-6, 1 - 1e-6)
        riskiest0 = np.clip(np.max(estimates), 1e-6, 1 - 1e-6)

        ts0 = compute_ts(riskiest0)
        J0 = compute_expected_value(ts0, safest0, riskiest0, t + 1)

        bonus = np.zeros(K)

        # Information gain for each arm
        for arm in range(K):
            JU = J0
            JL = J0
            p = estimates.copy()

            # Upper confidence perturbation
            if U_p[arm] < np.min(p) or U_p[arm] > np.max(p):
                p[arm] = U_p[arm]
                safest = np.clip(np.min(p), 1e-6, 1 - 1e-6)
                riskiest = np.clip(np.max(p), 1e-6, 1 - 1e-6)
                ts = compute_ts(riskiest)
                JU = compute_expected_value(ts, safest, riskiest, t + 1)

            # Lower confidence perturbation
            p = estimates.copy()
            if L_p[arm] < np.min(p) or L_p[arm] > np.max(p):
                p[arm] = L_p[arm]
                safest = np.clip(np.min(p), 1e-6, 1 - 1e-6)
                riskiest = np.clip(np.max(p), 1e-6, 1 - 1e-6)
                ts = compute_ts(riskiest)
                JL = compute_expected_value(ts, safest, riskiest, t + 1)

            bonus[arm] = max(JU - J0, JL - J0)

        # Standard Dynamic Programming / Bellman lookahead style allocation
        r_active = RA if t < TA else RD

        Q = np.zeros(K)
        v_next = compute_expected_value(ts0, safest0, riskiest0, t + 1)
        for arm in range(K):
            p_leave = estimates[arm]
            Q[arm] = (
                r_active
                + (p_leave * (H - 1 - t) * RP)
                + ((1.0 - p_leave) * v_next)
            )

        return int(np.argmax(Q + bonus))

    elif in_gap:
        lcb_values = estimates - bonuses
        return int(np.argmin(lcb_values))

    else:
        if t < current_l_ts:
            lcb_values = estimates - bonuses
            return int(np.argmin(lcb_values))
        else:
            ucb_values = estimates + bonuses
            return int(np.argmax(ucb_values))


# ================================================================
# One episode
# ================================================================


def run_episode(episode, theta_star_flat, algo, fixed_arms, env_uniforms):
    """Run one episode with fixed arms and a predetermined random stream."""

    arm_set = ArmSet(
        AdmissibleArmSet.fxd,
        D,
        K,
        ARM_NORM_UB,
    )
    arm_set.arm_list = np.asarray(fixed_arms[episode], dtype=float).copy()
    arms_list = arm_set.arm_list

    pmin, pmax = get_min_max(arm_set, theta_star_flat)

    # Keep LogisticOracle for its expected_reward helper, but do NOT call
    # oracle.pull(). The environment outcome is sampled from env_uniforms,
    # whose RNG is controlled independently by ENV_SEED.
    oracle = LogisticOracle(theta_star_flat)

    ep_reward = 0
    exit_round = H - 1

    for t in range(H):
        algo.update_ucb_bonus()

        if EXACT_TS:
            max_exit_prob = np.max(compute_estimates(algo, arm_set))
            t_s = compute_ts(max_exit_prob)
            action_idx = choose_action(algo, arm_set, t, t_s)
        else:
            lower_ts, upper_ts = upper_lower_ts(algo, arm_set)
            action_idx = choose_action(algo, arm_set, t, lower_ts, upper_ts)

        action = arms_list[action_idx]

        p_exit = float(sigmoid(np.dot(action, theta_star_flat)))
        termination = bool(env_uniforms[episode, t] < p_exit)

        algo.ctr += 1
        algo.learn(action, termination)

        reward = RA if t < TA else RD
        ep_reward += reward

        if termination:
            exit_round = t
            break

    return {
        "rewards": ep_reward,
        "exit": exit_round,
        "pmin": pmin,
        "pmax": pmax,
    }


# ================================================================
# Main
# ================================================================


def main():
    theta_star_structured = make_theta_star(THETA_SEED)
    theta_star_flat = theta_star_structured.reshape(D)

    fixed_arms = generate_fixed_arm_sets(ARM_SEED, theta_star_flat)

    print("THETA_SEED:", THETA_SEED)
    print("ARM_SEED:", ARM_SEED)
    print("ENV_SEED:", ENV_SEED)
    print()
    print("THETA_STAR shape:", theta_star_structured.shape)
    print("THETA_STAR:\n", theta_star_structured)
    print("theta_star_flat shape:", theta_star_flat.shape)
    print("K:", K)
    print("P:", P)
    print("ECOLog dimension D=K*P:", D)
    print("episodes:", N_EPISODES)
    print("horizon per episode:", H)
    print()

    config = {
        "algo_name": "adaECOLog",
        "dim": D,
        "param_norm_ub": np.linalg.norm(theta_star_flat) + 1.0,
        "arm_norm_ub": ARM_NORM_UB,
        "failure_level": FAILURE_LEVEL,
    }

    tot_rewards = []
    tot_exit = []

    for run in range(RUN):
        env_uniforms = make_environment_uniforms(ENV_SEED, run)

        algo = create_algo(config)
        algo.theta = np.full(algo.dim, -2.0)
        algo.l2reg = 15
        algo.vtilde_matrix = 15 * np.eye(D)
        algo.vtilde_matrix_inv = (1 / 15) * np.eye(D)

        rewards = []
        exit_rounds = []
        pmin = []
        pmax = []

        for episode in range(N_EPISODES):
            result = run_episode(
                episode,
                theta_star_flat,
                algo,
                fixed_arms,
                env_uniforms,
            )

            rewards.append(result["rewards"])
            exit_rounds.append(result["exit"])
            pmin.append(result["pmin"])
            pmax.append(result["pmax"])

            print(
                f"Episode {episode + 1:3d}: "
                f"Run {run + 1:3d}: "
                f"Exit = {result['exit']} "
                f"Pmin = {result['pmin']} "
                f"Pmax = {result['pmax']}"
            )

        tot_rewards.append(rewards)
        tot_exit.append(exit_rounds)

    return tot_rewards, tot_exit, pmin, pmax, theta_star_structured


if __name__ == "__main__":
    rewards, exit_rounds, pmin, pmax, theta_star = main()

    with open(FILENAME, "w") as f:
        save = {
            "seeds": {
                "theta_seed": THETA_SEED,
                "arm_seed": ARM_SEED,
                "env_seed": ENV_SEED,
            },
            "theta_star": theta_star.tolist(),
            "rewards": rewards,
            "exit": exit_rounds,
            "pmin": pmin,
            "pmax": pmax,
        }
        json.dump(save, f)