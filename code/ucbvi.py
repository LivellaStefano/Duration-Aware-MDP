import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import json

class BernoulliMABEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, preferences, TARGET, Ra, Rd, Rp, MAX_STEPS):
        super().__init__()
        self.true_probs = preferences
        self.is_active = True
        self.Ra = Ra
        self.Rd = Rd
        self.Rp = Rp
        self.NUM_ACTIONS = len(preferences)
        self.max_steps = MAX_STEPS
        self.TARGET = TARGET    
        self.current_step = 0

        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.observation_space = spaces.Discrete(2) # 0: Active, 1: Inactive

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.is_active = True
        return 0, {}

    def step(self, action):
        # 1. Evaluate reward for the CURRENT step and CURRENT state
        if self.is_active:
            reward = self.Ra if self.current_step < self.TARGET else self.Rd
        else:
            reward = self.Rp

        # 2. Check early termination event (exit probability)
        if self.is_active:
            exit_event = bool(self.np_random.random() < self.true_probs[action])
            if exit_event:
                self.is_active = False

        self.current_step += 1
        truncated = bool(self.current_step >= self.max_steps)
        
        next_state = 0 if self.is_active else 1
        
        return next_state, reward, False, truncated, {}


class UCBVI:
    def __init__(self, num_states, num_actions, horizon, bonus_type='hoeffding', delta=0.05, total_episodes=50000):
        self.S = num_states         # 2 states
        self.A = num_actions        # Number of actions
        self.H = horizon            # Episode horizon
        self.delta = delta
        self.K = total_episodes
        self.T = self.K * self.H    # Total steps across training horizon
        
        # Step-dependent counters
        self.N_sash = np.zeros((self.H, self.S, self.A, self.S), dtype=np.int32)
        self.N_sah = np.zeros((self.H, self.S, self.A), dtype=np.int32)
        self.R_sumh = np.zeros((self.H, self.S, self.A), dtype=np.float64)
        
        # Step-and-state visit counter: N'_{k, h}(y)
        self.N_sh = np.zeros((self.H + 1, self.S), dtype=np.int32)

        self.V = np.zeros((self.H + 1, self.S))
        self.Q = np.zeros((self.H, self.S, self.A))
        self.policy = np.zeros((self.H, self.S), dtype=np.int32)
        self.total_episodes = 0

    def compute_bonus_2(self, h, P_hat, counts):
        """ Implements exact bonus_2 from Azar et al. (2017) """
        # L = ln(5 * S * A * T / delta)
        L = np.log(5.0 * self.S * self.A * self.T / self.delta)
        
        # --- Term 1: Variance of V_{h+1} ---
        next_V = self.V[h + 1][None, None, :]  # Shape: (1, 1, S)
        expected_V = np.sum(P_hat * next_V, axis=-1)
        expected_V_sq = np.sum(P_hat * (next_V ** 2), axis=-1)
        var_V = np.maximum(0.0, expected_V_sq - (expected_V ** 2))
        
        term1 = np.sqrt((8.0 * L * var_V) / counts)
        
        # --- Term 2: Deterministic 1/N Bound ---
        term2 = (14.0 * self.H * L) / (3.0 * counts)
        
        # --- Term 3: Transition Probability Uncertainty Bound ---
        N_prime_next = np.maximum(self.N_sh[h + 1], 1)[None, None, :]  # Shape: (1, 1, S)
        inner_min = np.minimum(
            (100.0**2 * (self.H**3) * (self.S**2) * self.A * (L**2)) / N_prime_next,
            self.H**2
        )
        term3 = np.sqrt((8.0 * np.sum(P_hat * inner_min, axis=-1)) / counts)
        
        return term1 + term2 + term3

    def compute_value_iteration(self):
        # Terminal boundary condition
        self.V[self.H, :] = 0.0

        for h in range(self.H - 1, -1, -1):
            rem_h = float(self.H - h)
            counts = np.maximum(self.N_sah[h], 1)
            visited = self.N_sah[h] > 0
            
            # Transition probabilities P_hat(s' | s, a) at step h
            P_hat = np.zeros((self.S, self.A, self.S))
            P_hat[visited] = self.N_sash[h][visited] / self.N_sah[h][visited][:, None]
            
            # Expected rewards R_hat(s, a) at step h
            default_reward = +1 if h < self.H/2 else -1
            R_hat = np.full((self.S, self.A), default_reward, dtype=np.float64)
            R_hat[visited] = self.R_sumh[h][visited] / self.N_sah[h][visited]

            # Compute theoretical bonus_2
            bonus = self.compute_bonus_2(h, P_hat, counts)

            expected_next_V = np.sum(P_hat * self.V[h + 1][None, None, :], axis=-1)
            
            current_Q = R_hat + expected_next_V + bonus
            
            # Optimism for unvisited state-action pairs
            current_Q[~visited] = rem_h
            
            # State 1 (Inactive) is absorbing: force value to 0.0
            current_Q[1, :] = 0.0

            # Optimistic upper bound clipping at rem_h = H - h
            self.Q[h] = np.minimum(rem_h, current_Q)
            self.V[h] = np.max(self.Q[h], axis=-1)
            self.policy[h] = np.argmax(self.Q[h], axis=-1)

    def select_action(self, state, horizon_step):
        return self.policy[horizon_step, state]

    def update_counts(self, state, action, next_state, reward, h):
        self.N_sash[h, state, action, next_state] += 1
        self.N_sah[h, state, action] += 1
        self.R_sumh[h, state, action] += reward
        self.N_sh[h, state] += 1  # Track N'_{k, h}(y)


# --- Run Script ---
if __name__ == "__main__":
    Ta = 19  
    H = 40
    Ra = 1.0  
    Rd = -1.0 
    Rp = 0.0  
    NUM_EPISODES = 50000
    STATES = 2
    
    #V2
    #V5
    #true_action_probs = [0.119313, 0.000072, 0.046969, 0.051466, 0.086010, 0.044333, 0.095620, 0.035012, 0.068033, 0.025984]

    #V3
    #V4
    #true_action_probs = [0.685390, 0.000022, 0.196143, 0.578494, 0.168936, 0.270961, 0.498791, 0.254502, 0.392775, 0.307897]

    """
    true_action_probs = [0.046969,0.051466,0.086010,0.001699,0.044333,0.119313,0.095620,0.035012,0.094367,0.068033,0.045188,0.074728,0.080366,0.025984,0.092680,0.060602,
                         0.061644,0.094435,0.013879,0.023220,0.007442,0.030557,0.085144,0.067887,0.104719,0.119049,0.012113,0.010987,0.109405,0.084195,0.018511,0.092206,
                         0.080523,0.091877,0.011788,0.042835,0.012768,0.076702,0.045686,0.009259,0.000308,0.109365,0.032777,0.000154,0.063853,0.077697,0.000116,0.058876,
                         0.018331,0.025684,0.059051,0.074589,0.069527,0.046715,0.000254,0.117648,0.003709,0.018216,0.082260,0.026791,0.100824,0.093770,0.076397,0.075709,
                         0.061280,0.114857,0.094763,0.032944,0.118823,0.003981,0.075073,0.010929,0.050630,0.006899,0.000072,0.049719,0.106079,0.000406,0.078762,0.059909,
                         0.118707,0.103915,0.077649,0.035593,0.050558,0.086626,0.006330,0.097003,0.075149,0.117677,0.102966,0.102723,0.020894,0.016674,0.037366,0.020626,
                         0.072176,0.000133,0.092151,0.064924]
    """
    #V7 - 40
    #V8 - 400
    
    true_action_probs = [0.115682,0.000022,0.196143,0.578494,0.168936,0.000279,0.003550,0.065472,0.682907,0.270961,0.498791,0.254502,0.285236,0.392775,0.175457,0.592765,
                         0.000174,0.259423,0.176827,0.383482,0.307897,0.510934,0.671427,0.349730,0.255605,0.600444,0.552708,0.246764,0.507314,0.360440,0.620742,0.232539,
                         0.064011,0.000334,0.656049,0.103113,0.000473,0.468539,0.148953,0.414295,0.641014,0.080192,0.001403,0.066761,0.000307,0.611174,0.003354,0.277099,
                         0.553729,0.432634,0.115377,0.280690,0.053709,0.546318,0.171315,0.250793,0.118945,0.631232,0.507297,0.249132,0.293628,0.000141,0.595872,0.076236,
                         0.000101,0.421029,0.423298,0.403781,0.000455,0.089487,0.667032,0.282226,0.511056,0.280862,0.437097,0.231408,0.616539,0.292294,0.585990,0.685390,
                         0.125039,0.038151,0.150814,0.008184,0.206874,0.388671,0.352704,0.135953,0.107917,0.000319,0.666490,0.494435,0.097421,0.682501,0.328972,0.350159,
                         0.289149,0.357559,0.000107,0.224724]
   
    RUNS = 30
    rewards = []
    exit = []
    for run in range(RUNS):
        print(run)
        env = BernoulliMABEnv(true_action_probs, Ta, Ra, Rd, Rp, MAX_STEPS=H)
        agent = UCBVI(num_states=STATES, num_actions=env.action_space.n, horizon=H, bonus_type='bernstein')
        
        ep_reward = []
        ep_exit = []
        for episode in range(NUM_EPISODES):
            ep_reward.append(0)
            agent.total_episodes += 1
            agent.compute_value_iteration()
            
            state, _ = env.reset()
            truncated = False
            h = 0
            first = True
            while not truncated:
                action = agent.select_action(state, h)
                next_state, reward, _, truncated, _ = env.step(action)

                agent.update_counts(state, action, next_state, reward, h)
                state = next_state
                if state == 1 and first:
                   first = False
                   ep_exit.append(h)
                h += 1
                ep_reward[-1] += reward
            if first:
               ep_exit.append(h - 1)
        print(agent.policy)
        print("\nLearned Policy Matrix [step h, state (0: Active, 1: Inactive)]:")
        rewards.append(ep_reward)
        exit.append(ep_exit)
    
    with open("test_V7_4.json", "w") as f:
        save = {
            "rewards": rewards,
            "exit": exit
        }
        json.dump(save, f)