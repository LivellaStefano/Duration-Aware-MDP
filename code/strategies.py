import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json

class BernoulliMABEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, exit_probs, Ta, Ra, Rd, Rp, MAX_STEPS):
        super().__init__()
        self.exit_probs = exit_probs
        self.is_active = True
        self.Ra = Ra
        self.Rd = Rd
        self.Rp = Rp
        self.num_actions = len(exit_probs)
        self.max_steps = MAX_STEPS
        self.Ta = Ta   
        self.current_step = 0

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.is_active = True
        return 0, {}  

    def step(self, action):
        terminated = self.np_random.random() < self.exit_probs[action]
        
        reward = self.Ra if self.current_step < self.Ta else self.Rd 
        reward = reward if self.is_active else self.Rp
                
        self.current_step += 1
        
        truncated = bool(self.current_step >= self.max_steps)
        return reward, terminated, truncated, {}

class BernsteinUCBAgent:
    def __init__(self, Ra, Rd, Rp, H, Ta, num_actions, c=1.0, INFORMATION_GAIN=True):
        self.num_actions = num_actions
        self.c = c                               
        self.counts = np.zeros(num_actions)      # N(a)
        self.values = np.zeros(num_actions)      # Q(a): Estimated exit probability per arm
        self.sum_sq = np.zeros(num_actions)      
        self.total_steps = 0                     
        self.INFORMATION_GAIN = INFORMATION_GAIN
        self.Ra = Ra
        self.Rd = Rd
        self.Rp = Rp
        self.H = H
        self.Ta = Ta

    def compute_expected_value(self, ts, safest, riskiest, step=0):
        V = 0.0
        p_active = 1.0
        for t in range(step, self.H):
            if t < self.Ta:
                R_active = self.Ra
            else:
                R_active = self.Rd

            # choose arm depending on switching time
            if t < ts:
                q = 1.0 - safest
            else:
                q = 1.0 - riskiest

            # expected reward at this step
            V += p_active * R_active + (1.0 - p_active) * self.Rp

            # update probability of being active next step
            p_active *= q

        return V

    def select_action(self, step, l_ts, u_ts=None):
        self.total_steps += 1
        
        # Force initial exploration of unpulled arms
        if np.any(self.counts == 0):
            return np.argmin(self.counts)
        
        variance = (self.sum_sq / self.counts) - (self.values ** 2)
        variance = np.maximum(variance, 0.0) 

        # Handle variation between exact single t_s versus interval gap
        if u_ts is None:
            in_gap = False
            current_l_ts = l_ts
        else:
            in_gap = l_ts < step <= u_ts
            current_l_ts = l_ts

        if self.INFORMATION_GAIN and in_gap:
            # Confidence intervals on exit probabilities
            radius = self.c * (
                np.sqrt(2 * variance * np.log(self.total_steps) / self.counts)
                + 3 * np.log(self.total_steps) / self.counts
            )

            U_p = np.clip(self.values + radius, 1e-6, 1.0 - 1e-6)
            L_p = np.clip(self.values - radius, 1e-6, 1.0 - 1e-6)

            safest0 = np.clip(np.min(self.values), 1e-6, 1.0 - 1e-6)
            riskiest0 = np.clip(np.max(self.values), 1e-6, 1.0 - 1e-6)

            ts0 = compute_ts(self.Ta, self.H, self.Ra, self.Rd, self.Rp, riskiest0)
            J0 = self.compute_expected_value(ts0, safest0, riskiest0, step + 1)

            bonuses = np.zeros(self.num_actions)

            # Information gain for each arm
            for arm in range(self.num_actions):
                JU = J0
                JL = J0
                p = self.values.copy()

                # Upper confidence perturbation
                if U_p[arm] < np.min(p) or U_p[arm] > np.max(p):
                    p[arm] = U_p[arm]
                    safest = np.min(p)
                    riskiest = np.max(p)
                    ts = compute_ts(self.Ta, self.H, self.Ra, self.Rd, self.Rp, riskiest)
                    JU = self.compute_expected_value(ts, safest, riskiest, step + 1)

                # Lower confidence perturbation 
                if L_p[arm] < np.min(p) or L_p[arm] > np.max(p):
                    p = self.values.copy()
                    p[arm] = L_p[arm]
                    safest = np.min(p)
                    riskiest = np.max(p)
                    ts = compute_ts(self.Ta, self.H, self.Ra, self.Rd, self.Rp, riskiest)
                    JL = self.compute_expected_value(ts, safest, riskiest, step + 1)

                bonuses[arm] = max(JU - J0, JL - J0)

            r_active = self.Ra if step < self.Ta else self.Rd
            Q = np.zeros(self.num_actions)
            for arm in range(self.num_actions):
                p_leave = self.values[arm]
                v_next = self.compute_expected_value(ts0, safest0, riskiest0, step + 1)
                Q[arm] = r_active + (p_leave * (self.H - 1 - step) * self.Rp) + ((1.0 - p_leave) * v_next)

            return np.argmax(Q + bonuses)
        elif in_gap:
            log_t = np.log(self.total_steps)
            exploration_bonus = self.c * (np.sqrt(2 * variance * log_t / self.counts) + (3 * log_t / self.counts))
            ucb_values = self.values - exploration_bonus
            return np.argmin(ucb_values)
        else:
            log_t = np.log(self.total_steps)
            exploration_bonus = self.c * (np.sqrt(2 * variance * log_t / self.counts) + (3 * log_t / self.counts))

            if step < current_l_ts:
                ucb_values = self.values - exploration_bonus
                return np.argmin(ucb_values)
            else:
                ucb_values = self.values + exploration_bonus
                return np.argmax(ucb_values)

    def update(self, action, terminated):
        self.counts[action] += 1
        n = self.counts[action]
        
        exit_signal = 1.0 if terminated else 0.0
        
        self.values[action] += (exit_signal - self.values[action]) / n
        self.sum_sq[action] += exit_signal ** 2


def compute_ts(Ta, H, Ra, Rd, Rp, max_prob_exit):
    Ta = Ta + 1
    Ra_eff = Ra - Rp
    channel_den = Ra_eff - (Rd - Rp) * (1 - (1 - max_prob_exit)**(H - Ta))
    
    numerator_arg = Ra_eff / channel_den
        
    t_s = np.floor(Ta - np.log(numerator_arg) / np.log(1 - max_prob_exit))
    if np.isnan(t_s) or t_s < 0:
        t_s = 0
    return t_s - 1 # Adjust for zero-based indexing


def ts_interval(Ta, H, Ra, Rd, Rp, agent):
    if agent.total_steps == 0 or np.any(agent.counts == 0):
        return 0, Ta
        
    variance = (agent.sum_sq / agent.counts) - (agent.values ** 2)
    variance = np.maximum(variance, 0.0) 
    
    log_t = np.log(agent.total_steps)
    bonus = agent.c * (np.sqrt(2 * variance * log_t / agent.counts) + (3 * log_t / agent.counts))
    
    max_U_p = max(np.clip(agent.values + bonus, 1e-6, 1 - 1e-6))
    max_L_p = max(np.clip(agent.values - bonus, 1e-6, 1 - 1e-6))

    if np.isnan(max_U_p): max_U_p = 1.0 - 1e-6
    if np.isnan(max_L_p): max_L_p = 1e-6

    compute_ts_lower = compute_ts(Ta, H, Ra, Rd, Rp, max_L_p)
    compute_ts_upper = compute_ts(Ta, H, Ra, Rd, Rp, max_U_p)
    return int(compute_ts_lower), int(compute_ts_upper)

if __name__ == "__main__":
    EXACT_TS = False
    INFORMATION_GAIN = True
    #V2
    #V5
    #true_action_probs = [0.119313, 0.000072, 0.046969, 0.051466, 0.086010, 0.044333, 0.095620, 0.035012, 0.068033, 0.025984]

    #V3
    #V4
    #true_action_probs = [0.685390, 0.000022, 0.196143, 0.578494, 0.168936, 0.270961, 0.498791, 0.254502, 0.392775, 0.307897]


    #V6 - 40
    #V9 - 400
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
    
    print(max(true_action_probs), min(true_action_probs))
    Ta = 19
    H = 40
    Ra = 1.0  
    Rd = -1.0
    Rp = 0.0  
    num_episodes = 50000
    RUNS = 30
    run_rewards = []
    run_exit = []
    t_s_computed = []

    for run in range(RUNS):
        env = BernoulliMABEnv(true_action_probs, Ta, Ra, Rd, Rp, MAX_STEPS=H)
        agent = BernsteinUCBAgent(Ra, Rd, Rp, H, Ta, num_actions=env.action_space.n, c=0.5, INFORMATION_GAIN=INFORMATION_GAIN)
        
        print(f"Running simulation Run {run+1}/{RUNS}...")
        
        rewards = []
        exit = []
        for episode in range(num_episodes):
            obs, info = env.reset()
            terminated = False
            truncated = False
            
            if EXACT_TS:
                t_s = compute_ts(Ta, H, Ra, Rd, Rp, max_prob_exit=max(np.clip(agent.values, 1e-6, 1 - 1e-6)))
                L_ts, U_ts = t_s, None
            else:
                L_ts, U_ts = ts_interval(Ta, H, Ra, Rd, Rp, agent)

            if episode == 0:
                t_s_computed.append([])
            if EXACT_TS:
                t_s_computed[-1].append(int(L_ts))
            else:
                t_s_computed[-1].append((int(L_ts), int(U_ts)))

            step = 0
            last_reward = 0
            while not (terminated or truncated):
                action = agent.select_action(step, L_ts, U_ts)
                reward, terminated, truncated, info = env.step(action)
                
                agent.update(action, terminated)
                
                if terminated or truncated:
                   exit.append(step)
                step += 1
                last_reward += reward

            rewards.append(last_reward)
        run_rewards.append(rewards)
        run_exit.append(exit)

    print(f"Runs completed: {len(run_rewards)}, Episodes per run: {len(run_rewards[0])}")
    print(f"Final Estimated Probabilities (agent.values): {agent.values}")
   
    with open("test_V7_3.json", "w") as f:
        save = {
            "rewards": run_rewards,
            "exit": run_exit
        }
        json.dump(save, f)