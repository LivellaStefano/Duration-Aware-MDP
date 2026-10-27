def solve_forced_exit_mdp(Ta, H, actions, Ra=1.0, Rd=-1.0, Rp=0.0):
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
                best_action = i
        
        V[t] = best_expected_reward
        policy[t] = best_action
            
    return V, policy

# Configuration
actions = [ 
    {'p_leave': 0.0001}, 
    {'p_leave': 0.005},
    {'p_leave': 0.007},
    {'p_leave': 0.11},
    {'p_leave': 0.115},
    {'p_leave': 0.12}
]

Ta = 19  # Switch point (tau)
H = 40
Ra = 1.0
Rd = -1.0
Rp = 0.0

# Solve MDP
values, policy = solve_forced_exit_mdp(Ta=Ta, H=H, actions=actions, Ra=Ra, Rd=Rd, Rp=Rp)
print("Expected Rewards:", values)
print("Optimal Policy:", policy)