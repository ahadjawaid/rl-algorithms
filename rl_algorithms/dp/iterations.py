import numpy as np

def policy_evaluation(policy, P, gamma=1.0, theta=1e-10):
    # Finds the optimal value function for the policy provided
    n_states = len(P)  
    prev_V = np.zeros(n_states, dtype=np.float64)

    value_log, i = {}, 0 
    while True:
        V = np.zeros(n_states, dtype=np.float64)

        for state in range(n_states):
            action = policy[state]

            for prob, next_state, reward, done in P[state][action]:
                V[state] += prob * (reward + gamma * prev_V[next_state])

        value_log[i] = V

        if np.max(np.abs(prev_V - V)) < theta:
            break

        prev_V = V.copy()

    return V, value_log

def policy_improvement(V, P, gamma=1.0):
    n_states, n_actions = len(P), len(P[0])

    Q = np.zeros((n_states, n_actions), dtype=np.float64)

    for state in range(n_states):
        for action in range(n_actions):
            for prob, next_state, reward, done in P[state][action]:
                Q[state][action] += prob * (reward + gamma * V[next_state])

    new_policy =  {s: a for s, a in enumerate(np.argmax(Q, axis=1))}
    
    return new_policy

def policy_iteration(P, gamma=1.0, theta=1e-10):
    n_states = len(P)
    random_actions = np.random.choice(tuple(P[0].keys()), n_states)

    policy = {s: a for s, a in enumerate(random_actions)}


    info, i = {}, 0
    
    while True:
        old_policy = {s: policy[s] for s in range(n_states)}

        V, value_log = policy_evaluation(policy, P, gamma, theta)
        policy = policy_improvement(V, P, gamma)

        info[i] = value_log, policy
        i += 1

        if old_policy == {s: policy[s] for s in range(n_states)}:
            break

    return V, policy, info

def value_iteration(P, gamma=1.0, theta=1e-10):
    # Computes value function by taking max of action value and imporoving iteratively
    n_states, n_actions = len(P), len(P[0])

    V = np.zeros(n_states, dtype=np.float64)
    info, i = {}, 0
    while True:
        Q = np.zeros((n_states, n_actions), dtype=np.float64)

        for state in range(n_states):
            for action in range(n_actions):
                for prob, next_state, reward, done in P[state][action]:
                    Q[state][action] += prob * (reward + gamma * V[next_state])

        policy = {state: action for state, action in enumerate(np.argmax(Q, axis=1))}
        info[i] = np.max(Q, axis=1), policy
        i += 1

        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break

        V = np.max(Q, axis=1)

    policy = {s: a for s, a in enumerate(np.argmax(Q, axis=1))}

    return V, policy, info