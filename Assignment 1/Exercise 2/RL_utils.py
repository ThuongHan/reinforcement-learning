import numpy as np

def generate_random_policy(gridworld, deterministic=False, seed=None):
    """
    Randomly generates a deterministic or probabilistic policy pi(a|s).

    Returns:
        ndarray: 2-dimensional array of size (N x 4), where N is the number of states of gridworld.
        The 4 columns correspond, respectively, to the actions "U", "D", "L", "R".
    """

    if seed is not None:
        np.random.seed(seed)
    
    if deterministic:
        probs = np.zeros((gridworld.nstates, 4))
        probs[np.arange(gridworld.nstates), np.random.randint(4, size=(gridworld.nstates,))] = 1.0
    else:
        logits = np.random.normal(size=(gridworld.nstates, 4), scale=1)
        exp = np.exp(logits)
        probs = (exp.T / np.sum(exp, axis=1)).T

    # Setting the state values of the terminal states to zero
    for s in gridworld.special_states.keys():
        if gridworld.special_states[s].terminal:
            probs[s] = [0.0, 0.0, 0.0, 0.0]

    return(probs)

def generate_random_v(gridworld, seed=None):
    """
    Randomly generates a state value function.

    Returns:
        ndarray: 1-dimensional array of size N, where N is the number of states of gridworld. Each
        element is the value of that state.
    """

    if seed is not None:
        np.random.seed(seed)
    
    v_ = np.random.normal(size=(gridworld.nstates,), scale=10)
    
    # Setting the state values of the terminal states to zero
    for s in gridworld.special_states.keys():
        if gridworld.special_states[s].terminal:
            v_[s] = 0.0

    return v_

def greedy_policy(gridworld, v, gamma=0.95, use_argmax=True):
    """
    Computation of greedy policy, given a state-value function.

    Returns:
        policy (ndarray): 2-dimensional array of size (N x 4), where N is the number of states of gridworld.
        The 4 columns correspond, respectively, to the actions "U", "D", "L", "R".
    """

    # Action-value function Q(s,a) is computed using the Bellman optimality equation
    q = np.zeros((gridworld.nstates, 4))
    for s in gridworld.states:
        for i, a in enumerate(gridworld.actions):
            for (s_prime, r) in gridworld.p[(s, a)]:

                q[s, i] += gridworld.p[(s, a)][(s_prime, r)] * (r + gamma*v[s_prime])
    
    # Computation of greedy policy
    policy = np.zeros_like(q)
    if use_argmax:
        policy[np.arange(gridworld.nstates), np.argmax(q, axis=1)] = 1.0
    else:
        states_, actions_ = np.nonzero((q.T == np.max(q, axis=1)).T)
        for s in gridworld.states:

            policy[s, np.random.choice(actions_[states_ == s])] = 1.0

    return policy