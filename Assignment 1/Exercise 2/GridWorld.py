import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

class SpecialState(object):
    """Class representing special states in the gridworld, such as terminal states, hard wall states, water states 
    and fire states. Each special state has a location, a name, a reward for transitioning to that state (reward_to) 
    and a reward for transitioning from that state (reward_from). The color attribute is used for plotting purposes. 
    The terminal attribute indicates whether the state is a terminal state or not.
    """
    def __init__(self, location, name, reward_to=None, reward_from=None, color=None, terminal=False):

        self.location = location
        self.name = name
        self.reward_to = reward_to
        self.reward_from = reward_from
        self.color = color
        self.terminal = terminal

    def add(self, gw):
        assert self.location not in gw.special_states.keys(), \
            'The location of the special state {} is already occupied by another special state. Please choose a different location for the special state.'.format(self.name)
        gw.special_states[self.location] = self
        gw.update_p()
    
    def remove(self, gw):
        assert self.location in gw.special_states.keys(), \
            'The location of the special state {} is not occupied by any special state. Please choose a different location for the special state.'.format(self.name)
        del gw.special_states[self.location]
        gw.update_p()

class TerminalState(SpecialState):
    """
    Class representing terminal states in the gridworld. Terminal states are absorbing states, meaning that once the agent
    transitions to a terminal state, it remains in that state and receives the same reward for every subsequent transition.
    """
    def __init__(self, location, name='terminal', reward_from=0.0, color='red'):

        super().__init__(location=location, name=name, reward_from=reward_from, color=color, terminal=True)
        
class HardWallState(SpecialState):
    """
    Class representing hard wall states in the gridworld. Hard wall states are absorbing states, meaning that
    once the agent transitions to a hard wall state, it remains in that state and receives the same reward for every subsequent transition.
    """
    def __init__(self, location, name='hard_wall', color='grey', **kwargs):

        super().__init__(location=location, name=name, color=color, terminal=False, **kwargs)

class WaterState(SpecialState):
    """
    Class representing water states in the gridworld. Water states are terminal states, meaning that once the agent transitions to a water state, it remains in that state and receives the same reward for every subsequent transition.
    """
    def __init__(self, location, name='water', reward_to=10.0, reward_from=0.0, color='lightblue', **kwargs):

        super().__init__(location=location, name=name, reward_to=reward_to, reward_from=reward_from, color=color, terminal=True)

class FireState(SpecialState):
    """
    Class representing fire states in the gridworld. Fire states are terminal states, meaning that once the agent transitions to a fire state, it remains in that state and receives the same reward for every subsequent transition.
    """
    def __init__(self, location, name='fire', reward_to=-10.0, reward_from=0.0, color='orange', **kwargs):

        super().__init__(location=location, name=name, reward_to=reward_to, reward_from=reward_from, color=color, terminal=True)

class GridWorld(object):
    """
    GridWorld object. Default grid_size (5,5). Default terminal states: fire, water.
    """

    def __init__(self, grid_size=(5,5), default_reward=-1.0, special_states=None):
        
        self.grid_size = grid_size
        self.nrows, self.ncolumns = grid_size

        self.nstates = np.prod(grid_size) # number of states
        self.states = np.arange(self.nstates)

        self.actions = ['U', 'D', 'L', 'R']

        self.default_reward = default_reward

        self.special_states = {}
        if special_states is not None:
            for special_state in special_states:
                special_state.add(self)

        self.p = self.update_p(output=True)
    
    def s_to_grid(self, s):
        """
        Convert state s to grid coordinates (row, column).

        Return:
            np.array: A numpy array of the form [row, column] representing the grid coordinates of state s.
        """

        row = s // self.ncolumns
        column = s % self.ncolumns

        return np.array([row, column])
    
    def grid_to_s(self, grid_coordinates):
        """
        Convert grid coordinates (row, column) to state s.

        Return:
            int: An integer representing the state s corresponding to the given grid coordinates.
        """

        row, column = grid_coordinates
        s = row * self.ncolumns + column

        return s
    
    def a_to_grid(self, a):
        """
        Convert action a to grid coordinates (delta_row, delta_column).

        Return:
            np.array: A numpy array of the form [delta_row, delta_column] representing the grid coordinates of action a.
        """

        if a == 'U':
            return np.array([-1, 0])
        elif a == 'D':
            return np.array([1, 0])
        elif a == 'L':
            return np.array([0, -1])
        elif a == 'R':
            return np.array([0, 1])
        else:
            raise ValueError('Invalid action. Please choose one of the following actions: U, D, L, R.')
        
    def _default_s_new(self, s, a):

        s_new_tentative_grid = self.s_to_grid(s) + self.a_to_grid(a)
        if np.any(s_new_tentative_grid < 0) or np.any(s_new_tentative_grid >= self.grid_size):
            s_new = s # forbidden transitions remain in the same state
        else:
            s_new = self.grid_to_s(s_new_tentative_grid)

        return s_new

    def update_p(self, output=False):
        """
        Update conditional probability p(s',r|s, a) for gridworld.

        Return:
            dict: A dictionary with keys (s,a) for all valid combinations of s and a. The values
            of the dictionary are again a dictionary, with keys (s', r) and values p(s',r|s,a).
        """
        p = {}
        for s in self.states:
            for a in self.actions:
                
                s_new = self._default_s_new(s, a)

                if s in self.special_states.keys():
                    if self.special_states[s].terminal or self.special_states[s].name == 'hard_wall': # terminal states and hard wall states are absorbing states
                        if self.special_states[s].reward_from is None:
                            reward_from = self.default_reward
                        else:
                            reward_from = self.special_states[s].reward_from
                        p[(s,a)] = {(s, reward_from) : 1.0}
                elif s_new in self.special_states.keys():
                    if self.special_states[s_new].reward_to is None:
                        reward_to = self.default_reward
                    else:
                        reward_to = self.special_states[s_new].reward_to    
                    if self.special_states[s_new].name == 'hard_wall': # hard wall states
                        p[(s,a)] = {(s, reward_to) : 1.0}
                    elif self.special_states[s_new].name == 'water': # water states are absorbing states
                        p[(s,a)] = {(s_new, reward_to) : 1.0}
                    elif self.special_states[s_new].name == 'fire': # fire states are absorbing states
                        p[(s,a)] = {(s_new, reward_to) : 1.0}
                else:
                    p[(s,a)] = {(s_new, self.default_reward) : 1.0}
        
        self.p = p

        if output:
            return p
        else:
            return None
        
    def s_to_plot_grid(self, s):
        """
        Convert state s to grid coordinates (row, column) for plotting.

        Return:
            np.array: A numpy array of the form [column, row] representing the grid coordinates of state s for plotting.
        """

        # row = self.nrows - 1 - (s // self.ncolumns)
        row = (s // self.ncolumns)
        column = s % self.ncolumns

        return np.array([column, row])
        
    def plot_gridworld(self, ax=None, title=None, print_states=True, color_special_states=True, v=None):
        """
        Plots the gridworld. Each state is a tile, with color corresponding to the value of that state.
        The special states are colored according to their color attribute.
        """

        # Each state is a white, transparent tile
        grid_ = np.ones((*(self.grid_size), 4))
        grid_[:,:,-1] = 0.0

        if v is not None:
            cmap = plt.colormaps.get_cmap('Greens')
            norm = plt.Normalize(v.min(), v.max())
            grid_ = cmap(norm(v)).reshape(*(self.grid_size), 4)

        if color_special_states:
            for s, special_state in self.special_states.items():
                grid_[self.s_to_grid(s)[0], self.s_to_grid(s)[1], :3] = mcolors.to_rgba(special_state.color)[:3]
                grid_[self.s_to_grid(s)[0], self.s_to_grid(s)[1], -1] = 1.0
        
        # Plotting
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(grid_)

        # Title
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title('Gridworld')

        # Minor ticks
        ax.set_xticks(np.arange(-.5, self.ncolumns, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.nrows, 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

        # Remove minor ticks
        ax.tick_params(which='minor', bottom=False, left=False)

        # Remove other ticks and labels
        ax.tick_params(which='major', bottom=False, left=False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        # Print states
        if print_states:
            for s in self.states:
                ax.text(*self.s_to_plot_grid(s), s, ha="center", va="center")

    def plot_v(self, ax=None, v=None, title="State value function", print_states=False, **kwargs):
        """
        Plots the state value function v. Each state is a tile, with color corresponding to the value of that state.
        The value of each state is printed in the middle of the tile."""
        if v is None:
            v = np.zeros(self.nstates)

        # Plotting
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        self.plot_gridworld(ax=ax, title=title, print_states=print_states, v=v, **kwargs)

        for s in self.states:
            ax.text(*self.s_to_plot_grid(s), round(v[s], 2), ha="center", va="center")

    def plot_policy(self, ax=None, policy=None, title="Policy", print_states=False, **kwargs):
        """
        Plots the policy. Each state is a tile, with arrows corresponding to the probabilities of taking each action in that state.
        The value of each state is printed in the middle of the tile."""
        if policy is None:
            policy = np.random.rand(self.nstates, len(self.actions))
            policy = policy / np.sum(policy, axis=1, keepdims=True)

        # Plotting
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        self.plot_gridworld(ax=ax, title=title, print_states=print_states, **kwargs)

        # Print policy
        for s in self.states:
            for direction_idx in range(4):
                prob = policy[s, direction_idx]
                direction = [(0.0, -0.25), (0.0, 0.25), (-0.25, 0), (0.25, 0)][direction_idx] # U, D, L R
                ax.arrow(*self.s_to_plot_grid(s), *direction, head_width=0.1, alpha=prob)

    def plot(self, plots=[('grid',),('v',),('policy',)], plots_per_row=3, output_fig=False):
        """
        Plots multiple aspects of the gridworld in a single figure. The aspects to be plotted are specified 
        in the input argument "plots". Each element of "plots" is a tuple, where the first element is a string 
        specifying what to plot and the second element is a dictionary containing keyword arguments for the 
        corresponding plotting function. The first element of each tuple can be either "grid", "v" or "policy", 
        corresponding to the functions plot_gridworld, plot_v and plot_policy, respectively.
        """

        nplots = len(plots)
        nrows = int(np.ceil(nplots / plots_per_row))
        ncols = min(nplots, plots_per_row)

        fig, ax = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        axes = np.atleast_1d(ax).ravel()

        for plot_descriptor, ax_ in zip(plots, axes):
            
            if len(plot_descriptor) == 1:
                kwargs = {}
            else:
                kwargs = plot_descriptor[1]

            if 'grid' == plot_descriptor[0]:
                self.plot_gridworld(ax=ax_, **kwargs)
            elif 'v' == plot_descriptor[0]:
                self.plot_v(ax=ax_, **kwargs)
            elif 'policy' == plot_descriptor[0]:
                self.plot_policy(ax=ax_, **kwargs)

        if output_fig:
            return fig
        else:
            return None
        
    def plot_overview(self, v, policy):
        """Plots an overview of the gridworld, including the gridworld itself, the state value function and the policy.

        Each aspect is plotted in a separate subplot, but they are all shown in a single figure for better comparison.
        """
        self.plot(plots=[('grid',), ('policy', {'policy': policy}), ('v', {'v': v})])

class WaterFireGridWorld(GridWorld):
    """
    Gridworld with two terminal states: water and fire. The water state gives a positive reward and the fire state 
    gives a negative reward.
    """
    def __init__(self, grid_size=(5,5), default_reward=-1.0, special_states=None):
        
        if special_states is None:
            special_states = [
                WaterState(location=18),
                FireState(location=12)
            ]

        super().__init__(grid_size=grid_size, default_reward=default_reward, special_states=special_states)