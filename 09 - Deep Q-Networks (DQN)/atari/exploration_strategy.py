import numpy as np
import random
import network

# ----------------------------------------------#
# funciton to pre-calculated the epsilon values #
# ----------------------------------------------#

def create_epsilon_values(max, min, epsilon_decay_steps):
    # create the epsilons dictionary
    epsilons = {
        'max': max,
        'min': min,
        'decay_steps': epsilon_decay_steps,
        'values': np.logspace(start=0, stop=-2, num=epsilon_decay_steps, base=10)
    };
   
    # normalize the epsilons 
    epsilons['values'] = (epsilons['values'] - epsilons['values'].min()) / (epsilons['values'].max() - epsilons['values'].min());
    
    # scale the epsilons to the desired range
    epsilons['values'] = (max - min) * epsilons['values'] + min;

    return epsilons;

# --------------------------------------------------------------#
# function to implement the epsilon-greedy exploration strategy #
# --------------------------------------------------------------#

def epsilon_greedy(epsilons, model, state, action_size, step):
    # get the epsilon value    
    epsilon = epsilons['values'][step] if step < epsilons['decay_steps'] else epsilons['min'];

    # select the action using the epsilon-greedy strategy
    if random.uniform(0,1) < epsilon:
        return np.random.randint(action_size);
    else:
        return network.act(state, model);