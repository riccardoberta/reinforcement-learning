import numpy as np
import network

# -------------------------------------------------#
# function to evaluate a policy over some episodes # 
# -------------------------------------------------#

def run(env, model, episodes=1):

    # create a list to store the returns for each episode
    returns = [];

    # evaluate the policy for the given number of episodes
    for episode in range(episodes):

        # reset the environment before starting the episode
        state, done = env.reset()[0], False;

        # initialize the return
        returns.append(0);
        
        # interact with the environment until the episode is done
        while not done:

            # select the action using the policy
            action = network.act(state, model);
            
            # perform the selected action
            state, reward, terminal, truncated, info = env.step(action)
            
            # add the reward to the return
            returns[-1] += reward;

            # check if the episode is done
            done = terminal or truncated;

    # return the average return        
    return np.mean(returns);

# ---------------------------------------------------------#
# function to show the policy in action on the environment #
# ---------------------------------------------------------#

def show(env, model, n_episodes=1, max_steps=500):
    # show the policy for the given number of episodes
    for _ in range(n_episodes):

        # reset the environment
        state = env.reset()[0];

        # reset the done flag and the step counter
        done = False;
        step = 0;

        # interact with the environment until the episode is done
        while not done:

            # select the action using the policy
            action = network.act(state, model);

            # perform the selected action
            state, reward, terminated, truncated, info = env.step(action);

            # render the environment
            env.render();

            # check if the episode is done  
            if(terminated or truncated): done = True;

            # break if the maximum number of steps is reached
            if step > max_steps: break;

            # update the step counter
            step += 1;