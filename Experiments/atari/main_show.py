# import the necessary libraries
import matplotlib.pyplot as plt
import network
import environment
import evaluation
import time

# model file name
model_file_name = './best_model.pth';

# set the name of the Atari environment
env_name = 'ALE/Pong-v5';

# create the environment
env = environment.make_atari_env(env_name, mode='human');

# get the state and action sizes
state_size = env.observation_space.shape;
action_size = env.action_space.n;

# create the model
network.init_torch();
model = network.create(state_size, action_size);

# Set the backend to be used for the computation
network.init_torch();

# load the model
network.load(model, model_file_name);

# show the policy in action
evaluation.show(env, model, n_episodes=1, max_steps=5000);
