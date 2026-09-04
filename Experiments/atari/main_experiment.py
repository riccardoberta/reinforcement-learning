# import the necessary libraries
import network
import environment
import replay_buffer
import dqn
import numpy as np
import evaluation
import exploration_strategy
import matplotlib.pyplot as plt
import random

# set the seed for reproducibility
seed = 42;

np.random.seed(seed);
random.seed(seed);

# set the name of the Atari environment
env_name = 'ALE/Pong-v5';

# Set the backend to be used for the computation
network.init_torch(seed);

# create the environment
env = environment.make_atari_env(env_name, seed=seed);

# set the hyperparameters
max_episodes = 1000;
memory_size = 1000000; 
memory_start_size = 5000;
batch_size = 32; 
gamma = 0.99;
target_update_steps = 10000;
epsilon_max = 1.0;
epsilon_min = 0.1;
epsilon_decay_steps = 1000000;
learning_rate = 0.0001;
model_file_name = './best_model.pth';

# get the state and action sizes
state_size = env.observation_space.shape;
action_size = env.action_space.n;

# create the experience type
experience_type = replay_buffer.create_type(state_size);

# create the replay memory
memory = replay_buffer.initialize(memory_size, experience_type);

# create the online and target networks
online_model = network.create(state_size, action_size);
target_model = network.create(state_size, action_size);

# create the optimizer
online_optimizer = network.set_optimizer(online_model,learning_rate);

# create the epsilon values
epsilons = exploration_strategy.create_epsilon_values(epsilon_max, epsilon_min, epsilon_decay_steps);

# run the DQN algorithm
scores = dqn.run(env, online_model, target_model, online_optimizer,
                 memory, memory_start_size, batch_size, target_update_steps, 
                 epsilons, gamma, max_episodes, model_file_name);

# smooth the result using a sliding window
sliding_windows = 20
scores = np.convolve(scores, np.ones(sliding_windows)/sliding_windows, mode='valid');

# Show the moving average reward during the training process
plt.figure(figsize=(12,6))
plt.plot(scores, linewidth=1)
plt.title('Moving Avg Score')
plt.ylabel('Score')
plt.xlabel('Episodes')
plt.show()
