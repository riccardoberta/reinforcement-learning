import gymnasium;
import torch;
import numpy as np;
import random;
import matplotlib.pyplot as plt
import itertools

# create the environment
env = gymnasium.make('CartPole-v1', render_mode="rgb_array");
state_size = env.observation_space.shape;
action_size = env.action_space.n;

# define the experience data type
experience_type = np.dtype([
    ('state',      np.float32, state_size),   # current state
    ('action',     np.int8),                  # action taken
    ('reward',     np.float32),               # reward received
    ('next_state', np.float32, state_size),   # next state
    ('failure',    np.int8)                   # terminal flag (1 if done)
])

# Set the replay memory size
memory_size = 100000; 

# Create the replay memory
replay_memory = {
    'size': memory_size,
    'buffer': np.empty(shape=(memory_size,), dtype=experience_type),
    'index': 0,
    'entries': 0
}

# function to store an experience in the replay memory
def store_experience(experience):
    replay_memory['buffer'][replay_memory['index']] = experience;
    replay_memory['entries'] = min(replay_memory['entries'] + 1, replay_memory['size']);
    replay_memory['index'] += 1;
    replay_memory['index'] = replay_memory['index'] % replay_memory['size'];

# Set the batch size for sampling experiences
batch_size = 32;

# function to sample a batch of experiences from the replay memory
def sample_experiences():
    idxs = np.random.choice(range(replay_memory['entries']), batch_size, replace=False);
    experiences = replay_memory['buffer'][idxs];
    return experiences;

# define a random policy for exploration
def random_pi(state):
    return env.action_space.sample()

# function to evaluate a policy
def evaluate(pi, episodes=1):
    rewards = [];
    for episode in range(episodes):
        state, _ = env.reset();
        done = False;
        total_reward = 0.0
        while not done:
            action = pi(state);
            state, reward, terminal, truncated, _ = env.step(action)
            total_reward += reward;
            done = terminal or truncated;
        rewards.append(total_reward);
    return float(np.mean(rewards))

# define the neural network architecture
first_hidden_layer = 512;
second_hidden_layer = 128;

# create the neural network model
def create_network():
      dnn = torch.nn.Sequential( 
            torch.nn.Linear(state_size[0], first_hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(first_hidden_layer, second_hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(second_hidden_layer, action_size)
      )
      return dnn;

# create the Q-network and the target Q-network
online_q = create_network();
target_q = create_network();
learning_rate = 0.007;
optimizer = torch.optim.RMSprop(online_q.parameters(), lr=learning_rate);

# function to update the target Q-network
def update_target():
    for target, online in zip(target_q.parameters(), online_q.parameters()):
        target.data.copy_(online.data)

# define the DQN policy
def dqn_pi(state):
    state = torch.as_tensor(state, dtype=torch.float32)
    q_values = online_q(state).detach().numpy().squeeze()
    action = int(np.argmax(q_values))
    return action

# define decay parameters (max, min, steps)
epsilon_max = 1.0;
epsilon_min = 0.01;
epsilon_decay_steps = 10000;

# generate epsilons
epsilons = np.logspace(start=0, stop=-2, num=epsilon_decay_steps, base=10)
epsilons = (epsilons - epsilon_min) / (epsilon_max - epsilon_min);
epsilons = (epsilon_max - epsilon_min) * epsilons + epsilon_min;

def epsilon_greedy(state, step):
    epsilon = epsilons[step] if step < epsilon_decay_steps else epsilon_min;
    if random.random() < epsilon:
        action = random_pi(state)
    else:
        action = dqn_pi(state)
    return action;

# define the discount factor
gamma = 0.99;

# define the number of optimization epochs
epochs = 10;

# function to optimize the Q-network using a batch of experiences
def optimize():
    batch = sample_experiences();
    states      = torch.from_numpy(batch['state'].copy()).float();        
    actions     = torch.from_numpy(batch['action'].copy()).long();        
    rewards     = torch.from_numpy(batch['reward'].copy()).float();       
    next_states = torch.from_numpy(batch['next_state'].copy()).float();   
    failures    = torch.from_numpy(batch['failure'].copy()).float();  
    q_target_next = target_q(next_states).detach();
    max_q_target_next = q_target_next.max(1)[0];
    max_q_target_next *= (1 - failures.float())
    target = rewards + gamma * max_q_target_next;
    q_online_current = torch.gather(online_q(states), 1, actions.unsqueeze(1)).squeeze(1);
    td_error = target - q_online_current;
    loss = td_error.pow(2).mean();
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

# define the size of the replay memory before starting the training
memory_start_size = 1000;

# define the number of steps between target network updates
target_update_steps = 10;

# function to run the DQN algorithm
def dqn(max_episodes):
    scores = [];
    step = 0;
    update_target();
    for episode in range(max_episodes):
        state, _ = env.reset();
        done = False;
        while not done:
            action = epsilon_greedy(state, step);
            next_state, reward, terminal, truncated, _ = env.step(action);
            done = terminal or truncated;
            failure = terminal and not truncated;
            experience = (state, action, reward, next_state, failure);
            store_experience(experience);
            if replay_memory['entries'] > memory_start_size:
                optimize();
                if step % target_update_steps == 0:
                    update_target();
            state = next_state;
            step += 1;
        score = evaluate(dqn_pi, episodes=10);
        scores.append(score);
        message = 'Episode {:03}, score {:05.1f}';
        message = message.format(episode+1, score);
        print(message, end='\r', flush=True);
    return scores;

# function to run a complete experiment with multiple seeds
def experiment(max_episodes):
    global online_q, target_q, optimizer, replay_memory, epsilons;
    seeds = (12, 34, 56, 78, 90);
    results = [];
    for seed in seeds:
        print("Experiment seed: ", seed);
        torch.manual_seed(seed);
        np.random.seed(seed);
        random.seed(seed);
        env.reset(seed=seed);
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        online_q = create_network();
        target_q = create_network();
        optimizer = torch.optim.RMSprop(online_q.parameters(), lr=learning_rate);
        replay_memory = {
            'size': memory_size,
            'buffer': np.empty(shape=(memory_size,), dtype=experience_type),
            'index': 0,
            'entries': 0
        }
        epsilons = np.logspace(start=0, stop=-2, num=epsilon_decay_steps, base=10)
        epsilons = (epsilons - epsilon_min) / (epsilon_max - epsilon_min);
        epsilons = (epsilon_max - epsilon_min) * epsilons + epsilon_min;    
        scores = dqn(max_episodes);
        sliding_windows = 25;
        scores = np.convolve(scores, np.ones(sliding_windows)/sliding_windows, mode='valid');
        results.append(scores);
        print("");
    max_score = np.max(results, axis=0).T;
    min_score = np.min(results, axis=0).T;
    mean_score = np.mean(results, axis=0).T;
    experiment_results = {
        'max_score': max_score,
        'min_score': min_score,
        'mean_score': mean_score
    }    
    return experiment_results;

# function to run a grid of experiments with different hyperparameters
def run_grid_experiments(param_grid, max_episodes):

    global gamma, memory_size, memory_start_size
    global batch_size, target_update_steps, first_hidden_layer
    global second_hidden_layer, learning_rate
    global epsilon_max, epsilon_min, epsilon_decay_steps

    keys = list(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    total = len(combos)
    print(f"Running {total} experiments...")
    summary = []
    for idx, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        print(f"\nExperiment {idx}/{total}")
        for k, v in params.items():
            print(f"   {k:>20s} = {v}")
        print("")

        gamma=params["gamma"]
        memory_size=params["memory_size"]
        memory_start_size=params["memory_start_size"]
        batch_size=params["batch_size"]
        target_update_steps=params["target_update_steps"]
        first_hidden_layer=params["first_hidden_layer"]
        second_hidden_layer=params["second_hidden_layer"]
        learning_rate=params["learning_rate"]
        epsilon_max=params["epsilon_max"]
        epsilon_min=params["epsilon_min"]
        epsilon_decay_steps=params["epsilon_decay_steps"]

        exp_res = experiment(max_episodes=max_episodes)

        mean_score = exp_res["mean_score"]
        min_score = exp_res["min_score"]
        max_score = exp_res["max_score"]      
        avg_mean = float(np.mean(mean_score))
        final_mean = float(mean_score[-1])
        title_str = (
            f"DQN Learning Performance (Exp {idx}/{total})\n"
            f"γ={params['gamma']}, lr={params['learning_rate']}, "
            f"bs={params['batch_size']}, ep={params['epochs']}, "
            f"ε={params['epsilon']}, h1={params['first_hidden_layer']}, "
            f"h2={params['second_hidden_layer']}"
        )
        plt.figure(figsize=(12, 6))
        plt.title(title_str, fontsize=10)
        plt.ylabel('Score')
        plt.xlabel('Episodes')
        episodes = range(len(mean_score))
        plt.plot(mean_score, color='orange', linewidth=2, label='Mean score')
        plt.fill_between(episodes, min_score, max_score, color='orange', alpha=0.3, label='Range (min–max)')
        plt.legend()
        plt.tight_layout()
        filename = (f"nfq_gamma{params['gamma']}_"
                    f"lr{params['learning_rate']}_"
                    f"bs{params['batch_size']}_"
                    f"ep{params['epochs']}_"
                    f"eps{params['epsilon']}_"
                    f"h1_{params['first_hidden_layer']}_"
                    f"h2_{params['second_hidden_layer']}.png").replace('.', '_')
        plt.savefig('./nfq_results/' + filename, dpi=300, bbox_inches="tight")
        plt.close()
        summary.append({
            "index": idx,
            **params,
            "avg_mean": avg_mean,
            "final_mean": final_mean,
        })
    summary.sort(key=lambda r: (r["avg_mean"], r["final_mean"]), reverse=True)
    print("\nAll experiments completed.")
    print(f"Best configuration (by average mean score): Experiment {summary[0]['index']}")
    print(summary[0])
    return summary

# define the hyperparameter grid
param_grid = {
    "gamma": [0.99, 1.00],
    "learning_rate": [0.0005, 0.001, 0.005],
    "batch_size": [512, 1024, 2048],
    "epochs": [16, 32, 64],
    "epsilon": [0.5, 0.7],
    "first_hidden_layer": [128, 256],
    "second_hidden_layer": [64, 128]
}

param_grid = {
    "gamma": [0.99],
    "memory_size": [10000, 20000],
    "memory_start_size": [1000],
    "batch_size": [512, 1024, 2048],
    "target_update_steps": [100, 1000],
    "first_hidden_layer": [128],
    "second_hidden_layer": [256],
    "learning_rate": [0.0005, 0.001, 0.005],
    "epsilon_max": [0.7],
    "epsilon_min": [0.2],
    "epsilon_decay_steps": [10000, 20000]
}

# run the grid of experiments
summary = run_grid_experiments(param_grid, max_episodes=1500)
