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

# define the batch size
batch_size = 1024;

# define the experience data type
experience_type = np.dtype([
    ('state',      np.float32, state_size),   # current state
    ('action',     np.int8),                  # action taken
    ('reward',     np.float32),               # reward received
    ('next_state', np.float32, state_size),   # next state
    ('failure',    np.int8)                   # terminal flag (1 if done)
])

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

# create the neural network model
def create_network(first_hidden_layer, second_hidden_layer):
      dnn = torch.nn.Sequential( 
            torch.nn.Linear(state_size[0], first_hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(first_hidden_layer, second_hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(second_hidden_layer, action_size)
      )
      return dnn;

# create the Q-network
q = create_network(512, 128);
learning_rate = 0.001;
optimizer = torch.optim.RMSprop(q.parameters(), lr=learning_rate);

# define the NFQ policy
def nfq_pi(state):
    state = torch.as_tensor(state, dtype=torch.float32)
    q_values = q(state).detach().numpy().squeeze()
    action = int(np.argmax(q_values))
    return action

# Exploration vs exploitation parameter
epsilon = 0.5

# define the epsilon-greedy policy
def epsilon_greedy(state):
    if random.random() < epsilon:
        action = random_pi(state)
    else:
        action = nfq_pi(state)
    return action

# define the discount factor
gamma = 0.99;

# define the number of optimization epochs
epochs = 10;

# function to optimize the Q-network using a batch of experiences
def optimize(batch):
    states      = torch.from_numpy(batch['state'].copy()).float()        
    actions     = torch.from_numpy(batch['action'].copy()).long()        
    rewards     = torch.from_numpy(batch['reward'].copy() ).float()       
    next_states = torch.from_numpy(batch['next_state'].copy()).float()   
    failures    = torch.from_numpy(batch['failure'].copy()).float()      
    for epoch in range(epochs):
        with torch.no_grad():
            q_next = q(next_states);               
            max_q_next = q_next.max(dim=1).values;
            target = rewards + gamma * max_q_next * (1.0 - failures);
        q_current_all = q(states);
        q_current = q_current_all.gather(1, actions.unsqueeze(1)).squeeze(1);
        td_error = target - q_current;
        loss = (td_error ** 2).mean();
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    return loss;
                
# function to run the NFQ algorithm
def nfq(max_episodes):
    index = 0;
    scores = [];
    experiences_batch = np.empty(batch_size, dtype=experience_type);
    for episode in range(max_episodes):
        state, _ = env.reset();
        done = False;
        while not done:
            action = epsilon_greedy(state);
            next_state, reward, terminal, truncated, _ = env.step(action);
            done = terminal or truncated;
            failure = terminal and not truncated;
            experiences_batch[index] = (state, action, reward, next_state, failure);
            index += 1;
            if index == len(experiences_batch):
                optimize(experiences_batch);
                index = 0;
                experiences_batch = np.empty(shape=(len(experiences_batch),), dtype=experience_type);
            state = next_state;
        score = evaluate(nfq_pi, episodes=10);
        scores.append(score);
        message = 'Episode {:03}, score {:05.1f}';
        message = message.format(episode+1, score);
        print(message, end='\r', flush=True);        
    return scores;

# function to run a complete experiment with multiple seeds
def experiment(learning_rate, first_hidden_layer, second_hidden_layer, max_episodes):
    global q, optimizer;
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
        q = create_network(first_hidden_layer, second_hidden_layer);
        optimizer = torch.optim.RMSprop(q.parameters(), lr=learning_rate);
        scores = nfq(max_episodes);
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
    global gamma, batch_size, epochs, epsilon;
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
        batch_size=params["batch_size"]
        epochs=params["epochs"]
        epsilon=params["epsilon"]
        exp_res = experiment(            
            learning_rate=params["learning_rate"],
            first_hidden_layer=params["first_hidden_layer"],
            second_hidden_layer=params["second_hidden_layer"],
            max_episodes=max_episodes
        )
        mean_score = exp_res["mean_score"]
        min_score = exp_res["min_score"]
        max_score = exp_res["max_score"]      
        avg_mean = float(np.mean(mean_score))
        final_mean = float(mean_score[-1])
        title_str = (
            f"NFQ Learning Performance (Exp {idx}/{total})\n"
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

# run the grid of experiments
summary = run_grid_experiments(param_grid, max_episodes=1500)
