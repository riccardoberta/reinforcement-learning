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

# xxx
def select_action(state, greedy=False):
    state = torch.as_tensor(state, dtype=torch.float32)
    prefs = pi(state)
    dist = torch.distributions.Categorical(logits=prefs)
    if greedy:
        action = torch.argmax(prefs, dim=-1)
    else:
        action = dist.sample()
    logpa = dist.log_prob(action)
    return int(action.item()), logpa

gamma = 0.99;

# function to optimize the Q-network using a batch of experiences
def optimize(rewards, logpas, discounts):
    rewards   = torch.as_tensor(rewards, dtype=torch.float32)
    discounts = torch.as_tensor(discounts, dtype=torch.float32)
    steps = rewards.shape[0]
    causal_return = torch.empty(steps, dtype=torch.float32)
    for step in range(steps):
        causal_return[step] = torch.sum(discounts[:steps - step] * rewards[step:])
    logpas = torch.stack(logpas).squeeze(-1)
    loss = -(causal_return * logpas).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
                
# function to run the REINFORCE algorithm
def reinforce(gamma, max_episodes):
    max_steps = 1000;
    discounts = gamma ** np.arange(max_steps)
    scores = []; 
    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False;
        logpas = []
        rewards = []
        while not done:
            action, logpa = select_action(state);
            next_state, reward, terminal, truncated, _ = env.step(action);
            done = terminal or truncated
            logpas.append(logpa)
            rewards.append(reward)
            state = next_state;
        optimize(rewards, logpas, discounts);
        score = evaluate(episodes=10);
        scores.append(score);
        message = 'Episode {:03}, score {:05.1f}';
        message = message.format(episode+1, score);
        print(message, end='\r', flush=True);    
    return scores

# function to run a complete experiment with multiple seeds
def experiment(gamma, learning_rate, first_hidden_layer, second_hidden_layer, max_episodes):
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
        scores = reinforce(gamma, max_episodes);
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
        exp_res = experiment(
            gamma=params["gamma"],
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
        plt.savefig('./results/' + filename, dpi=300, bbox_inches="tight")
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
    "first_hidden_layer": [128, 512],
    "second_hidden_layer": [64, 128]
}

# run the grid of experiments
summary = run_grid_experiments(param_grid, max_episodes=1000)
