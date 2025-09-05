import exploration_strategy
import network
import replay_buffer
import evaluation
from torch.utils.tensorboard import SummaryWriter

# --------------------------------------- #
# function to implement the DQN algorithm #
# --------------------------------------- #    

def run(env, online_model, target_model, optimizer,
        memory, memory_start_size, batch_size, target_update_steps,
        epsilons, gamma, max_episodes, model_file_name):

    writer = SummaryWriter(log_dir="./logs")  # Log directory

    # create a score tracker for statistic purposes
    scores = [];

    # get the action size 
    action_size = env.action_space.n;

    # counter for the number of steps 
    step = 0;

    # save the best score and episode obtained so far
    best_score = -1000;
    best_episode = -1000;

    # update the target model with the online one
    network.update_target(target_model, online_model);

    # train until for the maximum number of episodes
    for episode in range(max_episodes):

        # reset the environment before starting the episode
        state, done = env.reset()[0], False;

        # interact with the environment until the episode is done
        while not done:

            # select the action using the exploration policy
            action = exploration_strategy.epsilon_greedy(epsilons, online_model, state, action_size, step);

            # perform the selected action
            next_state, reward, terminal, truncated, info = env.step(action);
            done = terminal or truncated;
            failure = terminal and not truncated;

            # store the experience into the replay buffer
            experience = (state, action, reward, next_state, failure);
            replay_buffer.store(memory, experience);

            # optimize the online model after the replay buffer is large enough
            if memory['entries'] > memory_start_size:
                network.optimize(memory, batch_size,  online_model, target_model, optimizer, gamma);

            # sometimes, synchronize the target model with the online model
            if step % target_update_steps == 0:
                network.update_target(target_model, online_model);

            # update current state to next state
            state = next_state;

            # update the step counter
            step += 1;

            # sometime print steps 
            if(step % 10 == 0):
                message = 'Episode {:03}, steps {:04}';
                message = message.format(episode+1, step);
                print(message, end='\r', flush=True);

        # After each episode, evaluate the policy
        score = evaluation.run(env, online_model, episodes=1);

        # in case the score is the best so far, store it
        if(score > best_score):
            best_score = score;
            best_episode = episode;
            network.save(online_model, model_file_name);

        # Log the loss to TensorBoard
        writer.add_scalar("Reward", score, episode)

        # store the score in the tracker
        scores.append(score);

        # print some informative logging 
        message = 'Episode {:03}, steps {:04}, current score {:05.1f}, best score {:05.1f} at episode {:03}';
        message = message.format(episode+1, step, score, best_score, best_episode);
        print(message, end='\r', flush=True);

    return scores;