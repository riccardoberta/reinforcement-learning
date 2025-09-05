import torch
import replay_buffer
import numpy as np

# ------------------------------------------ #
# A function to init torch with acceleration #
# ------------------------------------------ #
def init_torch(seed = None):
    # make device available globally
    global device;

    # set the seed for reproducibility 
    if seed is not None:
        torch.manual_seed(seed);
    
    # set the backend device to MPS, if available
    if torch.backends.mps.is_available():
        device = torch.device("mps");
    else:
        device = torch.device("cpu");

    # print the used device    
    print(f"Using device: {device}");

# ------------------------------------------------------------ #
# A function to create a network to approximate the Q-function #
# the network is a simple CNN with three convolutional layers  #
# followed by two fully connected layers                       #
# the input of the network is a stack of 4 frames              #
# the output of the network is the Q-values for each action    #
# ------------------------------------------------------------ #

def create(input_size, output_size):
    # create a sequential model
    dnn = torch.nn.Sequential(
          #conlutional layers
          torch.nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
          torch.nn.ReLU(),
          torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
          torch.nn.ReLU(),
          torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
          torch.nn.ReLU(),

          # flatten the output
          torch.nn.Flatten(),
          
          # fully connected layers
          torch.nn.Linear(7 * 7 * 64, 512),
          torch.nn.ReLU(),
          torch.nn.Linear(512, output_size));

    # return the network
    return dnn.to(device);

# ------------------------------------------------#
# A function to set the optimizer for the network #
# ------------------------------------------------#

def set_optimizer(model, learning_rate):
    optimizer = torch.optim.RMSprop(model.parameters(), 
                                    lr=learning_rate,   # Learning rate
                                    momentum=0.95,      # Momentum
                                    eps=0.01,           # Epsilon for numerical stability
                                    alpha=0.99);        # Smoothing constant (default: 0.99))
    return optimizer;

# ----------------------------------------------------------------#
# A function to update the target network with the online network #
# ----------------------------------------------------------------#

def update_target(target_model, online_model):
    # copy the parameters from the online model to the target model
    for target, online in zip(target_model.parameters(), online_model.parameters()):
        target.data.copy_(online.data);

# --------------------------------------------------- #
# function to make an optimization step using a batch # 
# of experiences from the replay memory
# --------------------------------------------------- #

def optimize(memory, batch_size, online_model, target_model, optimizer, gamma):
    # sample a batch of experiences
    batch = replay_buffer.sample(memory, batch_size);

    # prepare the experience as tensors
    states = torch.from_numpy(batch['state'].copy()).float().to(device);
    actions = torch.from_numpy(batch['action'].copy()).long().to(device);    
    next_states = torch.from_numpy(batch['next_state'].copy()).float().to(device);
    rewards = torch.from_numpy(batch['reward'].copy()).float().to(device);
    failures = torch.from_numpy(batch['failure'].copy()).long().to(device);

    # get the values of the Q-function at next state from the "target" network 
    # remember to detach, we need to treat these values as constants 
    q_target_next = target_model(next_states).detach();

    # get the max value 
    max_q_target_next = q_target_next.max(1)[0];

    # one important step, often overlooked, is to ensure 
    # that failure states are grounded to zero
    max_q_target_next *= (1 - failures.float())

    # calculate the target 
    target = rewards + gamma * max_q_target_next;

    # finally, we get the current estimate of Q(s,a)
    # here we query the current "online" network
    q_online_current = torch.gather(online_model(states), 1, actions.unsqueeze(1)).squeeze();

    # create the errors
    td_error = target - q_online_current;

    # calculate the loss
    loss = td_error.pow(2).mean();
    
    # backward pass: compute the gradients
    optimizer.zero_grad();
    loss.backward();

    # update model parameters
    optimizer.step();

    return loss.detach().cpu().numpy();

# ---------------------------------------------------------#
# A function to act greedily with respect  to the Q-values #
# ---------------------------------------------------------#

def act(state, model):
    # convert the state into a tensor
    state = np.array(state);
    state = torch.from_numpy(state).unsqueeze(0).to(device);

    # calculate q_values from the network
    q_values = model(state).detach();

    # act greedy
    action = np.argmax(q_values.cpu()).data.numpy();

    # return the action
    return action;


# --------------------------------------#
# A funcion to save the model to a file #
# --------------------------------------#

def save(model, path):
    torch.save(model.state_dict(), path);

# -----------------------------------------#
# A function to load the model from a file #
# -----------------------------------------#

def load(model, path):
    model.load_state_dict(torch.load(path));
    return model;

