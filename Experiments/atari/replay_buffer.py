import numpy as np

# ---------------------------------------#
# function to create the experience type #
# ---------------------------------------#

def create_type(state_size):
    experience_type = np.dtype([
        ('state', np.float32, (state_size)), 
        ('action', np.int32),                  
        ('reward', np.float32),               
        ('next_state', np.float32, (state_size)),  
        ('failure', np.int8)                  
    ])
    return experience_type;

# -------------------------------------#
# function to create the replay memory #
# -------------------------------------#

def initialize(memory_size, experience_type):
    replay_memory = {
        'size': memory_size,
        'buffer': np.empty(shape=(memory_size,), dtype=experience_type),
        'index': 0,
        'entries': 0
    }
    return replay_memory;

# -----------------------------------------------------#
# function to store an experience in the replay memory #
# -----------------------------------------------------#

def store(replay_memory, experience):
    # store the experience in the buffer
    replay_memory['buffer'][replay_memory['index']] = experience;

    # update the number of experiences in the buffer
    replay_memory['entries'] = min(replay_memory['entries'] + 1, replay_memory['size']);

    # update index, if the memory is full, start from the begging 
    replay_memory['index'] += 1;
    replay_memory['index'] = replay_memory['index'] % replay_memory['size'];

# -----------------------------------------------------------------#
# function to sample a batch of experiences from the replay memory #
# -----------------------------------------------------------------#
 
def sample(replay_memory, batch_size):
    # select uniformly at random a batch of experiences from the memory
    idxs = np.random.choice(range(replay_memory['entries']), batch_size, replace=False); 

    # return the batch of experiences
    experiences = replay_memory['buffer'][idxs];

    return experiences;
