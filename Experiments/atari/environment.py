import gymnasium as gym
import ale_py
import stable_baselines3
import numpy as np
gym.register_envs(ale_py)

# ------------------------------------------#
# A function to create an Atari environment #
# with some pre-processing                  #
# ------------------------------------------#

def make_atari_env(env_name, mode="rgb_array", seed=None):

    # create the environment
    env = gym.make(env_name, render_mode=mode, frameskip=1);

    if seed is not None:
        env.np_random = np.random.Generator(np.random.PCG64(seed));

    # apply the fire reset wrapper
    env = stable_baselines3.common.atari_wrappers.FireResetEnv(env);

    # apply a wrapper to skip frames, scale the observation and convert it to grayscale
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=True); 

    # apply a wrapper to stack the frames
    env = gym.wrappers.FrameStack(env, 4);

    # return the environment
    return env;
