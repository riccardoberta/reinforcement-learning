# Applied Reinforcement Learning

[Prof. Riccardo Berta](https://about.me/riccardo.berta)

The [course](https://corsi.unige.it/en/off.f/2025/ins/90595), offered within the Master's Degree in [Electronic Engineering](https://corsi.unige.it/en/corsi/11970) at the [University of Genoa](https://unige.it/), focuses on building autonomous agents that learn to act in an unknown environment in order to achieve a goal, using reinforcement learning algorithms.

## Learning Outcomes

The theoretical part of the course introduces the main reinforcement learning algorithms, from tabular methods to deep function approximation. The technical and practical labs give students the ability to implement solutions to real-world problems with the algorithms presented in class, using Python and Jupyter notebooks. The course aims to train professionals capable of designing and developing complex software applications based on artificial intelligence algorithms.

Regular attendance and active participation in the proposed activities (lectures and labs), together with individual study, will allow you to:

- use the correct terminology to describe the behavior of an autonomous agent;
- gain in-depth knowledge of the main reinforcement learning algorithms and critically analyze their differences;
- quantitatively evaluate the performance of an autonomous agent on a given problem;
- formulate a real-world problem in terms of autonomous agents;
- apply the algorithms to real problems using the Python programming language.

## Prerequisites

To effectively engage with the course content, the following background is required: programming skills in a high-level language, preferably Python (to understand and implement the algorithms and exercises); familiarity with probability, linear algebra and calculus (to follow the derivations); and familiarity with supervised machine learning and neural networks (to understand the algorithms based on function approximators).

## Syllabus

The course is organized in the following chapters:

| # | Chapter | Main topics |
|---|---------|-------------|
| 01 | Agent Architectures | PEAS, reactive and deliberative architectures, flocking example |
| 02 | The Reinforcement Learning Problem | Agent–environment interface, Markov Decision Processes, policies, Gymnasium |
| 03 | Dynamic Programming | Value functions, Bellman equations, policy and value iteration |
| 04 | Exploration vs Exploitation | Multi-armed bandits, ε-greedy, optimistic initialization, softmax, UCB, Thompson sampling, regret |
| 05 | Value-based Methods | Monte Carlo and Temporal Difference policy evaluation, SARSA, Q-learning, Double Q-learning |
| 06 | Model-based Methods | Learning a model, planning, Dyna-Q, trajectory sampling, prioritized sweeping |
| 07 | Generalizing with Neural Networks | Neural Fitted Q (NFQ), Deep Q-Network (DQN), Double DQN, Dueling DQN, Prioritized Experience Replay |
| 08 | Policy Gradient Methods | Policy gradient theorem, REINFORCE, baselines, Vanilla Policy Gradient |
| 09 | Actor-Critic Methods | One-step actor-critic, A3C, A2C, Generalized Advantage Estimation |
| 10 | Continuous Control | Stochastic and deterministic policies for continuous actions, DDPG, TD3, SAC |
| 11 | Proximal Policy Optimization | Step-size problem, importance sampling, TRPO, PPO |
| 12 | Offline RL and Decision Transformers | Offline setting, behaviour cloning, Conservative Q-Learning, Decision Transformers |
| 13 | Reinforcement Learning for Language Models | Language models as policies, reward models, RLHF with PPO, DPO, GRPO |

## Repository structure

Each chapter has its own folder (`01 - Agent Architectures`, `02 - The RL Problem`, ...) containing some or all of the following files:

- `notes.ipynb` — the lecture notes: intuition first, then the mathematical treatment, then the implementation, then its application to Gymnasium environments. A `notes.pdf` export is provided for offline reading.
- `exercises.ipynb` — exercises on the chapter topics (with a `exercises.pdf` export where available).
- `lab.ipynb` — the hands-on lab session for the chapter.
- `slides.pptx` — the slides used during the lectures.
- `papers/` — the original research papers referenced in the chapter.
- `drawings.excalidraw` — the source of the figures used in the notes and slides.

The notebook `Notation.ipynb` in the root folder collects the mathematical notation used consistently throughout the chapters. The slides `00 - Introduction.pptx` present the course organization.

## Running the notebooks

GitHub's notebook viewer does not render some features (interactive plots, animations), is slow, does not always display equations correctly, and large notebooks often fail to open. To play with the code you need to run the notebooks, in one of three ways:

1. **Locally**: clone this repository and start Jupyter (see [Required software packages](#required-software-packages) below). This is the best option if you want to explore the notebooks in read-write mode, modify them and run the code. All the chapters from 07 onward train neural networks: a GPU is not strictly necessary, but it makes the longer experiments (e.g., Atari Pong) considerably faster.

2. **Remotely**: click on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/riccardoberta/autonomous-agents). This opens the notebooks on Google Colab, a free online service that runs Jupyter notebooks (it uses Google Drive as a backend, so you can save your copies there). This is the best option if you want to quickly run the notebooks, as Google Colab comes with most dependencies pre-installed and offers free GPU runtimes. You cannot modify the original notebooks, but you can make a copy in your Google Drive and modify that.

3. **Online**: click on the <a href="https://nbviewer.jupyter.org/github/riccardoberta/autonomous-agents"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="Render nbviewer" /></a>. This opens the notebooks on nbviewer, a free online service that renders notebooks as static web pages. It is similar to GitHub's viewer but renders more features correctly, it is faster, and it also lets you download the notebook. You cannot modify the notebooks or run any code.

## Required software packages

To run the notebooks locally you need:

- [**Conda**](https://docs.conda.io/en/latest/) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) as package and environment manager, and a [**Python**](https://www.python.org/) 3.13 environment:

  ```bash
  conda create -n reinforcement-learning python=3.13
  conda activate reinforcement-learning
  ```

- [**Jupyter**](https://jupyter.org/) to run the notebooks, plus [**NumPy**](https://numpy.org/), [**Matplotlib**](https://matplotlib.org/) and [**joblib**](https://joblib.readthedocs.io/) (used to save and reload training results):

  ```bash
  pip install jupyterlab numpy matplotlib joblib
  ```

- The [**Gymnasium**](https://gymnasium.farama.org/) library (>= 1.0) with the environment families used in the course: toy-text (Frozen Lake, Taxi, Blackjack), classic-control (CartPole, Acrobot, Pendulum), Box2D (Lunar Lander) and Atari (Pong, through [ale-py](https://ale.farama.org/)):

  ```bash
  pip install "gymnasium[toy-text,classic-control,box2d,atari]"
  ```

  Box2D requires the `swig` build tool; if the installation fails, install it first with `conda install swig` (or with your system package manager).

- [**PyTorch**](https://pytorch.org/) (>= 2.0), used from chapter 07 onward for function approximation. Follow the [official instructions](https://pytorch.org/get-started/locally/) to pick the right build for your platform and GPU; the CPU-only build is sufficient for most of the notebooks:

  ```bash
  pip install torch
  ```

## Recommended reading

All the slides, source code, and other teaching materials used during the lectures are available on the Aul@web platform and in this repository. In general, the lecture notes are sufficient for exam preparation. The following books are suggested as supplementary texts, but students can also use other recent university-level books on Reinforcement Learning:

- Richard S. Sutton, Andrew G. Barto, **"Reinforcement Learning: An Introduction"**, 2nd ed., MIT Press (freely available [online](http://incompleteideas.net/book/the-book-2nd.html)).
- Miguel Morales, **"Grokking Deep Reinforcement Learning"**, Manning.

## Tools

The following tools (libraries with optimized implementations of the algorithms, environment collections, online courses) are useful companions to the course material:

- [**Farama Foundation**](https://farama.org/) is the non-profit organization that maintains Gymnasium and a wide set of RL environments.
- [**Gymnasium**](https://gymnasium.farama.org/) is the Python library that provides a collection of environments sharing a uniform interface, so that you can write general algorithms. It makes no assumptions about the structure of the agent and is compatible with any numerical computation library, such as PyTorch or TensorFlow.
- [**Stable Baselines3**](https://github.com/DLR-RM/stable-baselines3) is a set of reliable, optimized implementations of reinforcement learning algorithms in PyTorch.
- [**CleanRL**](https://github.com/vwxyzjn/cleanrl) provides high-quality single-file implementations of the most popular deep reinforcement learning algorithms, with research-friendly features.
- [**stable-retro**](https://stable-retro.farama.org/) provides a collection of classic video games with a Gymnasium-compatible interface (Sega Genesis, Super Nintendo, Atari 2600, and more).
- [**Hugging Face Deep RL Course**](https://huggingface.co/learn/deep-rl-course/unit0/introduction) is a free online course on Deep Reinforcement Learning, and [**TRL**](https://github.com/huggingface/trl) is the Hugging Face library for post-training language models with RLHF, DPO and GRPO (chapter 13).

## Exam description

The exam consists of a written test covering both the theoretical and practical topics presented in class. Students are expected to demonstrate a thorough understanding of the algorithms underlying the development of autonomous agents and the ability to implement them in Python. The exam assesses the comprehension of the concepts, the ability to apply the algorithms, and the practical implementation skills. Details on the preparation methods and on the level of depth required for each topic will be provided during the lectures. The problems and open-ended questions evaluate the students' ability to apply their knowledge in practical situations that may arise in real-world scenarios.
