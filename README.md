# Applied Reinforcement Learning

[Prof. Riccardo Berta](https://about.me/riccardo.berta)

The course, offered within the context of the Master's Degree in [Electronic Engineering](https://corsi.unige.it/en/corsi/8732) at the [University of Genoa](https://unige.it/), focuses on the topic of creating autonomous agents capable of moving and interacting with an unknown environment in order to achieve a specific goal, using reinforcement learning algorithms.

## Learning Outcomes

The main purpose of the theoretical part of the course is to introduce the main reinforcement learning algorithms, both tabular-based and function approximation-based. The technical and practical labs aim to provide students with the ability to implement solutions to real-world problems using the algorithms presented in class, using the Python language and Jupyter Notebook tool. The course aims to train professionals who are capable of designing and developing complex software applications using artificial intelligence algorithms.
Regular attendance and active participation in the proposed educational activities (lectures and technical-practical labs), along with individual study, will allow you to:

- acquire the correct terminology to adequately describe the behavior of an autonomous agent.
- gain in-depth knowledge of the main reinforcement learning algorithms and be able to critically analyze their differences.
- be capable of quantitatively evaluating the performance of an autonomous agent in a given problem.
- formulate a real-world problem in terms of autonomous agents.
- apply the algorithms to real problems using the Python programming language.

## Prerequisites

To effectively engage with the content of the course, it is necessary to have the following foundational knowledge: programming skills with a high-level programming language, preferably Python (in order to understand and implement the algorithms and exercises covered in the course); and familiarity with the supervised machine learning topic (in order to understand the algorithms based on function approximators).

## Syllabus

The following list outlines the topics covered in the course:

1. Introduction to autonomous agents and reinforcement learning
2. Markov Decision Processes (MDPs)
3. Dynamic Programming
4. Exploration vs Exploitation
5. Policy Evaluation
6. Policy Improvement
7. Model-based methods
8. Neural-fitted Q
9. Value-based methods
10. Policy-based methods
11. Actor-Critic methods
12. Advanced algorithms

By accessing the GitHub repository, you will have access to the lecture materials, including slides and notebooks, to further explore and study the course topics.

Unfortunately, GitHub's notebook viewer does not render some features, such as interactive plots, it's slower, the math equations are not always displayed correctly, and large notebooks often fail to open. So if you want to play with the code, you need to run the notebooks. You can do that in three ways:

1. **Locally**: you can clone this GitHub repository and start Jupyter Notebook. This is the best option if you want to explore the notebooks in read-write mode, modify them and run the code.

2. **Remotely**: you can click on the [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/riccardoberta/reinforcement-learning). This will open the notebook on Google Colab, a free online service that allows you to run Jupyter notebooks (it uses Google Drive as a backend, so you can save your notebooks there). This is the best option if you want to quickly view the notebooks, as Google Colab comes with all the dependencies pre-installed and ready to use. However, you will not be able to modify the notebooks (although you can make a copy and modify that copy in your Google Drive).

3. **Online**: you can click on the <a href="https://nbviewer.jupyter.org/github/riccardoberta/reinforcement-learning"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="Render nbviewer" /></a>. This will open the notebook on nbviewer, a free online service that renders notebooks as static web pages. This is similar to GitHub's notebook viewer, except that it properly renders more features (including interactive plots), it's faster, and it also allows you to download the notebook. However, you will not be able to modify the notebooks or run any code (except for the interactive plots).

## Recommended reading

All the slides, source code, and other teaching materials used during the lectures will be available on the Aul@web platform. In general, the lecture notes and materials on Aul@web are sufficient for exam preparation. The following books are suggested as supplementary texts, but students can also use other recent university-level books on Reinforcement Learning:

- Miguel Morales, **"Grokking Deep Reinforcement Learning"**, Manning.
- Richard S. Sutton, Andrew G. Barto, **"Reinforcement Learning: An Introduction"**, MIT Press.

These books can provide additional insights and resources for further study on the topic of Reinforcement Learning.

## Tools

This list provides link to useful tools (libraries with optimized implementation of algorithms, environments collections, etc.) that can be useful in learning the topics:

- [**Farama Foundation**](https://farama.org/) is a non-profit organization that aims to promote the development of artificial intelligence and robotics.
- [**Gymnasium**](https://gymnasium.farama.org/) is Python library that provides a collection of environments that share uniform interface, allowing you to write general algorithms. It makes no assumptions about the structure of the agent, and is compatible with any numerical computation library, such as TensorFlow or PyTorch
- [**Stable Baselines**](https://github.com/DLR-RM/stable-baselines3) is a set of optimized implementations of reinforcement learning algorithms in PyTorch.
- [**stable-retro**](https://stable-retro.farama.org/) provides a repo of classic video games with Gymnasium compatible interface, supported platforms includes Sega Genesis, Sega 32X, Super Nintendo, Atari 2600 and more.
- [**Hugging Face Deep RL Course**](https://huggingface.co/learn/deep-rl-course/unit0/introduction) is an online course on Deep Reinforcement Learning from Hugging Face.
- [**CleanRL**](https://github.com/vwxyzjn/cleanrl) is a library that provides high-quality single-file implementation with research-friendly features of all the popular reinforcement learning algorithms.

## Required software packages

In order to run locally the notebooks, you need to install the following software packages:

- [**Conda**](https://docs.conda.io/en/latest/) (>= 23.5) package and environment manager

- A [**Python**](https://www.python.org/) (11.0) environment:

`
conda create -n reinforcement-learning python=3.11
conda activate reinforcement-learning
`

- The [**Gymnasium**](https://gymnasium.farama.org/) (0.29) library:

`
pip install gymnasium
pip install 'gymnasium[classic-control]'
`

- The [**Matplotlib**](https://matplotlib.org/) (3.8) graphing library:

`
pip install matplotlib  
`

- The [**TensorFlow**](https://www.tensorflow.org/) (2.5) machine learning library:

`
pip install tensorflow
`

## Exam description

The exam consists of a written test covering both the theoretical and practical topics presented in class. Specifically, students are expected to demonstrate a thorough understanding of the functioning of the algorithms underlying the development of autonomous agents and their ability to implement them in the Python programming language. The exam will assess the student's comprehension of the concepts, their ability to apply the algorithms, and their practical implementation skills. Details regarding the exam preparation methods and the level of depth required for each topic will be provided during the course of the lectures. The written exam will assess the actual acquisition of fundamental knowledge on Reinforcement Learning algorithms and their application for building autonomous agents. The presented problems and open-ended questions will allow the evaluation of students' ability to apply their knowledge in practical situations that may arise in real-world scenarios.