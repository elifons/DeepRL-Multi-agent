[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

# RL: Multi-Agent

 This project trains a multi RL system (MARL) to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment, where two agents control rackets and bounce a ball over a net. To solve this we use a single DDPG agent that collects experience from both players, with a shared Replay Buffer.

![Trained Agent][image1]

### Installation

1. Clone repository:

   ```
   $ git clone https://github.com/elifons/DeepRL-Multi-agent
   $ cd DeepRL-Multi-agent
   $ pip install -r requirements.txt
   ```

   Alternatively, follow the instractions on this link https://github.com/udacity/deep-reinforcement-learning#dependencies to set up a python environment.

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

3. Place the file in the DeepRL-Multi-agent GitHub repository,  and unzip (or decompress) the file. 

### Environment

If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting started

Example command to run the code.

```
$ python3 main.py --dest exp_marl --n_episodes 1000
```

Or you can follow the instructions in `Tennis.ipynb` to get started with training your own agent.

**optional arguments:**

```
  --n_episodes N_EPISODES		max number of training episodes (default: 1000)
  --max_t MAX_T         		max. number of timesteps per episode (default: 3000)
  --learn_every LEARN_EVERY		number of timesteps to wait until updating network (default: 5)
  --num_learning NUM_LEARNING	number of updates (default: 10)
  --goal GOAL           		reward goal that considers the problem solved (default: 0.5)
  --dest DEST           		experiment dir (default: runs)
```



