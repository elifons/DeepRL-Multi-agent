  
from unityagents import UnityEnvironment
import numpy as np
from utils.ddpg_agent import Agent
from collections import deque
import torch
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import os.path as op

# Code based on  https://github.com/udacity/deep-reinforcement-learning.git

def create_directory(logdir):
    try:
        os.makedirs(logdir)
    except FileExistsError:
        pass

def ddpg(dir_, n_episodes=3000, max_t = 3000, learn_every=5, num_learn=10, goal=0.5):
    scores_window = deque(maxlen=100)
    scores1 = []
    scores2 = []
    scores = []
    max_score = -np.Inf
    
    
    agent1 = Agent(state_size=state_size, action_size=action_size, random_seed=17)
    agent2 = Agent(state_size=state_size, action_size=action_size, random_seed=17)    

    agent2.critic_local = agent1.critic_local
    agent2.critic_target = agent1.critic_target
    agent2.critic_optimizer = agent1.critic_optimizer

    agent2.actor_local = agent1.actor_local
    agent2.actor_target = agent1.actor_target
    agent2.actor_optimizer = agent1.actor_optimizer

    agent2.memory = agent1.memory
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent1.reset()
        agent2.reset()
        score1 = 0
        score2 = 0
#         while True:
        for t in range(max_t):
            state1 = state[0]
            state2 = state[1]
            state1 = state1.reshape((1,state.shape[1]))
            state2 = state2.reshape((1,state.shape[1]))
            
            action1 = agent1.act(state1)
            action2 = agent1.act(state2)
            env_info = env.step([action1, action2])[brain_name]        
            next_state = env_info.vector_observations   
            reward = env_info.rewards                   
            done = env_info.local_done                  
            
            next_state1 = next_state[0]
            next_state2 = next_state[1]
            
            for s, a, r, n_s, d in zip(state, [action1, action2], reward, next_state, done):
                agent1.add_memory(s, a, r, n_s, d)
                agent2.add_memory(s, a, r, n_s, d)

            state = next_state
            score1 += reward[0]
            score2 += reward[1]
                        
            if t%learn_every == 0:
                for _ in range(num_learn):
                    agent1.step()
                    agent2.step()
            
            if np.all(done):
                break 
        
        scores1.append(score1)
        scores2.append(score2)
        
        score_max = np.max([score1,score2])
        scores_window.append(score_max)
        scores.append(score_max)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_window), score_max), end="")
        if i_episode % 100 == 0:
            torch.save(agent1.actor_local.state_dict(), op.join(dir_, 'checkpoint_actor_1.pth'))
            torch.save(agent1.critic_local.state_dict(), op.join(dir_, 'checkpoint_critic_1.pth'))
            torch.save(agent2.actor_local.state_dict(), op.join(dir_, 'checkpoint_actor_2.pth'))
            torch.save(agent2.critic_local.state_dict(), op.join(dir_, 'checkpoint_critic_2.pth')) 
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))   
        if np.mean(scores_window)>=goal:
            print('\nEnvironment solved after {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent1.actor_local.state_dict(), op.join(dir_, 'checkpoint_actor_1.pth'))
            torch.save(agent1.critic_local.state_dict(), op.join(dir_, 'checkpoint_critic_1.pth'))
            torch.save(agent2.actor_local.state_dict(), op.join(dir_, 'checkpoint_actor_2.pth'))
            torch.save(agent2.critic_local.state_dict(), op.join(dir_, 'checkpoint_critic_2.pth')) 
            break
    return scores


if __name__ == '__main__':  
  
    # Inputs for the main function
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_episodes', default=1000, type=int, help='max number of training episodes')
    parser.add_argument('--max_t', default=3000, type=int, help='max. number of timesteps per episode')
    parser.add_argument('--learn_every', default=5, type=int, help='number of timesteps to wait until updating network')
    parser.add_argument('--num_learning', default=10, type=int, help='number of updates')
    parser.add_argument('--goal', default=0.5, type=float, help='reward goal that considers the problem solved')  
    parser.add_argument('--dest', default='runs', type=str, help='experiment dir')
    args = parser.parse_args() 

    # env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")
    env = UnityEnvironment(file_name="Tennis.app")
    path = args.dest
    create_directory(path)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents 
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    scores = ddpg(dir_=path, n_episodes=args.n_episodes, max_t=args.max_t, learn_every=args.learn_every, num_learn=args.num_learning, goal=args.goal)

    env.close()


    df_scores = pd.DataFrame(scores)
    df_scores.to_csv(op.join(path, 'scores_values.csv'))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(op.join(path, 'score.png'))













