import matplotlib.pyplot as plt
import numpy as np
import copy
import torch
import argparse

from operator import add

from cairo_gym.sawyer_vel_reach import SawyerVelReach
from ddpg_sawyer.ddpg_agent import Agent

POINT = (0.75, 0, .55)
START_POS = (0.5, 0.0, 0.8)

goal_point = POINT

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='swayer ddpg')
    parser.add_argument('--pretrained', type=str2bool, default=False, help='True or False')
    parser.add_argument('--noise', type=str2bool, default=True, help='True or False')
    parser.add_argument('--steps', type=int, default=1000, help='interger of number of steps')
    parser.add_argument('--episodes', type=int, default=100, help='interger of number of episodes')
    parser.add_argument('--save_name', default='checkpoint', help='checkpoint save name')
    args = parser.parse_args()
    
    env = SawyerVelReach()
    agent = Agent(env.observation_space, env.action_space)

    if args.pretrained:
        agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth', map_location="cpu"))
        agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth', map_location="cpu"))
        agent.actor_target.load_state_dict(torch.load('checkpoint_actor.pth', map_location="cpu"))
        agent.critic_target.load_state_dict(torch.load('checkpoint_critic.pth', map_location="cpu"))

    reward_list = []

    for i in range(args.episodes):
        env.reset()
        state = env._get_state()
        score = 0
        for t in range(args.steps):

            action = agent.act(state, args.noise)
            next_state, reward, done = env.step(action[0])

            agent.step(state, action, reward, next_state, done)

            state = copy.copy(next_state)
            score += reward
            

            if done:
                break
        
        print('Reward: {} | Episode: {}/{}'.format(score, i, args.episodes))
        reward_list.append(score)
        

    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
    torch.save(agent.actor_target.state_dict(), 'checkpoint_actor_t.pth')
    torch.save(agent.critic_target.state_dict(), 'checkpoint_critic_t.pth')

    print('Training saved')

    env.close()
    

    fig = plt.figure()
    plt.plot(np.arange(1, len(reward_list) + 1), reward_list)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
