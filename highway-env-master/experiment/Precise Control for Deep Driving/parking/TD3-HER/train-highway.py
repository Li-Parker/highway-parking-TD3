import gymnasium as gym
import numpy as np
import argparse
from TD3 import TD3
from utils import create_directory, plot_learning_curve, scale_action
from matplotlib import pyplot as plt
from visdom import Visdom

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=1000)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints_highway/TD3/')
parser.add_argument('--figure_file', type=str, default='./output_images_highway/reward.png')

args = parser.parse_args()


def main():
    # env = gym.make('highway-v0')
    # env.configure({
    #     'parking': True,
    #     # 其他的配置参数
    # })
    viz = Visdom()
    viz.line([[-550, -550]], [0.], win='TD3-parking_v0', opts=dict(title='TD3-parking_v0'))

    env = gym.make('highway-v0')
    env.config["duration"] = 50  # 持续时间
    env.config["vehicles_count"] = 5  # 停放车辆

    agent = TD3(alpha=0.0003, beta=0.0003, state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0], actor_fc1_dim=400, actor_fc2_dim=300,
                critic_fc1_dim=400, critic_fc2_dim=300, ckpt_dir=args.ckpt_dir, gamma=0.99,
                tau=0.005, action_noise=0.1, policy_noise=0.2, policy_noise_clip=0.5,
                delay_time=2, max_size=1000000, batch_size=256)
    create_directory(path=args.ckpt_dir, sub_path_list=['Actor', 'Critic1', 'Critic2', 'Target_actor',
                                                        'Target_critic1', 'Target_critic2'])

    total_reward_history = []
    avg_reward_history = []
    for episode in range(args.max_episodes):
        total_reward = 0
        done = False
        done_ = False
        observation_temp = env.reset()
        observation = []
        observation = np.append(observation, dict(observation_temp[0])["achieved_goal"])
        observation = np.append(observation, dict(observation_temp[0])["desired_goal"])
        observation = np.append(observation, dict(observation_temp[0])["observation"])
        while (not done) and (not done_):
            action = agent.choose_action(observation, train=True)
            action_ = scale_action(action, low=env.action_space.low, high=env.action_space.high)
            observation__temp, reward, done, done_, info = env.step(action_)  # done_:可以提前判断是否结束
            observation_ = []
            observation_ = np.append(observation_, dict(observation__temp)["achieved_goal"])
            observation_ = np.append(observation_, dict(observation__temp)["desired_goal"])
            observation_ = np.append(observation_, dict(observation__temp)["observation"])
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            total_reward += reward
            observation = observation_
            env.render()
        total_reward_history.append(total_reward)
        avg_reward = np.mean(total_reward_history[-100:])
        avg_reward_history.append(avg_reward)
        print('Ep: {} Reward: {} AvgReward: {}'.format(episode + 1, total_reward, avg_reward))
        viz.line([[total_reward, avg_reward]], [episode+1], win='TD3-parking_v0', update='append')

        if (episode + 1) % 200 == 0:
            agent.save_models(episode + 1)

    episodes = [i + 1 for i in range(args.max_episodes)]
    plot_learning_curve(episodes, avg_reward_history, title='AvgReward', ylabel='reward',
                        figure_file=args.figure_file)

    plt.imshow(env.render())
    plt.show()


if __name__ == '__main__':
    main()
