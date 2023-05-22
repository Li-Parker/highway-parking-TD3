import gymnasium as gym
import numpy as np
import argparse
from TD3 import TD3
from utils import create_directory, plot_learning_curve, scale_action
from matplotlib import pyplot as plt
from visdom import Visdom

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=3000)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints_updateForward/TD3/')
parser.add_argument('--figure_file', type=str, default='./output_images/reward_updateForward.png')

args = parser.parse_args()


def main():
    # env = gym.make('highway-v0')
    # env.configure({
    #     'parking': True,
    #     # 其他的配置参数
    # })

    env = gym.make('parking-v0')
    env.config["duration"] = 100
    env.config["vehicles_count"] = 5  # 停放车辆
    # 设置 Wr 参数
    wr = [1, 0.3, 0, 0, 0.02, 0.02]
    env.config["reward_weights"] = wr

    # 设置 rsg 参数
    env.config["success_goal_reward"] = 0.12
    # 设置 spots 参数

    # "collision_reward"
    env.config["collision_reward"] = -50
    # env.config["manual_control"] = True
    print(env.config)

    viz = Visdom()
    viz.line([[-50, -50]], [0.], win='TD3-parking_v0', opts=dict(title='TD3-parking_v0'))
    """
    learning rate = 0.0003 ===> alpha & beta
    γ = 0.99 ===> gamma
    τ = 0005 ===> tau
    replay buffer = 1000000 ===> max_size
    batch size = 256 ===> batch_size
    target_noise_clip = 0.5 ===> policy_noise_clip
    target_policy_noise = 0.2 ===> policy_noise
    policy_delay = 2 ===> delay_time
    train_freq = 100 ===> env.config["duration"] = 100
    Gaussian action noise = N(0,0.2) ===> action_noise = 0.2 ===> self.action_noise = math.sqrt(action_noise)
    Wr = [1, 0.3, 0, 0, 0.02, 0.02] ===>  wr = [1, 0.3, 0, 0, 0.02, 0.02] env.config["reward_weights"] = wr
    rsg = 0.12 ===>  env.config["success_goal_reward"] = 0.12
    spots=15*2 ===> 
    """
    agent = TD3(alpha=0.0003, beta=0.0003, state_dim=18,
                action_dim=env.action_space.shape[0], actor_fc1_dim=256, actor_fc2_dim=256,
                critic_fc1_dim=256, critic_fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.99,
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
        environment = True
        observation_temp = env.reset()

        # car_states = env.road
        # print("env.vehicle_states:________________________")
        # print(car_states.vehicles[1].LENGTH)
        # print("goal:______________________________")
        # print(env.goal.position)
        # print("controlled_vehicles:________________________________")
        # print(env.controlled_vehicles)
        for v in env.road.vehicles:
            if np.linalg.norm(v.position - env.goal.position) < 20:
                environment = False
                break
        while not environment:
            environment = True
            observation_temp = env.reset()
            for v in env.road.vehicles:
                if np.linalg.norm(v.position - env.goal.position) < 20:
                    environment = False
                    break
        observation = []
        observation = np.append(observation, dict(observation_temp[0])["achieved_goal"])
        observation = np.append(observation, dict(observation_temp[0])["desired_goal"])
        observation = np.append(observation, dict(observation_temp[0])["observation"])
        count = 0

        while (not done) and (not done_):
            action = agent.choose_action(observation, train=True)
            action_ = scale_action(action, low=env.action_space.low, high=env.action_space.high)
            observation__temp, reward, done, done_, info = env.step(action_)  # done_:可以提前判断是否结束
            is_success = info.get("is_success")
            is_crashed = info.get("crashed")
            if is_success:
                reward = reward + 10
            if is_crashed:
                control_vehicle_position = env.road.vehicles[0].position
                print("crashed_position:_____________")
                print(control_vehicle_position)
                add_reward = np.linalg.norm(control_vehicle_position - env.goal.position)
                print("add_reward:_________________________________")
                print(add_reward)
                reward = reward - add_reward
            count += 1

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
        viz.line([[total_reward, avg_reward]], [episode + 1], win='TD3-parking_v0', update='append')

        if (episode + 1) % 200 == 0:
            agent.save_models(episode + 1)

    episodes = [i + 1 for i in range(args.max_episodes)]
    plot_learning_curve(episodes, avg_reward_history, title='AvgReward', ylabel='reward',
                        figure_file=args.figure_file)

    plt.imshow(env.render())
    plt.show()


if __name__ == '__main__':
    main()
