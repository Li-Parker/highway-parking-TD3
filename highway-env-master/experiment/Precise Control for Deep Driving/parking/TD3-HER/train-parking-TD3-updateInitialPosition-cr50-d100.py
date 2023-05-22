import math

import gymnasium as gym
import numpy as np
import argparse
from TD3 import TD3
from utils import create_directory, plot_learning_curve, scale_action
from matplotlib import pyplot as plt
from visdom import Visdom
from calculateArea import *

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=3000)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints_updateInitialPosition_cr50_d100/TD3/')
parser.add_argument('--figure_file', type=str, default='./output_images/reward_updateInitialPosition_cr50_d100.png')

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
    viz.line([[-50, -50]], [0.], win='TD3_updateInitialPosition-parking_v0_cr50_d100',
             opts=dict(title='TD3_updateInitialPosition-parking_v0_cr50_d100'))
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
        initial_position_x = np.random.uniform(25, 30)
        initial_position_y = np.random.uniform(0, 8)
        if np.random.uniform(0, 1) > 0.5:
            initial_position_x = -initial_position_x
        if np.random.uniform(0, 1) > 0.5:
            initial_position_y = -initial_position_y
        env.controlled_vehicles[0].position = [initial_position_x, initial_position_y]  # 25~30  0~8
        print("all vehicles:____________________")
        print(env.road.vehicles)

        env.road.vehicles[0].position = [initial_position_x, initial_position_y]
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
                cos = env.controlled_vehicles[0].direction[0]
                sin = env.controlled_vehicles[0].direction[1]
                add_reward = 10
                vehicles_count = 0
                crashed_area = 0.2
                for v in env.road.vehicles:
                    if vehicles_count == 0:
                        print(v.position)
                        #  获取controlled_vehicle的矩形的四个端点的坐标，按照顺时针/逆时针记录到controlled_car中

                        tmp_x = v.position[0]
                        tmp_y = v.position[1]

                        """
                        增加一个碰撞惯性
                        """
                        speed = abs(env.controlled_vehicles[0].speed)
                        print("speed:#################################################")
                        print(speed)
                        speed = math.sqrt(speed)
                        addition_d = speed
                        print("addition_d")
                        print(addition_d)
                        tmp_x = tmp_x + addition_d*cos
                        tmp_y = tmp_y + addition_d*sin

                        tmp_x1 = tmp_x + 2 * cos + sin
                        tmp_y1 = tmp_y + 2 * sin - cos
                        tmp_x2 = tmp_x + 2 * cos - sin
                        tmp_y2 = tmp_y + 2 * sin + cos
                        tmp_x3 = 2 * tmp_x - tmp_x1
                        tmp_y3 = 2 * tmp_y - tmp_y1
                        tmp_x4 = 2 * tmp_x - tmp_x2
                        tmp_y4 = 2 * tmp_y - tmp_y2

                        controlled_car = np.array([[tmp_x1, tmp_y1], [tmp_x2, tmp_y2], [tmp_x3, tmp_y3], [tmp_x4, tmp_y4]], dtype=np.float32)
                        # print("controlled_car:________________")
                        # print(controlled_car)
                    elif np.linalg.norm(env.controlled_vehicles[0].position - v.position) < 20:
                        other_crashed_car = np.array(
                            [[v.position[0] + 1, v.position[1] + 2], [v.position[0] - 1, v.position[1] + 2],
                             [v.position[0] - 1, v.position[1] - 2], [v.position[0] + 1, v.position[1] - 2]],
                            dtype=np.float32)
                        # print("other_crashed_car:__________________")
                        # print(other_crashed_car)
                        crashed_area += overlap_area(other_crashed_car, controlled_car)
                    vehicles_count += 1
                print("crashed_area:______________________")
                print(crashed_area)
                # print("direction:_____________________")
                # print(env.controlled_vehicles[0].direction)
                reward = reward - add_reward*crashed_area
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
        viz.line([[total_reward, avg_reward]], [episode + 1], win='TD3_updateInitialPosition-parking_v0_cr50_d100',
                 update='append')

        if (episode + 1) % 200 == 0:
            agent.save_models(episode + 1)

    episodes = [i + 1 for i in range(args.max_episodes)]
    plot_learning_curve(episodes, avg_reward_history, title='AvgReward', ylabel='reward',
                        figure_file=args.figure_file)

    plt.imshow(env.render())
    plt.show()


if __name__ == '__main__':
    main()
