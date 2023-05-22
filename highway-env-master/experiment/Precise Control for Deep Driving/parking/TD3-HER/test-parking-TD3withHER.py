import gymnasium as gym
import imageio
import argparse

import numpy as np

from TD3withHER import TD3
from utils import scale_action

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints_HER/TD3/')
parser.add_argument('--figure_file', type=str, default='./output_images/LunarLander.gif')
parser.add_argument('--fps', type=int, default=30)
parser.add_argument('--render', type=bool, default=True)
parser.add_argument('--save_video', type=bool, default=True)

args = parser.parse_args()


def main():
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
    agent = TD3(alpha=0.0003, beta=0.0003, state_dim=18,
                action_dim=env.action_space.shape[0], actor_fc1_dim=256, actor_fc2_dim=256,
                critic_fc1_dim=256, critic_fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.99,
                tau=0.005, action_noise=0.1, policy_noise=0.2, policy_noise_clip=0.5,
                delay_time=2, max_size=1000000, batch_size=256)
    agent.load_models(3000)
    video = imageio.get_writer(args.figure_file, fps=args.fps)

    done = False
    done_ = False
    observation_temp = env.reset()
    observation = []
    observation = np.append(observation, dict(observation_temp[0])["achieved_goal"])
    observation = np.append(observation, dict(observation_temp[0])["desired_goal"])
    observation = np.append(observation, dict(observation_temp[0])["observation"])
    while not (done or done_):
        if args.render:
            env.render()
        action = agent.choose_action(observation, train=True)
        action_ = scale_action(action, low=env.action_space.low, high=env.action_space.high)
        observation__temp, reward, done, done_, info = env.step(action_)  # done_:可以提前判断是否结束
        is_success = info.get("is_success")
        is_crashed = info.get("crashed")
        if is_success:
            reward = reward + 10
        if is_crashed:
            reward = reward - 10
        observation_ = []
        observation_ = np.append(observation_, dict(observation__temp)["achieved_goal"])
        observation_ = np.append(observation_, dict(observation__temp)["desired_goal"])
        observation_ = np.append(observation_, dict(observation__temp)["observation"])
        observation = observation_
        if args.save_video:
            video.append_data(env.render(mode='rgb_array'))


if __name__ == '__main__':
    for _ in range(100):
        main()
