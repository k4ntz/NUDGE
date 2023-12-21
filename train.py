import csv
import os
import sys
import time
from pathlib import Path
from typing import Callable
import math

from torch.optim import Optimizer, Adam
import yaml

import numpy as np

from rtpt import RTPT
from tqdm import tqdm

from agents.logic_agent import LogicPPO
from agents.neural_agent import NeuralPPO
from utils import make_deterministic
from torch.utils.tensorboard import SummaryWriter
import datetime
from env import NudgeBaseEnv

OUT_PATH = Path("out/")
IN_PATH = Path("in/")


def epsilon_fn(episode: int):
    return max(math.exp(-episode / 500), 0.02)


def main(algorithm: str,
         environment: str,
         env_kwargs: dict = None,
         rules: str = "default",
         seed: int = 0,
         device: str = "cpu",
         total_steps: int = 800000,
         max_ep_len: int = 500,
         update_steps: int = None,
         epochs: int = 20,
         eps_clip: float = 0.2,
         gamma: float = 0.99,
         optimizer: Optimizer = Adam,
         lr_actor: float = 0.001,
         lr_critic: float = 0.0003,
         epsilon_fn: Callable = epsilon_fn,
         recover: bool = False,
         plot: bool = False,
         save_steps: int = 250000,
         print_steps: int = 2000,
         log_steps: int = 2000,
         ):
    """

    Args:
        algorithm: Either 'ppo' for Proximal Policy Optimization or 'logic'
            for First-Order Logic model
        environment: The name of the environment to use (prepared inside in/envs)
        env_kwargs: Optional settings for the environment
        rules: The name of the logic rule set to use
        seed: Random seed for reproduction
        device: For example 'cpu' or 'cuda:0'
        total_steps: Total number of time steps to train the agent
        max_ep_len: Maximum number of time steps per episode
        update_steps: Number of time steps between agent updates. Caution: if too
            high, causes OutOfMemory errors when running with CUDA.
        epochs: Number of epochs (k) per agent update
        eps_clip: Clipping factor epsilon for PPO
        gamma: Discount factor
        optimizer: The optimizer to use for agent updates
        lr_actor: Learning rate of the actor (policy)
        lr_critic: Learning rate of the critic (value fn)
        epsilon_fn: Function mapping episode number to epsilon (greedy) for
            exploration
        recover: If true, tries to reload an existing run that was interrupted
            before completion.
        plot: If true, plots the weights
        save_steps: Number of steps between each checkpoint save
        print_steps: Number of steps between each statistics summary print
        log_steps: Number of steps between each statistics logging
    """

    make_deterministic(seed)

    assert algorithm in ['ppo', 'logic']

    if update_steps is None:
        if algorithm == 'ppo':
            update_steps = max_ep_len * 4
        else:
            update_steps = 100

    env = NudgeBaseEnv.from_name(environment, mode=algorithm)

    now_str = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    experiment_dir = OUT_PATH / "runs" / environment / algorithm / now_str
    checkpoint_dir = experiment_dir / "checkpoints"
    image_dir = experiment_dir / "images"
    log_dir = experiment_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # TODO: print summary and save config

    # initialize agent
    if algorithm == "ppo":
        agent = NeuralPPO(env, lr_actor, lr_critic, optimizer,
                          gamma, epochs, eps_clip, device)
    else:  # logic
        agent = LogicPPO(env, rules, lr_actor, lr_critic, optimizer,
                         gamma, epochs, eps_clip, device)
        print('Candidate Clauses:')
        for clause in agent.policy.actor.clauses:
            print(clause)

    i_episode = 0
    weights_list = []

    if recover:
        if algorithm == 'ppo':
            step_list, reward_list = agent.load(checkpoint_dir)
        else:  # logic
            step_list, reward_list, weights_list = agent.load(checkpoint_dir)
        time_step = max(step_list)[0]
    else:
        step_list = []
        reward_list = []
        time_step = 0

    # track total training time
    start_time = time.time()
    print("Started training at (GMT):", start_time)

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    rtpt = RTPT(name_initials='HS', experiment_name='LogicRL',
                max_iterations=total_steps)

    # Start the RTPT tracking
    writer = SummaryWriter(str(log_dir))
    rtpt.start()

    # training loop
    pbar = tqdm(total=total_steps - time_step, file=sys.stdout)
    while time_step <= total_steps:
        #  initialize game
        state = env.reset()

        # state = initialize_game(env, args)
        current_ep_reward = 0

        epsilon = epsilon_fn(i_episode)

        for t in range(1, max_ep_len + 1):
            action = agent.select_action(state, epsilon=epsilon)

            state, reward, done = env.step(action)

            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step += 1
            pbar.update(1)
            rtpt.step()
            current_ep_reward += reward

            if time_step % update_steps == 0:
                agent.update()

            # printing average reward
            if time_step % print_steps == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print(f"Episode: {i_episode} \t\t Timestep: {time_step} \t\t Average Reward: {print_avg_reward}")
                print_running_reward = 0
                print_running_episodes = 0

                step_list.append([time_step])
                reward_list.append([print_avg_reward])
                if algorithm == 'logic':
                    weights_list.append([(agent.get_weights().tolist())])

            # save model weights
            if time_step % save_steps == 0:
                checkpoint_path = checkpoint_dir / f"step_{time_step}.pth"
                if algorithm == 'logic':
                    agent.save(checkpoint_path, checkpoint_dir, step_list, reward_list, weights_list)
                else:
                    agent.save(checkpoint_path, checkpoint_dir, step_list, reward_list)
                print("Saved model at:", checkpoint_path)
                print("Elapsed Time : ", time.time() - start_time)

                # save image of weights
                # if plot and algorithm == 'logic':
                #     plot_weights(agent.get_weights(), image_dir, time_step)

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        i_episode += 1
        writer.add_scalar('Episode reward', current_ep_reward, i_episode)
        writer.add_scalar('Epsilon', epsilon, i_episode)

    env.close()

    # print total training time
    with open(checkpoint_dir / 'data.csv', 'w', newline='') as f:
        dataset = csv.writer(f)
        header = ('steps', 'reward')
        dataset.writerow(header)
        data = np.hstack((step_list, reward_list))
        for row in data:
            dataset.writerow(row)

    if algorithm == 'logic':
        with open(checkpoint_dir / 'weights.csv', 'w', newline='') as f:
            dataset = csv.writer(f)
            for row in weights_list:
                dataset.writerow(row)

    end_time = time.time()
    print("Started training at (GMT):", start_time)
    print("Finished training at (GMT):", end_time)
    print("Total training time:", end_time - start_time)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = IN_PATH / "config" / "default.yaml"

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    main(**config)
