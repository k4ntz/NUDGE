import csv
import os
import sys
import time
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from rtpt import RTPT
from torch.optim import Optimizer, Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nudge.agents.logic_agent import LogicPPO
from nudge.agents.neural_agent import NeuralPPO
from nudge.env import NudgeBaseEnv
from nudge.utils import make_deterministic, save_hyperparams
from nudge.utils import exp_decay
from nudge.utils import print_program
from argparse import ArgumentParser



OUT_PATH = Path("out/")
IN_PATH = Path("in/")


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
         epsilon_fn: Callable = exp_decay,
         recover: bool = False,
         save_steps: int = 250000,
         stats_steps: int = 2000,
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
        save_steps: Number of steps between each checkpoint save
        stats_steps: Number of steps between each statistics summary timestamp
    """

    make_deterministic(seed)

    assert algorithm in ['ppo', 'logic']

    if env_kwargs is None:
        env_kwargs = dict()

    if update_steps is None:
        if algorithm == 'ppo':
            update_steps = max_ep_len * 4
        else:
            update_steps = 100

    env = NudgeBaseEnv.from_name(environment, mode=algorithm, **env_kwargs)

    now = datetime.now()
    experiment_dir = OUT_PATH / "runs" / environment / algorithm / now.strftime("%y-%m-%d-%H-%M")
    checkpoint_dir = experiment_dir / "checkpoints"
    image_dir = experiment_dir / "images"
    log_dir = experiment_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    save_hyperparams(signature=signature(main),
                     local_scope=locals(),
                     save_path=experiment_dir / "config.yaml",
                     print_summary=True)

    # initialize agent
    if algorithm == "ppo":
        agent = NeuralPPO(env, lr_actor, lr_critic, optimizer,
                          gamma, epochs, eps_clip, device)
    else:  # logic
        agent = LogicPPO(env, rules, lr_actor, lr_critic, optimizer,
                         gamma, epochs, eps_clip, device)
        print_program(agent)
        # print('Candidate Clauses:')
        # for clause in agent.policy.actor.clauses:
        #     print(clause)
        # print()

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
    print("Started training at ", now.strftime("%H:%M"))

    # printing and logging variables
    running_ret = 0  # running return
    n_episodes = 0

    rtpt = RTPT(name_initials='HS', experiment_name='LogicRL',
                max_iterations=total_steps)

    # Start the RTPT tracking
    writer = SummaryWriter(str(log_dir))
    rtpt.start()

    pbar = tqdm(total=total_steps - time_step, file=sys.stdout)
    while time_step < total_steps:
        state = env.reset()
        ret = 0  # return
        n_episodes += 1
        epsilon = epsilon_fn(i_episode)

        # Play episode
        for t in range(max_ep_len):
            action = agent.select_action(state, epsilon=epsilon)

            state, reward, done = env.step(action)

            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step += 1
            pbar.update(1)
            rtpt.step()
            ret += reward

            if time_step % update_steps == 0:
                agent.update()

            # printing average reward
            if time_step % stats_steps == 0:
                # print average reward till last episode
                avg_return = running_ret / n_episodes
                avg_return = round(avg_return, 2)

                print(f"\nEpisode: {i_episode} \t\t Timestep: {time_step} \t\t Average Reward: {avg_return}")
                running_ret = 0
                n_episodes = 1

                step_list.append([time_step])
                reward_list.append([avg_return])
                if algorithm == 'logic':
                    weights_list.append([(agent.get_weights().tolist())])

            # save model weights
            if time_step % save_steps == 1:
                checkpoint_path = checkpoint_dir / f"step_{time_step}.pth"
                if algorithm == 'logic':
                    agent.save(checkpoint_path, checkpoint_dir, step_list, reward_list, weights_list)
                else:
                    agent.save(checkpoint_path, checkpoint_dir, step_list, reward_list)
                print("\nSaved model at:", checkpoint_path)

            if done:
                break

        running_ret += ret
        i_episode += 1
        writer.add_scalar('Return', ret, i_episode)
        writer.add_scalar('Epsilon', epsilon, i_episode)

    env.close()
    pbar.close()

    with open(experiment_dir / 'data.csv', 'w', newline='') as f:
        dataset = csv.writer(f)
        header = ('steps', 'reward')
        dataset.writerow(header)
        data = np.hstack((step_list, reward_list))
        for row in data:
            dataset.writerow(row)

    if algorithm == 'logic':
        with open(experiment_dir / 'weights.csv', 'w', newline='') as f:
            dataset = csv.writer(f)
            for row in weights_list:
                dataset.writerow(row)

    end_time = time.time()
    print("Finished training at", datetime.now().strftime("%H:%M"))
    print(f"Total training time: {(end_time - start_time) / 60 :.0f} min")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("-g", "--game", type=str, default=None)
    parser.add_argument("-d", "--device", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.config is None:
        config_path = IN_PATH / "config" / "default.yaml"
    else:
        config_path = Path(args.config)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
        
    
    if args.game is not None:
        config["environment"] = args.game
    if args.device is not None:
        config["device"] = args.device

    main(**config)
