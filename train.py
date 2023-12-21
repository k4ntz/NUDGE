import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

from rtpt import RTPT
from tqdm import tqdm

from agents.logic_agent import LogicPPO
from agents.neural_agent import NeuralPPO
from config import *
from utils import make_deterministic
from torch.utils.tensorboard import SummaryWriter
import datetime
from env import NudgeBaseEnv


OUT_PATH = Path("out/")


def main():
    device = "cpu"

    ################### args definition ###################
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", help="Seed for pytorch + env",
                        required=False, action="store", dest="seed", type=int, default=0)
    parser.add_argument("-alg", "--algorithm", help="algorithm that to use",
                        action="store", dest="alg", required=True,
                        choices=['ppo', 'logic'])
    parser.add_argument("-m", "--mode", help="the game mode you want to play with",
                        required=True, action="store", dest="m",
                        choices=['getout', 'threefish', 'loot', 'atari'])
    parser.add_argument("-env", "--environment", help="environment of game to use",
                        required=True, action="store", dest="env",
                        choices=['getout', 'threefish', 'loot', 'freeway', 'kangaroo', 'asterix', 'loothard'])
    parser.add_argument("-r", "--rules", dest="rules", default=None, required=False)
    parser.add_argument('-p', '--plot', help="plot the image of weights", type=bool, default=False, dest='plot')
    parser.add_argument('-re', '--recovery', help='recover from crash', default=False, type=bool, dest='recover')
    # arg = ['-alg', 'logic', '-m', 'threefish', '-env', 'threefish', '-p', 'True', '-r', 'threefish_human_assisted']
    args = parser.parse_args()

    #####################################################
    # load environment
    print("training environment name : " + args.env.capitalize())
    make_deterministic(args.seed)

    #####################################################
    # config setting
    if args.alg == 'ppo':
        update_timestep = max_ep_len * 4
    elif args.alg == 'logic' and args.m == 'atari':
        # a large num causes out of memory
        update_timestep = 100
        # print("PUT BACK 20 ! ")
        # update_timestep = 7
        # max_ep_len = 100
    else:
        update_timestep = max_ep_len * 2

    if args.m == 'loot' and args.alg == 'ppo':
        max_training_timesteps = 5000000
    else:
        max_training_timesteps = 800000
    #####################################################

    env = NudgeBaseEnv.from_name(args.env, mode=args.alg)

    #####################################################
    # config = {
    #     "seed": args.seed,
    #     "learning_rate_actor": lr_actor,
    #     "learning_rate_critic": lr_critic,
    #     "epochs": K_epochs,
    #     "gamma": gamma,
    #     "eps_clip": eps_clip,
    #     "max_steps": max_training_timesteps,
    #     "eps start": 1.0,
    #     "eps end": 0.02,
    #     "max_ep_len": max_ep_len,
    #     "update_freq": max_ep_len * 2,
    #     "save_freq": max_ep_len * 50,
    # }
    # if args.rules is not None:
    #     runs_name = str(args.rules) + '_seed_' + str(args.seed)
    # else:
    #     runs_name = str(args.m) + '_' + args.alg + '_seed_' + str(args.seed)

    # wandb.init(project="GETOUT-BS", entity="nyrus", config=config, name=runs_name)
    # wandb.init(project="LOOT", entity="nyrus", config=config, name=runs_name)
    # wandb.init(project="THREEFISH", entity="nyrus", config=config, name=runs_name)

    checkpoint_dir = OUT_PATH / "checkpoints" / args.env / args.alg / str(args.seed)
    image_dir = OUT_PATH / "images" / args.env / args.alg / str(args.seed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    checkpoint_path = checkpoint_dir / "{}_{}.pth".format(args.env, 0)

    print("save checkpoint path :", checkpoint_path)

    #####################################################

    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")

    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)

    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

    print("--------------------------------------------------------------------------------------------")

    # print("state space dimension : ", state_dim)
    # print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")

    print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")

    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")

    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    #####################################################

    print("============================================================================================")

    ################# training procedure ################
    #
    # initialize agent
    if args.alg == "ppo":
        agent = NeuralPPO(lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip, args, device)
    elif args.alg == "logic":
        agent = LogicPPO(lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip, args, device)
        print('Candidate Clauses:')
        for clause in agent.policy.actor.clauses:
            print(clause)
    else:
        raise ValueError("Invalid algorithm.")

    i_episode = 0
    weights_list = []

    if args.recover:
        if args.alg == 'logic':
            step_list, reward_list, weights_list = agent.load(checkpoint_dir)
        else:
            step_list, reward_list = agent.load(checkpoint_dir)
        time_step = max(step_list)[0]
    else:
        step_list = []
        reward_list = []
        time_step = 0

    # track total training time
    start_time = time.time()
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    rtpt = RTPT(name_initials='HS', experiment_name='LogicRL',
                max_iterations=max_training_timesteps)

    # Start the RTPT tracking
    folder_name = f"{args.m}_{args.env}_{args.alg}_{args.rules}_s{args.seed}"
    folder_name += datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    writer = SummaryWriter(f"runs/{folder_name}")
    rtpt.start()

    # training loop
    pbar = tqdm(total=max_training_timesteps-time_step, file=sys.stdout)
    while time_step <= max_training_timesteps:
        #  initialize game
        state = env.reset()

        # state = initialize_game(env, args)
        current_ep_reward = 0

        epsilon = epsilon_func(i_episode)

        for t in range(1, max_ep_len + 1):
            action = agent.select_action(state, epsilon=epsilon)

            state, reward, done = env.step(action)

            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step += 1
            pbar.update(1)
            rtpt.step()
            current_ep_reward += reward

            if time_step % update_timestep == 0:
                agent.update()

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))
                # wandb.log({'reward': print_avg_reward}, step=time_step)
                print_running_reward = 0
                print_running_episodes = 0

                step_list.append([time_step])
                reward_list.append([print_avg_reward])
                if args.alg == 'logic':
                    weights_list.append([(agent.get_weights().tolist())])

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                checkpoint_path = checkpoint_dir / "{}_{}_step_{}.pth".format(args.alg, args.env,
                                                                         time_step)
                print("saving model at :", checkpoint_path)
                if args.alg == 'logic':
                    agent.save(checkpoint_path, checkpoint_dir, step_list, reward_list, weights_list)
                else:
                    agent.save(checkpoint_path, checkpoint_dir, step_list, reward_list)
                print("model saved")
                print("Elapsed Time  : ", time.time() - start_time)
                print("--------------------------------------------------------------------------------------------")

                # save image of weights
                # if args.plot:
                #     if args.alg == 'logic':
                #         plot_weights(agent.get_weights(), image_directory, time_step)

            # break; if the episode is over
            if done:
                # print("Game over. New episode.")
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        i_episode += 1
        writer.add_scalar('Episode reward', current_ep_reward, i_episode)
        writer.add_scalar('Epsilon', epsilon, i_episode)

    env.close()

    # print total training time
    print("============================================================================================")
    with open(checkpoint_dir + '/' + 'data.csv', 'w', newline='') as f:
        dataset = csv.writer(f)
        header = ('steps', 'reward')
        dataset.writerow(header)
        data = np.hstack((step_list, reward_list))
        for row in data:
            dataset.writerow(row)
    if args.alg == 'logic':
        with open(checkpoint_dir + '/' + 'weights.csv', 'w', newline='') as f:
            dataset = csv.writer(f)
            for row in weights_list:
                dataset.writerow(row)

    end_time = time.time()
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == "__main__":
    main()
