import argparse
import os

from env_src.render_atari import render_atari
from env_src.procgen.render import render_loot, render_ecoinrun, render_threefish
from env_src.getout.getout.render import render_getout
from nudge.agents.neural_agent import NeuralPlayer
from nudge.agents.logic_agent import LogicPlayer
from nudge.agents.random_agent import RandomPlayer
from nudge.utils import make_deterministic
from utils import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", help="Seed for pytorch + env", default=0,
                        required=False, action="store", dest="seed", type=int)
    parser.add_argument("-alg", "--algorithm", help="algorithm that to use",
                        action="store", dest="alg", required=True,
                        choices=['ppo', 'logic', 'random', 'human'])
    parser.add_argument("-m", "--mode", help="the game mode you want to play with",
                        required=True, action="store", dest="m",
                        choices=['getout', 'threefish', 'loot', 'ecoinrun', 'atari'])
    parser.add_argument("-env", "--environment", help="environment of game to use",
                        required=True, action="store", dest="env",
                        choices=['getout', 'getoutplus', 'getout4en',
                                 'threefish', 'threefishcolor',
                                 'loot', 'lootcolor', 'lootplus', 'loothard',
                                 'ecoinrun', 'freeway', 'kangaroo', 'asterix'])
    parser.add_argument("-r", "--rules", dest="rules", default=None,
                        required=False,
                        choices=['getout_human_assisted', 'getout_bs_top10', 'getout_bs_rf1',
                                 'getout_bs_rf3', 'getoutplus', 'getout_redundant_actions',
                                 'threefish_human_assisted', 'threefishcolor', 'threefish_bs_top5', 'threefish_bs_rf3',
                                 'threefish_bs_rf1', 'threefish_redundant_actions',
                                 'loot_human_assisted', 'loot_bs_top5', 'loot_bs_rf3', 'loot_bs_rf1', 'loothard',
                                 'loot_redundant_actions', 'freeway_bs_rf1', 'asterix_bs_rf1'
                                 ])
    parser.add_argument("-l", "--log", help="record the information of games", action="store_true")
    parser.add_argument("-rec", "--record", help="record the rendering of the game", action="store_true")
    parser.add_argument("--log_file_name", help="the name of log file", required=False, dest='logfile')
    parser.add_argument("--render", help="render the game", action="store_true", dest="render")
    # arg = ['-alg', 'human', '-m', 'getout', '-env', 'getout','-l','True']
    args = parser.parse_args()

    # fix seed
    make_deterministic(args.seed)

    # load trained_model
    if args.alg not in ['random', 'human']:
        # read filename from models
        current_path = os.path.dirname(__file__)

        # model_name = input('Enter file name: ')
        if args.alg == "logic":
            model_name = "beam_search_top1.pth"
        elif args.alg == "ppo":
            model_name = "ppo_.pth"
        else:
            models_folder = os.path.join(current_path, 'models', args.m, args.alg)
            print(f"Please use one of the following agent: {os.listdir(models_folder)}")
            model_name = input('Enter file name: ')
        # model_file = os.path.join(models_folder, model_name)
        model_file = os.path.join(current_path, 'models', args.m, args.alg, model_name)
        model = load_model(model_file)
    else:
        model = None

    ###################### logging ######################
    #### log files for multiple runs are NOT overwritten
    if args.log:
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + args.m + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        #### get number of log files in log directory
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run
        if args.alg == 'human':
            player_name = input("Please give your name :")
            log_f_name = log_dir + args.alg + '_' + args.env + '_' + player_name + "_log_" + str(run_num) + ".csv"
        else:
            log_f_name = log_dir + args.alg + '_' + args.env + "_log_" + str(run_num) + ".csv"
        args.logfile = log_f_name
        print("current logging run number for " + args.env + " : ", run_num)
        print("logging at : " + log_f_name)

    #### create agent
    if args.alg == 'ppo':
        agent = NeuralPlayer(args, model)
    elif args.alg == 'logic':
        agent = LogicPlayer(args, model)
    elif args.alg == 'random':
        agent = RandomPlayer(args)
    elif args.alg == 'human':
        agent = 'human'

    #### Continue to render
    if args.m == 'getout':
        render_getout(agent, args)
    elif args.m == 'threefish':
        render_threefish(agent, args)
    elif args.m == 'loot':
        render_loot(agent, args)
    elif args.m == 'ecoinrun':
        render_ecoinrun(agent, args)
    elif args.m == 'atari':
        render_atari(agent, args)
    else:
        print("Wrong game provided")
        exit(1)

if __name__ == "__main__":
    main()
