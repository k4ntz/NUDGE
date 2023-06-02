import argparse
import torch
import os

from utils import make_deterministic
from utils_game import render_getout, render_threefish, render_loot, render_ecoinrun, render_atari
from agents.neural_agent import ActorCritic, NeuralPlayer
from agents.logic_agent import NSFR_ActorCritic, LogicPlayer
from agents.random_agent import RandomPlayer

device = torch.device('cuda:0')


def load_model(model_path, args, set_eval=True):
    with open(model_path, "rb") as f:
        model = NSFR_ActorCritic(args).to(device)
        model.load_state_dict(state_dict=torch.load(f))

    model = model.actor
    model.as_dict = True

    if set_eval:
        model = model.eval()

    return model


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
                        choices=['getout', 'getoutplus',
                                 'threefish', 'threefishcolor',
                                 'loot', 'lootcolor', 'lootplus',
                                 'ecoinrun'])
    parser.add_argument("-r", "--rules", dest="rules", default=None,
                        required=False,
                        choices=['getout_human_assisted', 'getout_bs_top10', 'getout_bs_rf1',
                                 'getout_bs_rf3', 'getoutplus', 'getout_redundant_actions',
                                 'threefish_human_assisted', 'threefishcolor', 'threefish_bs_top5', 'threefish_bs_rf3',
                                 'threefish_bs_rf1', 'threefish_redundant_actions',
                                 'loot_human_assisted', 'loot_bs_top5', 'loot_bs_rf3', 'loot_bs_rf1',
                                 'loot_redundant_actions'
                                 ])
    parser.add_argument("-l", "--log", help="record the information of games", type=bool, default=False, dest="log")
    parser.add_argument("--log_file_name", help="the name of log file", required=False, dest='logfile')
    parser.add_argument("--render", help="render the game", action="store_true", dest="render")
    # arg = ['-alg', 'human', '-m', 'getout', '-env', 'getout','-l','True']
    args = parser.parse_args()

    # fix seed
    make_deterministic(args.seed)

    # load trained_model
    # read filename from models
    current_path = os.path.dirname(__file__)

    if args.alg == "logic":
        model_name = "beam_search_top1.pth"
    else:
        models_folder = os.path.join(current_path, 'models', args.m, args.alg)
        print(f"Please use one of the following agent: {os.listdir(models_folder)}")
        model_name = input('Enter file name: ')
    model_file = os.path.join(current_path, 'models', args.m, args.alg, model_name)
    model = load_model(model_file, args)

    agent = LogicPlayer(args, model)

    # import ipdb;ipdb.set_trace()

    print("\n\n\nClauses")
    for clause in agent.model.clauses:
        print(clause)

    # #### Continue to render
    # if args.m == 'getout':
    #     render_getout(agent, args)
    # elif args.m == 'threefish':
    #     render_threefish(agent, args)
    # elif args.m == 'loot':
    #     render_loot(agent, args)
    # elif args.m == 'ecoinrun':
    #     render_ecoinrun(agent, args)
    # elif args.m == 'atari':
    #     render_atari(agent, args)



if __name__ == "__main__":
    main()
