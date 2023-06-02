import torch
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from .facts_converter import FactsConverter
from .nsfr import NSFReasoner
from .logic_utils import build_infer_module, get_lang
from .valuation_cj import CJValuationModule
from .valuation_bf import BFValuationModule
from .valuation_h import HValuationModule
from .valuation_a import AValuationModule
from .valuation_aa import AAValuationModule

device = torch.device('cuda:0')


def get_nsfr_model(args, train=False):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = os.path.join(current_path, 'data/lang/')
    # device = torch.device('cuda:0')
    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, args.m, args.rules)
    if args.m == 'getout':
        VM = CJValuationModule(lang=lang, device=device)
    elif args.m == 'threefish':
        VM = BFValuationModule(lang=lang, device=device)
    elif args.m == 'loot':
        VM = HValuationModule(lang=lang, device=device)
    elif args.m == 'atari' and "freeway" in args.env.lower():
        VM = AValuationModule(lang=lang, device=device)
    elif args.m == 'atari' and "asterix" in args.env.lower():
        VM = AAValuationModule(lang=lang, device=device)
    FC = FactsConverter(lang=lang, valuation_module=VM, device=device)
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    m = len(prednames)
    # m = 5
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, train=train)
    return NSFR


def get_nsfr(mode, rule):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = os.path.join(current_path, 'data/lang/')

    device = torch.device('cuda:0')
    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, mode, rule)
    if mode == 'getout':
        VM = CJValuationModule(lang=lang, device=device)
    elif mode == 'threefish':
        VM = BFValuationModule(lang=lang, device=device)
    FC = FactsConverter(lang=lang, valuation_module=VM, device=device)
    m = len(clauses)
    # m = 5
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=True, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, train=True)
    return NSFR


def explaining_nsfr(NSFR, extracted_states):
    V_T = NSFR(extracted_states)
    # prednames = NSFR.prednames
    predicts = NSFR.predict_multi(v=V_T)
    explaining = NSFR.print_explaining(predicts)
    return explaining


def get_predictions(extracted_states, NSFR):
    predictions = NSFR(extracted_states)
    predicts = NSFR.print_explaining(predictions)
    return predicts


def extract_for_cgen_explaining(coin_jump):
    """
    extract state to metric
    input: coin_jump instance
    output: extracted_state to be explained

    x:  agent, key, door, enemy, position_X, position_Y
    y:  obj1(agent), obj2(key), obj3(door)，obj4(enemy)

    To be changed when using object-detection tech
    """
    # TODO
    num_of_feature = 6
    num_of_object = 4
    representation = coin_jump.level.get_representation()
    extracted_states = np.zeros((num_of_object, num_of_feature))
    for entity in representation["entities"]:
        if entity[0].name == 'PLAYER':
            extracted_states[0][0] = 1
            extracted_states[0][-2:] = entity[1:3]
            # 27 is the width of map, this is normalization
            # extracted_states[0][-2:] /= 27
        elif entity[0].name == 'KEY':
            extracted_states[1][1] = 1
            extracted_states[1][-2:] = entity[1:3]
            # extracted_states[1][-2:] /= 27
        elif entity[0].name == 'DOOR':
            extracted_states[2][2] = 1
            extracted_states[2][-2:] = entity[1:3]
            # extracted_states[2][-2:] /= 27
        elif entity[0].name == 'GROUND_ENEMY':
            extracted_states[3][3] = 1
            extracted_states[3][-2:] = entity[1:3]
            # extracted_states[3][-2:] /= 27

    if sum(extracted_states[:, 1]) == 0:
        key_picked = True
    else:
        key_picked = False

    def simulate_prob(extracted_states, num_of_objs, key_picked):
        for i, obj in enumerate(extracted_states):
            obj = add_noise(obj, i, num_of_objs)
            extracted_states[i] = obj
        if key_picked:
            extracted_states[:, 1] = 0
        return extracted_states

    def add_noise(obj, index_obj, num_of_objs):
        mean = torch.tensor(0.1)
        std = torch.tensor(0.05)
        noise = torch.abs(torch.normal(mean=mean, std=std)).item()
        rand_noises = torch.randint(1, 5, (num_of_objs - 1,)).tolist()
        rand_noises = [i * noise / sum(rand_noises) for i in rand_noises]
        rand_noises.insert(index_obj, 1 - noise)

        for i, noise in enumerate(rand_noises):
            obj[i] = rand_noises[i]
        return obj

    extracted_states = simulate_prob(extracted_states, num_of_object, key_picked)

    return torch.tensor(extracted_states, device="cuda:0")
