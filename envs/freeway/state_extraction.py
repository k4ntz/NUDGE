import numpy as np


def extract_logic_state(state, **kwargs):
    num_of_feature = 6
    num_of_object = 11
    state_extracted = np.zeros((num_of_object, num_of_feature))

    for i, entity in enumerate(state):
        if entity.category == "Chicken" and i == 0:
            state_extracted[0][0] = 1
            state_extracted[0][-2:] = entity.xy
        elif entity.category == 'Car':
            state_extracted[i - 1][1] = 1
            state_extracted[i - 1][-2:] = entity.xy

    return state_extracted


def extract_neural_state(state, **kwargs):
    state_extracted = []
    for i, inst in enumerate(state):
        if inst.category == "Chicken" and i == 1:
            state_extracted.append([1, 0, 0, 0] + list(inst.xy))
        elif inst.category == "Car":
            state_extracted.append([0, 1, 0, 0] + list(inst.xy))

    return np.array(state_extracted).reshape(-1)
