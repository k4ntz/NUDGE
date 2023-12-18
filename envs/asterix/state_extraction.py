import numpy as np


def extract_logic_state(state, **kwargs):
    n_features = 6
    n_objects = 11
    state_extracted = np.zeros((n_objects, n_features))

    for i, entity in enumerate(state):
        if entity.category == "Player":
            state_extracted[i][0] = 1
            state_extracted[i][-2:] = entity.xy
        elif entity.category == 'Enemy':
            state_extracted[i][1] = 1
            state_extracted[i][-2:] = entity.xy
        elif "Reward" in entity.category:
            state_extracted[i][2] = 1
            state_extracted[i][-2:] = entity.xy
        else:
            state_extracted[i][3] = 1
            state_extracted[i][-2:] = entity.xy

    return state_extracted


def extract_neural_state(state, **kwargs):
    raw_state = []
    for i, inst in enumerate(state):
        if inst.category == "Player" and i == 0:
            raw_state.append([1, 0, 0, 0] + list(inst.xy))
        elif inst.category == "Enemy":
            raw_state.append([0, 1, 0, 0] + list(inst.xy))
        elif "Reward" in inst.category:
            raw_state.append([0, 0, 1, 0] + list(inst.xy))
        else:
            raw_state.append([0, 0, 0, 1] + list(inst.xy))

    if len(raw_state) < 11:
        raw_state.extend([[0] * 6 for _ in range(11 - len(raw_state))])

    return raw_state
