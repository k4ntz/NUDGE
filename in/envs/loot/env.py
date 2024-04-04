from typing import Sequence

from nudge.env import NudgeBaseEnv
import numpy as np
from env_src.procgen.procgen import ProcgenGym3Env
import torch


class NudgeEnv(NudgeBaseEnv):
    name = "loot"
    pred2action = {
        'left': 1,
        'down': 3,
        'idle': 4,
        'up': 5,
        'right': 7,
    }
    pred_names: Sequence

    def __init__(self, mode: str, variant: str = "loot"):
        super().__init__(mode)
        assert variant in ["loot", "loothard", "lootplus", "lootcolor"]
        self.variant = variant
        self.env = ProcgenGym3Env(num=1, env_name='loot', render_mode=None)

    def reset(self):
        _, state, _ = self.env.observe()
        return self.convert_state(state)

    def step(self, action, is_mapped: bool = False):
        if not is_mapped:
            action = self.map_action(action)
        action = np.array([action])
        self.env.act(action)  # FIXME
        reward, state, done = self.env.observe()
        reward = reward[0]
        return self.convert_state(state), reward, done

    def extract_logic_state(self, raw_state):
        states = torch.from_numpy(raw_state['positions']).squeeze()
        if self.variant == "lootplus":
            # input shape: [X,Y]* [agent,key_b,door_b,key_g,door_g,key_r,door_r]
            # output shape:[agent, key, door, blue, green, red ,got_key, X, Y]
            state_extracted = torch.tensor([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 0]], dtype=torch.float64)
            state_extracted[:, -2:] = states[:]
            for i, state in enumerate(state_extracted):
                if state[-1] == 0 and state[-2] == 0:
                    state_extracted[i] = torch.zeros((1, 9))
                elif i in [2, 4, 6] and state[-1] != 0 and state_extracted[i - 1][1] == 0:
                    state_extracted[i][-3] = 1

        elif self.variant == "loothard":
            # input shape: [X,Y]* [agent, key_b, door_b, key_g, door_g, exit]
            # output shape:[agent, key, door, blue, green, exit,got_key, X, Y]
            state_extracted = torch.tensor([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=torch.float64)
            states = states[0:6]
            state_extracted[:, -2:] = states[:]
            for i, state in enumerate(state_extracted):
                if state[-1] == 0 and state[-2] == 0:
                    state_extracted[i] = torch.zeros((1, 9))
                elif i in [2, 4] and state[-1] != 0 and state_extracted[i - 1][1] == 0: # setting got_key
                    state_extracted[i][-3] = 1

        elif self.variant in ["loot", "lootcolor"]:
            # input shape: [X,Y]* [agent,key_b,door_b,key_g,door_g]
            # output shape:[agent, key, door, blue/green, red/brown ,got_key, X, Y]
            state_extracted = torch.tensor([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0]], dtype=torch.float64)
            states = states[0:5]
            state_extracted[:, -2:] = states[:]
            for i, state in enumerate(state_extracted):
                # 0 mean object does not exist
                if state[-1] == 0 and state[-2] == 0:
                    # then set to all attributes 0
                    state_extracted[i] = torch.zeros((1, 8))
                # if key = 0 but door !=0, means key of this door has picked
                elif i in [2, 4] and state[-1] != 0 and state_extracted[i - 1][1] == 0:
                    state_extracted[i][-3] = 1

        else:
            raise ValueError(f"Invalid Loot variant '{self.variant}'.")

        state_extracted = state_extracted.unsqueeze(0)
        return state_extracted

    def extract_neural_state(self, raw_state):
        state = raw_state['positions']

        if self.variant == 'lootplus':
            state_extracted = np.array([[0, 0, 0, 0],
                                  [0, 0, 1, 1],
                                  [0, 0, 2, 1],
                                  [0, 0, 1, 2],
                                  [0, 0, 2, 2],
                                  [0, 0, 1, 3],
                                  [0, 0, 2, 3]], dtype=np.float32)
            state_extracted[:, 0:2] = state[0][:]

        elif self.variant == 'loothard':
            state_extracted = np.array([[0, 0, 0, 0],
                                  [0, 0, 1, 1],
                                  [0, 0, 2, 1],
                                  [0, 0, 1, 2],
                                  [0, 0, 2, 2],
                                  [0, 0, 3, 0],
                                  [0, 0, 0, 0]], dtype=np.float32)
            state_extracted[:, 0:2] = state[0][:]

        elif self.variant == 'loot':
            state_extracted = np.array([[0, 0, 0, 0],
                                  [0, 0, 1, 1],
                                  [0, 0, 2, 1],
                                  [0, 0, 1, 2],
                                  [0, 0, 2, 2],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]], dtype=np.float32)
            state_extracted[:, 0:2] = state[0][:]

        elif self.variant == 'lootcolor':
            state_extracted = np.array([[0, 0, 0, 0],
                                  [0, 0, 1, 10],
                                  [0, 0, 2, 10],
                                  [0, 0, 1, 20],
                                  [0, 0, 2, 20],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]], dtype=np.float32)
            state_extracted[:, 0:2] = state[0][:]

        else:
            raise ValueError(f"Invalid Loot variant '{self.variant}'.")

        state_extracted = state_extracted.reshape(-1).tolist()
        return torch.tensor(state_extracted)

    def map_action(self, model_action) -> int:
        if self.mode == 'ppo':
            action_space = [1, 3, 4, 5, 7]
            return action_space[model_action]
        else:
            return super().map_action(model_action)

    def close(self):
        self.env.close()
