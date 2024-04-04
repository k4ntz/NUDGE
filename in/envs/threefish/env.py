from typing import Sequence

from nudge.env import NudgeBaseEnv
import numpy as np
from env_src.procgen.procgen import ProcgenGym3Env
import torch


class NudgeEnv(NudgeBaseEnv):
    name = "threefish"
    pred2action = {
        'left': 1,
        'down': 3,
        'idle': 4,
        'up': 5,
        'right': 7,
    }
    pred_names: Sequence

    def __init__(self, mode: str, variant: str = "threefish"):
        super().__init__(mode)
        assert variant in ["threefish", "threefishcolor"]
        self.variant = variant
        self.env = ProcgenGym3Env(num=1, env_name='threefish', render_mode=None)

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
        raw_state = torch.from_numpy(raw_state['positions']).squeeze()

        if self.variant == "threefish":
            # input shape: [X,Y,radius]
            # output shape:[agent, fish, radius, X, Y]
            logic_state = torch.zeros((3, 5))
            for i, raw_state in enumerate(raw_state):
                if i == 0:
                    logic_state[i, 0] = 1  # agent
                    logic_state[i, 2] = raw_state[i, 2]  # radius
                    logic_state[i, 3] = raw_state[i, 0]  # X
                    logic_state[i, 4] = raw_state[i, 1]  # Y
                else:
                    logic_state[i, 1] = 1  # fish
                    logic_state[i, 2] = raw_state[i, 2]  # radius
                    logic_state[i, 3] = raw_state[i, 0]  # X
                    logic_state[i, 4] = raw_state[i, 1]  # Y

        elif self.variant == "threefishcolor":
            # input shape: [X, Y, color, radius]
            # output shape: [agent, fish, green, red,radius, X, Y]
            logic_state = torch.zeros((3, 7))
            for i, raw_state in enumerate(raw_state):
                if i == 0:
                    logic_state[i, 0] = 1  # agent
                    logic_state[i, -3] = raw_state[i, 3]  # radius
                    logic_state[i, -2] = raw_state[i, 0]  # X
                    logic_state[i, -1] = raw_state[i, 1]  # Y
                else:
                    logic_state[i, 1] = 1  # fish
                    if raw_state[i, 2] == 1:
                        logic_state[i, 2] = 1  # green
                    else:
                        logic_state[i, 3] = 1  # red
                    logic_state[i, -3] = raw_state[i, 3]  # radius
                    logic_state[i, -2] = raw_state[i, 0]  # X
                    logic_state[i, -1] = raw_state[i, 1]  # Y

        else:
            raise ValueError(f"Invalid ThreeFish variant '{self.variant}'.")

        logic_state = logic_state.unsqueeze(0)
        return logic_state.cuda()

    def extract_neural_state(self, raw_state):
        neural_state = raw_state['positions'][:, :, 0:3].reshape(-1).tolist()
        return torch.tensor(neural_state)

    def map_action(self, model_action) -> int:
        if self.mode == 'ppo':
            action_space = [1, 3, 4, 5, 7]
            return action_space[model_action]
        else:
            return super().map_action(model_action)

    def close(self):
        self.env.close()
