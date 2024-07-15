from typing import Sequence
import torch
from nudge.env import NudgeBaseEnv
from ocatari.core import OCAtari
from hackatari.core import HackAtari
import numpy as np
import torch as th
from ocatari.ram.kangaroo import MAX_ESSENTIAL_OBJECTS
import gymnasium
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from utils import load_cleanrl_envs


from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def make_env(env):
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AutoResetWrapper(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


class NudgeEnv(NudgeBaseEnv):
    name = "kangaroo"
    pred2action = {
        'noop': 0,
        'fire': 1,
        'up': 2,
        'right': 3,
        'left': 4,
        'down': 5,
    }
    pred_names: Sequence

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False, seed=None):
        super().__init__(mode)
        self.env = HackAtari(env_name="ALE/Kangaroo-v5", mode="ram", obs_mode="ori", modifs=[("random_init"), ("disable_monkeys"), ("disable_coconut"), ("change_level0")],\
            render_mode=render_mode, render_oc_overlay=render_oc_overlay)
        # for learning script from cleanrl
        self.env._env = make_env(self.env._env)
        self.n_actions = 6
        self.n_raw_actions = 18
        self.n_objects = 49
        self.n_features = 4  # visible, x-pos, y-pos, right-facing
        self.seed = seed

        # Compute index offsets. Needed to deal with multiple same-category objects
        self.obj_offsets = {}
        offset = 0
        for (obj, max_count) in MAX_ESSENTIAL_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_ESSENTIAL_OBJECTS.keys())

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed)
        # raw_state, _ = self.raw_env.reset(seed=self.seed)
        # raw_state = raw_state.unsqueeze(0)
        state = self.env.objects
        raw_state = obs #self.env.dqn_obs
        logic_state, neural_state =  self.extract_logic_state(state), self.extract_neural_state(raw_state)
        # if len(logic_state.shape) == 2:
        logic_state = logic_state.unsqueeze(0)
        return logic_state, neural_state
        # return  self.convert_state(state, raw_state)

    def step(self, action, is_mapped: bool = False):
        # if not is_mapped:
        #     action = self.map_action(action)
        # step RAM env
        # obs, reward, done, _, _ = self.env.step(action)
        # action = array([2]) or action = torch.tensor(2)
        # try:
        #     assert action.shape[0] == 1, "invalid only 1 action for env.step"
        #     action = action[0]
        # except IndexError:
        #     action = action
        

            
        # obs, reward, done, truncations, infos = self.env.step(action)
        obs, reward, truncations, done, infos = self.env.step(action)
        
        # ste RGB env
        # x = self.raw_env.step(action.unsqueeze(0)) 
        # raw_obs, raw_reward, raw_done, _, _ = x
        # assert reward == raw_reward and done == raw_done, "Two envs conflict: rewards: {} and {}, dones: {} and {}".format(reward, raw_reward, done, raw_done)  
        # assert done == raw_done, "Two envs conflict: dones: {} and {}".format(done, raw_done)  
        state = self.env.objects
        raw_state = obs #self.env.dqn_obs
        # raw_state = raw_obs
        # raw_state = raw_state.unsqueeze(0)
        logic_state, neural_state = self.convert_state(state, raw_state)
        # if len(logic_state.shape) == 2:
        logic_state = logic_state.unsqueeze(0)
        return (logic_state, neural_state), reward, done, truncations, infos

    def extract_logic_state(self, input_state):
        """ in ocatari/ram/kangaroo.py :
        MAX_ESSENTIAL_OBJECTS = {
            'Player': 1,
            'Child': 1,
            'Fruit': 3,
            'Bell': 1,
            'Platform': 20,
            'Ladder': 6,
            'Monkey': 4,
            'FallingCoconut': 1,
            'ThrownCoconut': 3,
            'Life': 8,
            'Time': 1,}       
        """
        state = th.zeros((self.n_objects, self.n_features), dtype=th.int32)

        obj_count = {k: 0 for k in MAX_ESSENTIAL_OBJECTS.keys()}

        for obj in input_state:
            if obj.category not in self.relevant_objects:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            if obj.category == "Time":
                state[idx] = th.Tensor([1, obj.value, 0, 0])
            else:
                orientation = obj.orientation.value if obj.orientation is not None else 0
                state[idx] = th.tensor([1, *obj.center, orientation])
            obj_count[obj.category] += 1
        return state

    def extract_neural_state(self, raw_input_state):
        # print(raw_input_state.shape)
        return torch.Tensor(raw_input_state).unsqueeze(0)#.float()

    def close(self):
        self.env.close()
