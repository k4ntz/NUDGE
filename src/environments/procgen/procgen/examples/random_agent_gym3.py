"""
Example random agent script using the gym3 API to demonstrate that procgen works
"""
import gym3
from gym3 import types_np
from procgen import ProcgenGym3Env

env = ProcgenGym3Env(num=1, env_name="threefish", render_mode="rgb_array")
env = gym3.ViewerWrapper(env, info_key="rgb")
step = 0
while True:
    a = env.ac_space
    b = types_np.sample(env.ac_space, bshape=(env.num,))
    env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
    rew, obs, first = env.observe()
    # states = env.callmethod("get_state")
    print(f"step {step} reward {rew} first {first}")
    if step > 0 and first:
        break
    step += 1
