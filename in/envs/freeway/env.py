# from typing import Sequence
#
# from nudge.env import NudgeBaseEnv
# from ocatari.core import OCAtari
# import numpy as np
#
#
# class NudgeEnv(NudgeBaseEnv):
#     name = "freeway"
#     pred2action = {
#         'noop': 0,
#         'up': 1,
#         'down': 2,
#     }
#     pred_names: Sequence
#
#     def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False):
#         super().__init__(mode)
#         self.env = OCAtari(env_name="ALE/Freeway-v5", mode="ram",
#                            render_mode=render_mode, render_oc_overlay=render_oc_overlay)
#
#     def reset(self):
#         self.env.reset()
#         state = self.env.objects
#         return self.convert_state(state)
#
#     def step(self, action, is_mapped: bool = False):
#         if not is_mapped:
#             action = self.map_action(action)
#         _, reward, terminated, truncated, _ = self.env.step(action)
#         done = terminated or truncated
#         state = self.env.objects
#         return self.convert_state(state), reward, done
#
#     def extract_logic_state(self, raw_state):
#         num_of_feature = 6
#         num_of_object = 11
#         logic_state = np.zeros((num_of_object, num_of_feature))
#
#         for i, entity in enumerate(raw_state):
#             if entity.category == "Chicken" and i == 0:
#                 logic_state[0][0] = 1
#                 logic_state[0][-2:] = entity.xy
#             elif entity.category == 'Car':
#                 logic_state[i - 1][1] = 1
#                 logic_state[i - 1][-2:] = entity.xy
#
#         return logic_state
#
#     def extract_neural_state(self, raw_state):
#         neural_state = []
#         for i, inst in enumerate(raw_state):
#             if inst.category == "Chicken" and i == 1:
#                 neural_state.append([1, 0, 0, 0] + list(inst.xy))
#             elif inst.category == "Car":
#                 neural_state.append([0, 1, 0, 0] + list(inst.xy))
#
#         return np.array(neural_state).reshape(-1)
#
#     def close(self):
#         self.env.close()

from typing import Sequence
from nudge.env import NudgeBaseEnv
from ocatari.core import OCAtari
import numpy as np

class NudgeEnv(NudgeBaseEnv):
    name = "freeway"
    pred2action = {'noop': 0,
                   'up': 1,
                   'down': 2}
    pred_names: Sequence

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False,
                 max_cars: int = 10, lane_height: float = 14.0):
        super().__init__(mode)
        self.env = OCAtari(env_name="ALE/Freeway-v5", mode="ram",
                           render_mode=render_mode, render_oc_overlay=render_oc_overlay)
        self.MAX_CARS = max_cars
        self.NUM_FEAT = 6           # 4 type one-hot + x + y
        self.LANE_H = lane_height

        self.collision_penalty = -1  # tune: -0.1 .. -0.5 usually works
        self.COLL_DX = 8.0  # horiz proximity for collision (px)
        self.COLL_DY = 6.0  # vert proximity for collision (px)

    def _is_chicken(o):
        return getattr(o, "category", "") in ("Chicken", "chicken")

    @staticmethod
    def _is_car(o):
        return getattr(o, "category", "") in ("Car", "car")

    @staticmethod
    def _xy(o):
        # OCAtari objects expose .xy (x,y) in pixels
        return np.array(getattr(o, "xy", (0.0, 0.0)), dtype=np.float32)

    def _collision_happened(self, objs) -> bool:
        """Heuristic: chicken overlaps a car within (COLL_DX, COLL_DY)."""
        chickens = [o for o in objs if self._is_chicken(o)]
        cars = [o for o in objs if self._is_car(o)]
        if not chickens:
            return False
        xch, ych = self._xy(chickens[0])
        for c in cars:
            xc, yc = self._xy(c)
            if abs(xc - xch) <= self.COLL_DX and abs(yc - ych) <= self.COLL_DY:
                return True
        return False

    def reset(self):
        self.env.reset()
        return self.convert_state(self.env.objects)

    def step(self, action, is_mapped: bool = False):
        if not is_mapped:
            action = self.map_action(action)

        _, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        objs = self.env.objects
        total_reward = float(reward)

        # ---- collision shaping ----
        if self._collision_happened(objs):
            total_reward += self.collision_penalty

        # ---- action shaping ----
        if action == self.pred2action['up']:
            total_reward += 0.005     # small penalty/bonus for moving up
        elif action == self.pred2action['down']:
            total_reward += 0.1    # small penalty/bonus for moving down
        elif action == self.pred2action['noop']:
            total_reward += 0.1   # small penalty/bonus for standing still

        state = self.convert_state(objs)
        return state, total_reward, done





    # ---------- helpers ----------
    @staticmethod
    def _is_chicken(obj): return getattr(obj, "category", "") in ("Chicken", "chicken")
    @staticmethod
    def _is_car(obj):     return getattr(obj, "category", "") in ("Car", "car")

    def _split_objects(self, raw_state):
        chickens = [o for o in raw_state if self._is_chicken(o)]
        cars     = [o for o in raw_state if self._is_car(o)]
        chicken  = chickens[0] if chickens else None
        # Stable sort for determinism. You can also sort by distance to chicken.
        cars.sort(key=lambda o: (o.xy[1], o.xy[0]))     # by y then x
        return chicken, cars[: self.MAX_CARS]

    # ---------- logic state (matrix) ----------
    def extract_logic_state(self, raw_state):
        chicken, cars = self._split_objects(raw_state)
        num_objects = 1 + self.MAX_CARS
        ls = np.zeros((num_objects, self.NUM_FEAT), dtype=np.float32)

        # Row 0: chicken
        if chicken is not None:
            ls[0, 0] = 1.0                               # [1,0,0,0]
            ls[0, -2:] = chicken.xy

        # Rows 1..K: cars
        for idx, car in enumerate(cars, start=1):
            ls[idx, 1] = 1.0                             # [0,1,0,0]
            ls[idx, -2:] = car.xy

        return ls

    # ---------- neural state (flat vector) ----------
    def extract_neural_state(self, raw_state):
        chicken, cars = self._split_objects(raw_state)
        entries = []

        # chicken first
        if chicken is not None:
            entries.append([1, 0, 0, 0, *chicken.xy])
        else:
            entries.append([1, 0, 0, 0, 0.0, 0.0])

        # cars next (pad/truncate to MAX_CARS)
        for car in cars:
            entries.append([0, 1, 0, 0, *car.xy])
        # pad if fewer cars
        for _ in range(self.MAX_CARS - len(cars)):
            entries.append([0, 1, 0, 0, 0.0, 0.0])

        arr = np.asarray(entries, dtype=np.float32)      # shape [1+MAX_CARS, 6]
        return arr.reshape(-1)

    def close(self):
        self.env.close()
