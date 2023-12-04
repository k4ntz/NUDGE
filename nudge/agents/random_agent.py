import random
import numpy as np


class RandomPlayer:
    def __init__(self, args):
        self.args = args
        self.nb_action = None

    def act(self, state):
        # TODO how to do if-else only once?
        if self.args.m == 'getout':
            action = self.getout_actor()
        elif self.args.m == 'threefish':
            action = self.threefish_actor()
        elif self.args.m == 'loot':
            action = self.loot_actor()
        elif "atari" in self.args.m:
            action = random.randint(0, self.nb_actions-1)
        return action

    def getout_actor(self):
        # action = coin_jump_actions_from_unified(random.randint(0, 10))
        return random.randint(0, 10)

    def threefish_actor(self):
        return np.random.randint([9])

    def loot_actor(self):
        return np.random.randint([9])
