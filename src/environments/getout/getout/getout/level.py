from .player import Player
from .block import Block, NoneBlock


class Level:

    def __init__(self, width, height):
        self.noneBlock = NoneBlock()

        # (27, 16)
        self.width = width
        self.height = height
        self.blocks: [[Block]] = [[self.noneBlock for _ in range(width)] for _ in range(height)]
        self.block_list = []
        self.entities = []

        self.meta = {}

        self.coins_collected = 0
        self.key_collected = 0

        self.terminated = False
        self.lost = False

        self.reward_values = {
            'coin': 1,
            'powerup': 1,
            'enemy': 1,
            'door': 20,
            'lose': -20,
            'key': 10,
            'step_depletion': -0.1,  # with 10fps and no reward the score decays at a rate of 1.0 per second
        }

        # self.gravity = 9.81
        self.gravity = 0.0981*2

        self.reward = 0

    def add_block(self, x, y, block):
        old_block = self.blocks[y][x]
        if old_block is not self.noneBlock:
            self.block_list.remove((x, y, old_block))
        self.blocks[y][x] = block
        self.block_list.append((x, y, block))

    def step(self):
        self.reward = 0
        self.take_reward(self.reward_values['step_depletion'])
        if self.terminated:
            return

        for entity in self.entities:
            entity.step()

    def add_coins(self, n):
        self.coins_collected += n

    def add_key(self, n):
        self.key_collected += n

    def get_key(self):
        return self.key_collected

    def take_reward(self, reward):
        self.reward += reward

    def get_reward(self):
        return self.reward

    def terminate(self, lost):
        if self.terminated:
            return
        if lost:
            self.take_reward(self.reward_values['lose'])
        self.lost = lost
        self.terminated = True

    def get_representation(self):
        entities = []
        for entity in self.entities:
            entities.append(entity.get_representation())
        representation = {
            "level": self.meta,
            "coins": self.coins_collected,
            "lost": self.lost,
            "entities": entities
        }
        return representation

    def render(self, camera, frame):
        for x, y, block in self.block_list:
            block.render(x, y, camera, frame)
        for entity in self.entities:
            entity.render(camera, frame)
