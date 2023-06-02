import math
import random

from .block import Block
from .getout import Getout
from .door import Door
from .key import Key
from .groundEnemy import GroundEnemy, GroundEnemy2, GroundEnemy3, BuzzSaw1, BuzzSaw2


class ParameterizedLevelGenerator:

    def __init__(self, enemy=False, enemies=False, key_door=False, print_seed=False):
        self.print_seed = print_seed
        self.enemy = enemy
        self.key_door = key_door
        self.enemies = enemies

    def generate(self, getout: Getout, seed=None, dynamic_reward=False, spawn_all_entities=False):
        seed = random.randint(0, 500) if seed is None else seed
        rng = random.Random()
        rng.seed(seed, version=3)
        if self.print_seed:
            print("seed", seed)

        level = getout.level
        resource_loader = getout.resource_loader

        grassSprite = resource_loader.get_sprite('rock',
                                                 'Ground/Grass/grass.png') if resource_loader is not None else None
        snowSprite = resource_loader.get_sprite('snow', 'Ground/Snow/snow.png') if resource_loader is not None else None
        solidBlock = Block(True, False, False, grassSprite)
        snowBlock = Block(True, False, False, snowSprite)
        for x in range(level.width):
            level.add_block(x, 0, solidBlock)
            level.add_block(x, 1, solidBlock)
            level.add_block(x, level.height - 1, solidBlock)
            level.add_block(x, level.height - 2, solidBlock)

        for y in range(level.height):
            level.add_block(0, y, solidBlock)
            level.add_block(1, y, solidBlock)
            level.add_block(level.width - 1, y, solidBlock)
            level.add_block(level.width - 2, y, solidBlock)

        positions = [
            (7, 2),
            (11, 2),
            (15, 2),
            (19, 2),
            (23, 2),
            (9, 2),
            (3, 2),
            (27, 2),
            (31, 2)
        ]

        if getout.width > 27:
            positions = [(max(min(getout.width-5, int(pos[0] * getout.width / 27)), 3), pos[1]) for pos in positions]

        # positions = [
        #     (6, 2),
        #     (8, 2),
        #     (10, 2),
        #     (12, 2),
        #     (14, 2),
        #     (16, 2),
        #     (18, 2),
        # ]

        rng.shuffle(positions)

        getout.player.x = positions[0][0] - 0.5
        getout.player.y = positions[0][1]

        if self.enemy:
            level.entities.append(GroundEnemy(level, positions[3][0], positions[3][1], resource_loader=resource_loader))
        elif self.enemies:
            level.entities.append(Key(level, positions[1][0] - 0.5, positions[1][1], resource_loader=resource_loader))
            # level.entities.append(Key(level, positions[6][0] - 0.5, positions[1][1], resource_loader=resource_loader))
            level.entities.append(Door(level, positions[2][0], positions[2][1], resource_loader=resource_loader))
            level.entities.append(GroundEnemy(level, positions[3][0], positions[3][1], resource_loader=resource_loader))
            level.entities.append(GroundEnemy2(level, positions[4][0], positions[4][1], resource_loader=resource_loader))
            level.entities.append(GroundEnemy3(level, positions[5][0], positions[5][1], resource_loader=resource_loader))
            level.entities.append(BuzzSaw1(level, positions[6][0], positions[6][1], resource_loader=resource_loader))
            level.entities.append(BuzzSaw2(level, positions[7][0], positions[7][1], resource_loader=resource_loader))
            # level.entities.append(GroundEnemy(level, positions[6][0], positions[6][1], resource_loader=resource_loader))
            # level.entities.append(GroundEnemy3(level, positions[4][0], positions[4][1], resource_loader=resource_loader))
            # level.entities.append(GroundEnemy4(level, positions[4][0], positions[4][1], resource_loader=resource_loader))
        elif self.key_door:
            level.entities.append(Key(level, positions[1][0] - 0.5, positions[1][1], resource_loader=resource_loader))
            level.entities.append(Door(level, positions[2][0], positions[2][1], resource_loader=resource_loader))
        else:
            level.entities.append(Key(level, positions[1][0] - 0.5, positions[1][1], resource_loader=resource_loader))
            level.entities.append(Door(level, positions[2][0], positions[2][1], resource_loader=resource_loader))
            level.entities.append(GroundEnemy(level, positions[3][0], positions[3][1], resource_loader=resource_loader))
            # level.entities.append(GroundEnemy(level, positions[4][0], positions[4][1], resource_loader=resource_loader))
            # level.entities.append(Door(level, 21, 2, resource_loader=resource_loader))

        # setup rewards
        if dynamic_reward:
            reward_value = 5
            getout.level.reward_values['coin'] = min(1, rng.randint(0, 3)) * reward_value
            getout.level.reward_values['powerup'] = min(1, rng.randint(0, 3)) * reward_value
            getout.level.reward_values['enemy'] = min(1, rng.randint(0, 3)) * reward_value

            # with a quarter probability make a reward negative
            if rng.randint(0, 3) == 0:
                neg_reward = ['coin', 'powerup', 'enemy'][rng.randint(0, 2)]
                getout.level.reward_values[neg_reward] = -getout.level.reward_values[neg_reward]
        else:
            # getout1.level.reward_values['coin'] = 5
            getout.level.reward_values['coin'] = 3
            getout.level.reward_values['powerup'] = 1
            getout.level.reward_values['enemy'] = 9
            getout.level.reward_values['key'] = 10
            getout.level.reward_values['door'] = 20

        level.meta["seed"] = seed
        level.meta["reward_coin"] = getout.level.reward_values['coin']
        level.meta["reward_powerup"] = getout.level.reward_values['powerup']
        level.meta["reward_enemy"] = getout.level.reward_values['enemy']
        level.meta["reward_key"] = getout.level.reward_values['key']
        level.meta["reward_door"] = getout.level.reward_values['door']
