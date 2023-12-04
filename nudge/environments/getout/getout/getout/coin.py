from enum import Enum

from .entity import Entity
from .entityEncoding import EntityID


class CoinColor(Enum):

    GOLD = 0
    #RED = 1


class Coin(Entity):

    def __init__(self, level, x, y, color=CoinColor.GOLD, resource_loader=None):
        super().__init__(level, EntityID.COIN, x, y)

        self.color = color
        self.sprites = self._load_sprites(resource_loader) if resource_loader is not None else None

    def _load_sprites(self, resource_loader):
        sprites = {}

        sprite_path = 'Items/'

        padding = 15
        animation_pad = 10
        sprites["coin"] = [
            resource_loader.get_sprite('coin_1', sprite_path+'coinGold.png', bbox=(32-padding,32-padding,96+padding,96+padding)),
            resource_loader.get_sprite('coin_2', sprite_path+'coinGold.png', bbox=(32-padding-animation_pad,32-padding,96+padding+animation_pad,96+padding)),
            resource_loader.get_sprite('coin_3', sprite_path+'coinGold.png', bbox=(32-padding-4*animation_pad,32-padding,96+padding+4*animation_pad,96+padding)),
            resource_loader.get_sprite('coin_4', sprite_path + 'coinGold.png', bbox=(32 - padding - animation_pad, 32 - padding, 96 + padding + animation_pad, 96 + padding)),
        ]
        return sprites

    def step(self):
        pass

    def _get_parameterization(self):
        return [self.color, 0, 0, 0]

    def render(self, camera, frame):
        sprite_sequence = self.sprites[self._get_state_string()]
        seq_id = frame // 3 % len(sprite_sequence)
        camera.paint_sprite(self.x - self.size[0] / 2, self.y, self.size, sprite_sequence[seq_id])

    def _get_state_string(self):
        return f"coin"
