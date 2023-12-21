from enum import Enum

from .entity import Entity
from .entityEncoding import EntityID

class Key(Entity):

    def __init__(self, level, x, y,  resource_loader=None):
        super().__init__(level, EntityID.KEY, x, y)

        self.sprites = self._load_sprites(resource_loader) if resource_loader is not None else None

    def _load_sprites(self, resource_loader):
        sprites = {}

        sprite_path = 'Items/'

        sprites["key"] = [
            resource_loader.get_sprite('key_1', sprite_path+'keyBlue.png')
        ]
        return sprites

    def step(self):
        pass

    def render(self, camera, frame):
        sprite_sequence = self.sprites[self._get_state_string()]
        seq_id = frame // 3 % len(sprite_sequence)
        camera.paint_sprite(self.x - self.size[0] / 2, self.y, self.size, sprite_sequence[seq_id])

    def _get_state_string(self):
        return f"key"
