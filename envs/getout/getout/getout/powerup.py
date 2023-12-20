from .entity import Entity
from .entityEncoding import EntityID
from .resource_loader import rotate


class PowerUp(Entity):

    def __init__(self, level, x, y, resource_loader=None):
        super().__init__(level, EntityID.POWERUP, x, y)

        self.sprites = self._load_sprites(resource_loader) if resource_loader is not None else None

    def _load_sprites(self, resource_loader):
        sprites = {}

        is_show_jewel = False

        if is_show_jewel:
            sprite_path = 'Items/'

            padding = 15
            animation_pad = 5

            powerup_file = 'gemRed.png' # 'star.png'
            sprites["powerup"] = [
                resource_loader.get_sprite('gem_1', sprite_path+powerup_file, bbox=(32-padding,32-padding,96+padding,96+padding)),
                resource_loader.get_sprite('gem_2', sprite_path+powerup_file, bbox=(
                    32-padding-animation_pad,
                    32-padding-animation_pad,
                    96+padding+animation_pad,
                    96+padding+animation_pad))
            ]
        else:
            powerup_file = '../custom/Items/sword.png'
            sprites["powerup"] = [
                resource_loader.get_sprite('sword_1', powerup_file, pad=45),
                resource_loader.get_sprite('sword_2', powerup_file, pad=30)
            ]

        return sprites

    def step(self):
        pass

    def render(self, camera, frame):
        sprite_sequence = self.sprites[self._get_state_string()]
        seq_id = frame // 3 % len(sprite_sequence)
        camera.paint_sprite(self.x - self.size[0] / 2, self.y, self.size, sprite_sequence[seq_id])

    def _get_state_string(self):
        return f"powerup"
