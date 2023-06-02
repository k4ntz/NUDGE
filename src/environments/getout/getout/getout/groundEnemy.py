import time

from .entity import Entity
from .entityEncoding import EntityID
from .resource_loader import flip_horizontal


class GroundEnemy(Entity):

    def __init__(self, level, x, y, resource_loader=None):
        super().__init__(level, EntityID.GROUND_ENEMY, x, y)

        self.floating = True # avoid y computation

        self.facing_left = True

        self.size = (1.0, 0.5)
        self._update_bounding_box()

        self.move_speed = 0.49
        self.max_movement = 0.49
        self.sprites = self._load_sprites(resource_loader) if resource_loader is not None else None

        self.can_be_killed_by_jump = False

        self.is_enemy = True
        #self.last_time = time.time()

    def _load_sprites(self, resource_loader):
        sprites = {}

        sprite_path = 'Enemies/'

        sprites["walking_left"] = [
            resource_loader.get_sprite('slimeBlue_walking1_l', sprite_path+'slimeBlue.png'),
            #resource_loader.get_sprite('slimeBlue_walking2_l', sprite_path+'slimeBlue_move.png')
        ]
        sprites["walking_right"] = [
            resource_loader.get_sprite('slimeBlue_walking1_r', sprite_path+'slimeBlue.png', transform=flip_horizontal),
            #resource_loader.get_sprite('slimeBlue_walking2_r', sprite_path+'slimeBlue_move.png', transform=flip_horizontal)
        ]
        return sprites

    def step(self):

        self.ax = -self.move_speed if self.facing_left else self.move_speed

        super(GroundEnemy, self).step()

        # if collision with a wall change direction
        if self.collision_x:
            self.facing_left = not self.facing_left

        # if time.time() - self.last_time >= 5:
        #     self._speed_up()
        #     self.last_time = time.time()

    def render(self, camera, frame):
        sprite_sequence = self.sprites[self._get_state_string()]
        seq_id = frame // 2 % len(sprite_sequence)
        camera.paint_sprite(self.x - self.size[0] / 2, self.y, self.size, sprite_sequence[seq_id])

    def _get_state_string(self):
        return f"walking_{'left' if self.facing_left else 'right'}"

    # def _speed_up(self):
    #     self.move_speed += 0.1


class GroundEnemy2(Entity):

    def __init__(self, level, x, y, resource_loader=None):
        super().__init__(level, EntityID.GROUND_ENEMY2, x, y)

        self.floating = True # avoid y computation

        self.facing_left = True

        self.size = (1.0, 0.5)
        self._update_bounding_box()

        self.move_speed = 0.39
        self.max_movement = 0.49
        self.sprites = self._load_sprites(resource_loader) if resource_loader is not None else None

        self.can_be_killed_by_jump = False

        self.is_enemy = True
        #self.last_time = time.time()

    def _load_sprites(self, resource_loader):
        sprites = {}

        sprite_path = 'Enemies/'

        sprites["walking_left"] = [
            resource_loader.get_sprite('slimeBlue_walking1_l', sprite_path+'slimeBlue.png'),
            #resource_loader.get_sprite('slimeBlue_walking2_l', sprite_path+'slimeBlue_move.png')
        ]
        sprites["walking_right"] = [
            resource_loader.get_sprite('slimeBlue_walking1_r', sprite_path+'slimeBlue.png', transform=flip_horizontal),
            #resource_loader.get_sprite('slimeBlue_walking2_r', sprite_path+'slimeBlue_move.png', transform=flip_horizontal)
        ]
        return sprites

    def step(self):

        self.ax = -self.move_speed if self.facing_left else self.move_speed

        super(GroundEnemy2, self).step()

        # if collision with a wall change direction
        if self.collision_x:
            self.facing_left = not self.facing_left

        # if time.time() - self.last_time >= 5:
        #     self._speed_up()
        #     self.last_time = time.time()

    def render(self, camera, frame):
        sprite_sequence = self.sprites[self._get_state_string()]
        seq_id = frame // 2 % len(sprite_sequence)
        camera.paint_sprite(self.x - self.size[0] / 2, self.y, self.size, sprite_sequence[seq_id])

    def _get_state_string(self):
        return f"walking_{'left' if self.facing_left else 'right'}"


class GroundEnemy3(Entity):

    def __init__(self, level, x, y, resource_loader=None):
        super().__init__(level, EntityID.GROUND_ENEMY3, x, y)

        self.floating = True # avoid y computation

        self.facing_left = True

        self.size = (1.0, 0.5)
        self._update_bounding_box()

        self.move_speed = 0.39
        self.max_movement = 0.49
        self.sprites = self._load_sprites(resource_loader) if resource_loader is not None else None

        self.can_be_killed_by_jump = False

        self.is_enemy = True
        #self.last_time = time.time()

    def _load_sprites(self, resource_loader):
        sprites = {}

        sprite_path = 'Enemies/'

        sprites["walking_left"] = [
            resource_loader.get_sprite('slimeGreen_walking1_l', sprite_path+'slimeGreen.png'),
            #resource_loader.get_sprite('slimeBlue_walking2_l', sprite_path+'slimeBlue_move.png')
        ]
        sprites["walking_right"] = [
            resource_loader.get_sprite('slimeGreen_walking1_r', sprite_path+'slimeGreen.png', transform=flip_horizontal),
            #resource_loader.get_sprite('slimeBlue_walking2_r', sprite_path+'slimeBlue_move.png', transform=flip_horizontal)
        ]
        return sprites

    def step(self):

        self.ax = -self.move_speed if self.facing_left else self.move_speed

        super(GroundEnemy3, self).step()

        # if collision with a wall change direction
        if self.collision_x:
            self.facing_left = not self.facing_left

        # if time.time() - self.last_time >= 5:
        #     self._speed_up()
        #     self.last_time = time.time()

    def render(self, camera, frame):
        sprite_sequence = self.sprites[self._get_state_string()]
        seq_id = frame // 2 % len(sprite_sequence)
        camera.paint_sprite(self.x - self.size[0] / 2, self.y, self.size, sprite_sequence[seq_id])

    def _get_state_string(self):
        return f"walking_{'left' if self.facing_left else 'right'}"


class GroundEnemy4(Entity):

    def __init__(self, level, x, y, resource_loader=None):
        super().__init__(level, EntityID.GROUND_ENEMY4, x, y)

        self.floating = True # avoid y computation

        self.facing_left = True

        self.size = (1.0, 0.5)
        self._update_bounding_box()

        self.move_speed = 0.39
        self.max_movement = 0.49
        self.sprites = self._load_sprites(resource_loader) if resource_loader is not None else None

        self.can_be_killed_by_jump = False

        self.is_enemy = True
        #self.last_time = time.time()

    def _load_sprites(self, resource_loader):
        sprites = {}

        sprite_path = 'Enemies/'

        sprites["walking_left"] = [
            resource_loader.get_sprite('slimeGreen_walking1_l', sprite_path+'slimeGreen.png'),
            #resource_loader.get_sprite('slimeBlue_walking2_l', sprite_path+'slimeBlue_move.png')
        ]
        sprites["walking_right"] = [
            resource_loader.get_sprite('slimeGreen_walking1_r', sprite_path+'slimeGreen.png', transform=flip_horizontal),
            #resource_loader.get_sprite('slimeBlue_walking2_r', sprite_path+'slimeBlue_move.png', transform=flip_horizontal)
        ]
        return sprites

    def step(self):

        self.ax = -self.move_speed if self.facing_left else self.move_speed

        super(GroundEnemy4, self).step()

        # if collision with a wall change direction
        if self.collision_x:
            self.facing_left = not self.facing_left

        # if time.time() - self.last_time >= 5:
        #     self._speed_up()
        #     self.last_time = time.time()

    def render(self, camera, frame):
        sprite_sequence = self.sprites[self._get_state_string()]
        seq_id = frame // 2 % len(sprite_sequence)
        camera.paint_sprite(self.x - self.size[0] / 2, self.y, self.size, sprite_sequence[seq_id])

    def _get_state_string(self):
        return f"walking_{'left' if self.facing_left else 'right'}"


class GroundEnemy5(Entity):

    def __init__(self, level, x, y, resource_loader=None):
        super().__init__(level, EntityID.GROUND_ENEMY5, x, y)

        self.floating = True # avoid y computation

        self.facing_left = True

        self.size = (1.0, 0.5)
        self._update_bounding_box()

        self.move_speed = 0.39
        self.max_movement = 0.49
        self.sprites = self._load_sprites(resource_loader) if resource_loader is not None else None

        self.can_be_killed_by_jump = False

        self.is_enemy = True
        #self.last_time = time.time()

    def _load_sprites(self, resource_loader):
        sprites = {}

        sprite_path = 'Enemies/'

        sprites["walking_left"] = [
            resource_loader.get_sprite('slimeGreen_walking1_l', sprite_path+'slimeGreen.png'),
            #resource_loader.get_sprite('slimeBlue_walking2_l', sprite_path+'slimeBlue_move.png')
        ]
        sprites["walking_right"] = [
            resource_loader.get_sprite('slimeGreen_walking1_r', sprite_path+'slimeGreen.png', transform=flip_horizontal),
            #resource_loader.get_sprite('slimeBlue_walking2_r', sprite_path+'slimeBlue_move.png', transform=flip_horizontal)
        ]
        return sprites

    def step(self):

        self.ax = -self.move_speed if self.facing_left else self.move_speed

        super(GroundEnemy5, self).step()

        # if collision with a wall change direction
        if self.collision_x:
            self.facing_left = not self.facing_left

        # if time.time() - self.last_time >= 5:
        #     self._speed_up()
        #     self.last_time = time.time()

    def render(self, camera, frame):
        sprite_sequence = self.sprites[self._get_state_string()]
        seq_id = frame // 2 % len(sprite_sequence)
        camera.paint_sprite(self.x - self.size[0] / 2, self.y, self.size, sprite_sequence[seq_id])

    def _get_state_string(self):
        return f"walking_{'left' if self.facing_left else 'right'}"


class BuzzSaw1(Entity):

    def __init__(self, level, x, y, resource_loader=None):
        super().__init__(level, EntityID.BUZZSAW1, x, y)

        self.floating = True # avoid y computation

        self.facing_left = True

        self.size = (1.0, 0.5)
        self._update_bounding_box()

        self.move_speed = 0.
        self.max_movement = 0.49
        self.sprites = self._load_sprites(resource_loader) if resource_loader is not None else None

        self.can_be_killed_by_jump = False

        self.is_enemy = True
        #self.last_time = time.time()

    def _load_sprites(self, resource_loader):
        sprites = {}

        sprite_path = 'Enemies/'

        sprites["walking_left"] = [
            resource_loader.get_sprite('saw_l', sprite_path+'sawHalf.png'),
            #resource_loader.get_sprite('slimeBlue_walking2_l', sprite_path+'slimeBlue_move.png')
        ]
        sprites["walking_right"] = [
            resource_loader.get_sprite('saw_r', sprite_path+'sawHalf_move.png'),
            #resource_loader.get_sprite('slimeBlue_walking2_r', sprite_path+'slimeBlue_move.png', transform=flip_horizontal)
        ]
        return sprites

    def step(self):

        self.ax = -self.move_speed if self.facing_left else self.move_speed

        super(BuzzSaw1, self).step()

        self.facing_left = not self.facing_left

        # if time.time() - self.last_time >= 5:
        #     self._speed_up()
        #     self.last_time = time.time()

    def render(self, camera, frame):
        sprite_sequence = self.sprites[self._get_state_string()]
        seq_id = frame // 2 % len(sprite_sequence)
        camera.paint_sprite(self.x - self.size[0] / 2, self.y, self.size, sprite_sequence[seq_id])

    def _get_state_string(self):
        return f"walking_{'left' if self.facing_left else 'right'}"
    

class BuzzSaw2(Entity):

    def __init__(self, level, x, y, resource_loader=None):
        super().__init__(level, EntityID.BUZZSAW2, x, y)

        self.floating = True # avoid y computation

        self.facing_left = True

        self.size = (1.0, 0.5)
        self._update_bounding_box()

        self.move_speed = 0.
        self.max_movement = 0.49
        self.sprites = self._load_sprites(resource_loader) if resource_loader is not None else None

        self.can_be_killed_by_jump = False

        self.is_enemy = True
        #self.last_time = time.time()

    def _load_sprites(self, resource_loader):
        sprites = {}

        sprite_path = 'Enemies/'

        sprites["walking_left"] = [
            resource_loader.get_sprite('saw_l', sprite_path+'sawHalf.png'),
            #resource_loader.get_sprite('slimeBlue_walking2_l', sprite_path+'slimeBlue_move.png')
        ]
        sprites["walking_right"] = [
            resource_loader.get_sprite('saw_r', sprite_path+'sawHalf_move.png', transform=flip_horizontal),
            #resource_loader.get_sprite('slimeBlue_walking2_r', sprite_path+'slimeBlue_move.png', transform=flip_horizontal)
        ]
        return sprites

    def step(self):

        self.ax = -self.move_speed if self.facing_left else self.move_speed

        super(BuzzSaw2, self).step()

        # if collision with a wall change direction
        self.facing_left = not self.facing_left

        # if time.time() - self.last_time >= 5:
        #     self._speed_up()
        #     self.last_time = time.time()

    def render(self, camera, frame):
        sprite_sequence = self.sprites[self._get_state_string()]
        seq_id = frame // 2 % len(sprite_sequence)
        camera.paint_sprite(self.x - self.size[0] / 2, self.y, self.size, sprite_sequence[seq_id])

    def _get_state_string(self):
        return f"walking_{'left' if self.facing_left else 'right'}"