from .boundingbox import BoundingBox

def sign(x):
    return -1 if x < 0 else 0 if x == 0 else 1


class Entity:

    def __init__(self, level, entity_id, x, y):
        self.level = level

        self._entity_id = entity_id

        # position is the bottom center of the entity
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0

        self.ax = 0
        self.ay = 0

        self.is_enemy = False

        self.size = (1.0, 1.0) #assumption is never greater than (1,1)
        self.halfWidth = self.size[0] / 2

        self.is_supported = False
        self.flying = False
        self.floating = False

        self.use_support = True
        self.solid_ground = False

        self.max_movement = 0.49 # if above 0.5 we need to add additional checks to not glitch into blocks
        self.vx_decay = 0.5
        self.max_fall_speed = 0.9

        self.air_control_factor = 0.3

        self.collision_x = False
        self.collision_y = False

        self.graphic = None

        self.checks_collision = False

        self.bounding_box = BoundingBox()
        # update bounding box in case this entity does no stepping
        self._update_bounding_box()

    def step(self):
        flying_or_floating = self.flying or self.floating
        is_in_air = not self.is_supported and not flying_or_floating
        if is_in_air:
            friction_factor = 1
            move_control_factor = self.air_control_factor
        else:
            friction_factor = self.vx_decay
            move_control_factor = 1
        self.vx = self.vx * friction_factor + self.ax * move_control_factor
        if abs(self.vx) < 0.05:
            self.vx = 0

        # clip between [-max_movement, +max_movement]
        if self.vx > 0:
            self.vx = min(self.vx, self.max_movement)
        else:
            self.vx = max(self.vx, -self.max_movement)

        if flying_or_floating:
            self.vy = self.ay
        else:
            # jumping is only supported if we have support TODO this might be changed
            if self.ay > 0 and self.is_supported:
                self.vy = self.ay

        self.calculate_physics()

        if self.checks_collision:
            self._check_collisions()

    def calculate_physics(self):
        def get_blocks_left():
            top_y = self.y + self.size[1]
            bottom_y = self.y + self.size[1] - 1e-5 # epsilon to make sure the block we are on is always covered
            #top_y = self.y + self.vy + self.size[1]
            #bottom_y = self.y + self.vy - 1e-5 # epsilon to make sure the block we are on is always covered
            block_top = self.level.blocks[int(top_y)][int(self.x)-1]
            block_bottom = self.level.blocks[int(bottom_y)][int(self.x)-1]
            return block_top, block_bottom

        def get_blocks_right():
            top_y = self.y + self.size[1]
            bottom_y = self.y + self.size[1] - 1e-5 # epsilon to make sure the block we are on is always covered
            #top_y = self.y + self.vy + self.size[1]
            #bottom_y = self.y + self.vy - 1e-5  # epsilon to make sure the block we are on is always covered
            block_top = self.level.blocks[int(top_y)][int(self.x)+1]
            block_bottom = self.level.blocks[int(bottom_y)][int(self.x)+1]
            return block_top, block_bottom

        self.collision_x = False
        if self.vx != 0:
            if self.vx < 0:
                left_x = self.x - self.halfWidth
                if (left_x + self.vx) % 1.0 > 1 + self.vx:
                    # passing block boundary
                    block_top, block_bottom = get_blocks_left()
                    if not block_top.is_passable or not block_bottom.is_passable:
                        # collision
                        self.collision_x = True
                        self.x = int(self.x) + self.halfWidth + 1e-5
                        self.vx = 0
            else:
                right_x = self.x + self.halfWidth
                if (right_x + self.vx) % 1.0 < self.vx:
                    # passing block boundary
                    block_top, block_bottom = get_blocks_right()
                    if not block_top.is_passable or not block_bottom.is_passable:
                        # collision
                        self.collision_x = True
                        self.x = int(self.x) + 1 - self.halfWidth - 1e-5
                        self.vx = 0
            self.x += self.vx

        def get_blocks_below():
            left_x = self.x - self.halfWidth
            right_x = self.x + self.halfWidth - 1e-5 # epsilon to make sure the block we are on is always covered
            block_left = self.level.blocks[int(self.y)-1][int(left_x)]
            block_right = self.level.blocks[int(self.y)-1][int(right_x)]
            return block_left, block_right

        def get_blocks_above():
            left_x = self.x - self.halfWidth
            right_x = self.x + self.halfWidth - 1e-5 # epsilon to make sure the block we are on is always covered
            block_left = self.level.blocks[int(self.y+self.size[1])+1][int(left_x)]
            block_right = self.level.blocks[int(self.y+self.size[1])+1][int(right_x)]
            return block_left, block_right

        ry = self.y % 1.0 #distance to upper boundary of lower block

        self.collision_y = False
        # check if standing or falling
        if self.vy <= 0:
            # check if we are near the lower block boundary
            if ry < 1e-3:
                # look at blocks below us
                block_left, block_right = get_blocks_below()
                self.is_supported = block_left.has_support or block_right.has_support
                self.solid_ground = not block_left.is_passable or not block_right.is_passable
                if self.is_supported:
                    self.collision_y = True
                    self.vy = 0
        else:
            # we never have support when jumping
            self.is_supported = False
            self.solid_ground = False

        #check if we are floating in the current block
        current_block = self.level.blocks[int(self.y)][int(self.x)]
        self.floating = current_block.is_floatable

        # apply gravity if we are not floating flying or using support
        if not (self.floating or self.flying) and not (self.is_supported and self.use_support)\
                and not self.solid_ground:
            self.vy = max(self.vy - self.level.gravity, -self.max_fall_speed)

        if self.vy != 0:
            if self.vy < 0:
                # falling down
                if ry < -self.vy:
                    # we will move across the block boundary, look at blocks below
                    block_left, block_right = get_blocks_below()
                    will_be_supported = (not block_left.is_passable or not block_right.is_passable) \
                                        or ((block_left.has_support or
                                            block_left.has_support) and self.use_support)
                    if will_be_supported:
                        self.collision_y = True
                        self.is_supported = True # allows jumping on the next frame
                        self.vy = -ry + 1e-4
            else:
                # jumping up
                ry = 1 - ry
                if ry < self.vy:
                    # we will move across the block boundary, look at blocks above
                    block_left, block_right = get_blocks_above()
                    if not block_left.is_passable or not block_right.is_passable:
                        #hit ceiling
                        self.collision_y = True
                        self.vy = max(0, ry - 1e-4)
            self.y += self.vy

        self.ax = 0
        self.ay = 0

        self._update_bounding_box()

    def _check_collisions(self):
        pass

    def _update_bounding_box(self):
        left = self.x - self.halfWidth
        self.bounding_box.set(left, self.y, self.size[0], self.size[1])

    def _get_parameterization(self):
        return [0, 0, 0, 0]

    def get_representation(self):
        parameterization = self._get_parameterization()
        return [self._entity_id, self.x, self.y, self.vx, self.vy, *parameterization]

    def render(self, camera, frame):
        camera.paint_rect(self.x-self.halfWidth, self.y, self.size)
        #camera.paint_rect(self.x-0.5*camera.zoom, self.y, self.size)
