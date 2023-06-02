class HUD:

    def __init__(self, getout, resource_loader):
        self.getout = getout
        self.resource_loader = resource_loader
        self.level = getout.level

        self.hud_sprite_size = 32
        self.sprites = self._load_sprites(resource_loader) if resource_loader is not None else None

    def _load_sprites(self, resource_loader):
        sprites = {
            'coin': resource_loader.get_sprite('hud_coin', 'HUD/hudCoin.png', sprite_size=self.hud_sprite_size),
            'powerup': resource_loader.get_sprite('hud_jewel', 'HUD/hudJewel_red.png',
                                                  sprite_size=self.hud_sprite_size),
            'enemy': resource_loader.get_sprite('hud_enemy', 'Enemies/slimeBlue.png', sprite_size=self.hud_sprite_size),
            'key': resource_loader.get_sprite('hud_key', 'HUD/hudKey_blue.png', sprite_size=self.hud_sprite_size),
        }

        sprites['-'] = resource_loader.get_sprite(f'hud_-', f'HUD/hudMinus.png', sprite_size=self.hud_sprite_size)
        for i in range(10):
            sprites[str(i)] = resource_loader.get_sprite(f'hud_{i}', f'HUD/hud{i}.png',
                                                         sprite_size=self.hud_sprite_size)

        return sprites

    def render(self, camera, step=None):
        sz = self.hud_sprite_size
        sz2 = self.hud_sprite_size // 2

        x = 10
        camera.paint_sprite_absolute(x, 10, self.sprites['coin'])
        x += sz2 + 5
        x += self.paint_int(self.getout.level.reward_values['coin'], camera, x, 10, sz2)
        x += 2 * sz

        camera.paint_sprite_absolute(x, 10, self.sprites['powerup'])
        x += sz2 + 5
        x += self.paint_int(self.getout.level.reward_values['powerup'], camera, x, 10, sz2)
        x += 2 * sz

        camera.paint_sprite_absolute(x, 10, self.sprites['enemy'])
        x += sz2 + 5
        x += self.paint_int(self.getout.level.reward_values['enemy'], camera, x, 10, sz2)
        x += 2 * sz

        camera.paint_sprite_absolute(x, 10, self.sprites['key'])
        x += sz2 + 5
        x += self.paint_int(self.getout.level.reward_values['key'], camera, x, 10, sz2)
        x += 2 * sz

        self.paint_int(self.getout.score, camera, camera.width - 4 * sz2, 10, sz2)

        if step is not None:
            self.paint_int(step, camera, camera.width / 2 - 2 * sz2, 10, sz2)

    def paint_int(self, n, camera, x, y, sz2):
        ox = x

        if n < 0:
            camera.paint_sprite_absolute(x, y, self.sprites['-'])
            x += sz2
            n = abs(n)
        chr = n // 100
        if chr != 0:
            camera.paint_sprite_absolute(x, y, self.sprites[self.int_to_chr(chr)])
            x += sz2
        n -= chr * 100

        chr = n // 10
        camera.paint_sprite_absolute(x, 10, self.sprites[self.int_to_chr(chr)])
        x += sz2

        n -= chr * 10
        chr = n // 1
        camera.paint_sprite_absolute(x, 10, self.sprites[self.int_to_chr(chr)])
        return x - ox

    def int_to_chr(self, x):
        if x == 0:
            return '0'
        elif x == 1:
            return '1'
        elif x == 2:
            return '2'
        elif x == 3:
            return '3'
        elif x == 4:
            return '4'
        elif x == 5:
            return '5'
        elif x == 6:
            return '6'
        elif x == 7:
            return '7'
        elif x == 8:
            return '8'
        elif x == 9:
            return '9'
