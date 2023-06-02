from PIL import Image
from PIL.ImageDraw import ImageDraw


class Camera:

    def __init__(self, width, height, x=0, y=0, zoom=16):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.screen = Image.new('RGBA', (width, height))
        self.draw = ImageDraw(self.screen)
        self.zoom = zoom
        self._bgcolor = (160, 160, 205)

    def start_render(self):
        self.screen.paste(self._bgcolor, (0, 0, self.screen.size[0], self.screen.size[1]))

    def end_render(self):
        pass

    def paint_rect(self, x, y, size, color=(255, 0, 0)):
        x = x * self.zoom - self.x
        y = y * self.zoom - self.y

        y = self.height - y
        self.draw.rectangle((x, y, x+size[0]*self.zoom, y+size[1]*self.zoom), fill=color)

    def paint_sprite(self, x, y, size, sprite):
        x = x * self.zoom - self.x
        y = y * self.zoom - self.y

        y = self.height - y
        self.screen.paste(sprite, (int(x), int(y)), mask=sprite)

    def paint_sprite_absolute(self, x, y, sprite):
        #y = self.height - y  # make y start at the bottom
        self.screen.paste(sprite, (int(x), int(y)), mask=sprite)
