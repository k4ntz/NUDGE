import pathlib

from PIL import Image, ImageOps


class ResourceLoader:

    def __init__(self, path, sprite_size=16, no_loading=False):
        if path is str:
            path = pathlib.Path(path)
        self._path = path
        self._sprite_size = sprite_size
        self.no_loading = no_loading
        self._sprites = {}

    def get_sprite(self, name, rel_path, bbox=None, transform=None, sprite_size=None, pad=None):
        if sprite_size is None:
            sprite_size = self._sprite_size

        sprite = self._sprites.get(name, None)
        if sprite is None:
            sprite = self.load_image(rel_path)
            if bbox is not None:
                sprite = sprite.crop(bbox)
            if pad is not None:
                sprite = self._pad(sprite, pad)
            if sprite_size is not None:
                sprite = sprite.resize((sprite_size, sprite_size))
            if transform is not None:
                sprite = transform(sprite)
            self._sprites[name] = sprite
        return sprite

    def _pad(self, im, pad_w, pad_h=None, color=None):
        if pad_h is None:
            pad_h = pad_w
        if color is None:
            color = tuple([0] * len(im.getbands()))
        w, h = im.size
        w += 2*pad_w
        h += 2*pad_h
        im_pad = Image.new(im.mode, (w, h), color)
        im_pad.paste(im, (pad_w, pad_h))
        return im_pad

    def set_sprite(self, name, image):
        self._sprites[name] = image

    def load_image(self, rel_path):
        if self.no_loading:
            return None

        with Image.open(self._path.joinpath(rel_path)) as img:
            img.load()
        return img


def flip_horizontal(sprite):
    return ImageOps.mirror(sprite)


def rotate(deg):
    return lambda im: im.rotate(deg)
