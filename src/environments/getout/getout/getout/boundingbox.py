
class BoundingBox:

    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def set(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def intersects(self, other):
        l2 = other.x
        r1 = self.x + self.w
        l1 = self.x
        r2 = other.x + other.w
        t2 = other.y + other.h
        b2 = other.y
        b1 = self.y
        t1 = self.y + self.h
        return not (l2 > r1 or r2 < l1 or t2 < b1 or b2 > t1)
