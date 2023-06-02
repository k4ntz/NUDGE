from .camera import Camera


class TrackingCamera(Camera):

    def __init__(self, width, height, tracked_object, **kwargs):
        Camera.__init__(self, width, height, **kwargs)
        self.do_track = tracked_object is not None
        self.tracked_object = tracked_object

    def start_render(self):
        super().start_render()
        if self.do_track:
            self.x = self.tracked_object.x * self.zoom - (self.width / 2)
            self.y = self.tracked_object.y * self.zoom - (self.height / 2)
