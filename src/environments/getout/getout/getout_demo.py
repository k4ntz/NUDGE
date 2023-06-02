import random
import time

import numpy as np
from imageviewer import ImageViewer

from src.environments.getout.getout.getout.paramLevelGenerator import ParameterizedLevelGenerator
from src.environments.getout.getout.getout.getout import Getout

from src.environments.getout.getout.getout.actions import GetoutActions

KEY_SPACE = 32
#KEY_SPACE = 32
KEY_w = 119
KEY_a = 97
KEY_s = 115
KEY_d = 100
KEY_r = 114

def setup_image_viewer(getout):
    viewer = ImageViewer(
        "getout1",
        getout.camera.height,
        getout.camera.width,
        monitor_keyboard=True,
        #relevant_keys=set('W','A','S','D','SPACE')
    )
    return viewer

def create_getout_instance():
    seed = random.random()

    coin_jump = Getout(start_on_first_action=True)
    #level_generator = DummyGenerator()
    level_generator = ParameterizedLevelGenerator()
    level_generator.generate(coin_jump, seed=seed)
    coin_jump.render()

    return coin_jump


def run():
    coin_jump = create_getout_instance()
    viewer = setup_image_viewer(coin_jump)

    # frame rate limiting
    fps = 10
    target_frame_duration = 1 / fps
    last_frame_time = 0

    while True:
        # control framerate
        current_frame_time = time.time()
        # limit frame rate
        if last_frame_time + target_frame_duration > current_frame_time:
            sl = (last_frame_time + target_frame_duration) - current_frame_time
            time.sleep(sl)
            continue
        last_frame_time = current_frame_time # save frame start time for next iteration

        # step game
        action = []
        if KEY_r in viewer.pressed_keys:
            coin_jump = create_getout_instance()
        else:
            if KEY_a in viewer.pressed_keys:
                action.append(GetoutActions.MOVE_LEFT)
            if KEY_d in viewer.pressed_keys:
                action.append(GetoutActions.MOVE_RIGHT)
            if (KEY_SPACE in viewer.pressed_keys) or (KEY_w in viewer.pressed_keys):
                action.append(GetoutActions.MOVE_UP)
            if KEY_s in viewer.pressed_keys:
                action.append(GetoutActions.MOVE_DOWN)

        reward = coin_jump.step(action)
        score = coin_jump.get_score()

        np_img = np.asarray(coin_jump.camera.screen)
        viewer.show(np_img[:, :, :3])

        terminated = coin_jump.level.terminated
        #if terminated:
        #    break
        if viewer.is_escape_pressed:
            break

    print("Maze terminated")


if __name__ == "__main__":
    run()

