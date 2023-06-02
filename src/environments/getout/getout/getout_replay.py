import random
import time
from argparse import ArgumentParser
import pathlib
import pickle

import numpy as np
from imageviewer import ImageViewer

from src.getout.getout.getout import ParameterizedLevelGenerator
from src.getout.getout.getout import Getout

KEY_SPACE = 32
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


def create_getout_instance(seed=None):
    seed = random.randint(0, 10000) if seed is None else seed

    coin_jump = Getout(start_on_first_action=True)
    coin_jump.show_step_counter = True
    #level_generator = DummyGenerator()
    level_generator = ParameterizedLevelGenerator()
    level_generator.generate(coin_jump, seed=seed)
    coin_jump.render()

    return coin_jump


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-r", "--replay_file", dest="replay_file")
    args = parser.parse_args()

    replay_file = pathlib.Path(args.replay_file)

    return args, replay_file

def run():
    args, replay_file = parse_args()

    data = load_recording(replay_file)
    actions = data["actions"]
    actions_count = len(actions)

    coin_jump = create_getout_instance(data['meta']['level']['seed'])
    viewer = setup_image_viewer(coin_jump)

    # frame rate limiting
    fps = 10
    target_frame_duration = 1 / fps
    last_frame_time = 0

    i = 0
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
        action = actions[i] if i != -1 else []
        reward = coin_jump.step(action)

        np_img = np.asarray(coin_jump.camera.screen)
        viewer.show(np_img[:, :, :3])

        if i != -1:
            i += 1
            if i == actions_count:
                i = -1

        #terminated = coin_jump.level.terminated
        #if terminated:
        #    break
        if viewer.is_escape_pressed:
            break

    print("Maze terminated")


def load_recording(replay_file):
    with open(replay_file, 'rb') as f:
        #data = {
        #    'actions': actions, =[ACTION,, ...]
        #    'meta': getout1.level.get_representation(),
        #    'score': getout1.score
        #}
        data = pickle.load(f)
        print("loading", data)
        return data


if __name__ == "__main__":
    run()

