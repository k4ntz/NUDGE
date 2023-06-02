import time
from argparse import ArgumentParser
import pathlib
import pickle

import numpy as np

from src.getout.getout.getout import create_getout_instance
from imageviewer import ImageViewer

from src.getout.getout.getout import GetoutActions

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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-s", "--save_dir", dest="save_dir", default=None)
    args = parser.parse_args()

    if args.save_dir is None:
        save_dir = pathlib.Path(__file__).parent.resolve().joinpath('recordings')
    else:
        save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    return args, save_dir

def run():
    args, save_dir = parse_args()

    coin_jump = create_getout_instance(start_on_first_action=True)
    viewer = setup_image_viewer(coin_jump)

    # frame rate limiting
    fps = 10
    target_frame_duration = 1 / fps
    last_frame_time = 0

    actions = [None] * 10000 # reserve some space for recording actions
    actions_count = 0
    is_recording = False

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
            coin_jump = create_getout_instance(start_on_first_action=True)
            actions_count = 0
            is_recording = False
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

        if not is_recording and coin_jump.has_started and not coin_jump.level.terminated:
            is_recording = True

        if is_recording:
            actions[actions_count] = action
            actions_count += 1

            if coin_jump.level.terminated:
                save_recording(coin_jump, actions[:actions_count], save_dir)
                actions_count = 0
                is_recording = False



        np_img = np.asarray(coin_jump.camera.screen)
        viewer.show(np_img[:, :, :3])

        #terminated = coin_jump.level.terminated
        #if terminated:
        #    break
        if viewer.is_escape_pressed:
            break

    print("Maze terminated")


def save_recording(getout, actions, save_dir):
    timestamp_sec = int(time.time() * 1000.0)
    i = 0
    while save_dir.joinpath(f'{timestamp_sec}_{i}').exists():
        i += 1
    file_path = save_dir.joinpath(f'{timestamp_sec}_{i}')
    with open(file_path, 'wb') as f:
        data = {
            'actions': actions,
            'meta': getout.level.get_representation(),
            'score': getout.score
        }
        print(f"saving to {file_path}")
        print(data)
        pickle.dump(data, f, protocol=4)


if __name__ == "__main__":
    run()

