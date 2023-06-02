# a logic player example of getout
import time
import argparse
import numpy as np
import sys

sys.path.insert(0, '../')
from src.utils import make_deterministic
from src.environments.getout.getout.imageviewer import ImageViewer
from src.environments.getout.getout.getout.helpers import create_getout_instance
from nsfr.utils import get_nsfr_model, get_predictions
from src.agents.utils_getout import extract_logic_state_getout

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
        # relevant_keys=set('W','A','S','D','SPACE')
    )
    return viewer


def explaining_to_action(explaining):
    if 'jump' in explaining:
        return 3
    elif 'left' in explaining:
        return 1
    elif 'right' in explaining:
        return 2
    elif 'stay' in explaining:
        return 0


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", help="Seed for pytorch + env", default=0,
                        required=False, action="store", dest="seed", type=int)
    parser.add_argument("-m", "--mode", help="the game mode you want to play with",
                        required=False, action="store", dest="m", default='getout',
                        choices=['getout'])
    parser.add_argument("-e", "--env", help="The environment of getout", default="Getout", dest="env",
                        choices=["Getout", "GetoutPlus", "GetoutE", "GetoutKD"])

    parser.add_argument("-r", "--rules", dest="rules", default='getout_human_assisted',
                        required=False,
                        choices=['getout_human_assisted', 'getout_kd', 'getout_e', 'getoutplus'])
    args = parser.parse_args()
    make_deterministic(args.seed)

    if args.env == "Getout":
        coin_jump = create_getout_instance()
    elif args.env == "GetoutKD":
        coin_jump = create_getout_instance(key_door=True)
    elif args.env == "GetoutE":
        coin_jump = create_getout_instance(enemy=True)
    else:
        coin_jump = create_getout_instance(enemies=True)
    viewer = setup_image_viewer(coin_jump)

    # frame rate limiting
    fps = 10
    target_frame_duration = 1 / fps
    last_frame_time = 0
    last_explaining = None

    nsfr = get_nsfr_model(args, train=False)
    while True:
        # control framerate
        current_frame_time = time.time()
        # limit frame rate
        if last_frame_time + target_frame_duration > current_frame_time:
            sl = (last_frame_time + target_frame_duration) - current_frame_time
            time.sleep(sl)
            continue
        last_frame_time = current_frame_time  # save frame start time for next iteration

        # step game
        action = []

        # if KEY_r in viewer.pressed_keys:
        #     if args.env == "getout":
        #         coin_jump = create_getout_instance()
        #     elif args.env == "getout_kd":
        #         coin_jump = create_getout_instance(key_door=True)
        #     elif args.env == "getout_e":
        #         coin_jump = create_getout_instance(enemy=True)

        if not coin_jump.level.terminated:

            # extract state for expextracted_statelaining
            extracted_state = extract_logic_state_getout(coin_jump, args)
            explaining = get_predictions(extracted_state, nsfr)
            action = explaining_to_action(explaining)

            if last_explaining is None:
                print(explaining)
                last_explaining = explaining
            elif explaining != last_explaining:
                print(explaining)
                last_explaining = explaining
        else:
            if args.env == "Getout":
                coin_jump = create_getout_instance()
            elif args.env == "GetoutKD":
                coin_jump = create_getout_instance(key_door=True)
            elif args.env == "GetoutE":
                coin_jump = create_getout_instance(enemy=True)
            else:
                coin_jump = create_getout_instance(enemies=True)
            action = 0
            print("--------------------------     next game    --------------------------")

        reward = coin_jump.step(action)
        score = coin_jump.get_score()
        np_img = np.asarray(coin_jump.camera.screen)
        viewer.show(np_img[:, :, :3])

        terminated = coin_jump.level.terminated
        # if terminated:
        #     print("score = ", score)
        if viewer.is_escape_pressed:
            break

    print("Maze terminated")


if __name__ == "__main__":
    run()
