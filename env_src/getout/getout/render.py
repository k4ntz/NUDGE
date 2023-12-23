import csv
import time
import os
import cv2

import numpy as np
import pandas as pd
from PIL import ImageDraw, ImageFont

from .actions import GetoutActions
from .getout import Getout
from .paramLevelGenerator import ParameterizedLevelGenerator
from ..imageviewer import ImageViewer

font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
font = ImageFont.truetype(font_path, size=40)


def render_getout(agent, args):
    envname = args.env
    # envname = "getout"
    KEY_SPACE = 32
    # KEY_SPACE = 32
    KEY_w = 119
    KEY_a = 97
    KEY_s = 115
    KEY_d = 100
    KEY_r = 114
    KEY_LEFT = 65361
    KEY_RIGHT = 65363
    KEY_UP = 65362

    def setup_image_viewer(getout):
        print(getout.camera.height, getout.camera.width)
        viewer = ImageViewer(
            "getout",
            getout.camera.height,
            getout.camera.width,
            monitor_keyboard=True,
        )
        return viewer

    def create_getout_instance(args, seed=None):
        if args.env == 'getoutplus':
            enemies = True
        else:
            enemies = False
        # level_generator = DummyGenerator()
        getout = Getout()
        level_generator = ParameterizedLevelGenerator(enemies=enemies)
        level_generator.generate(getout, seed=seed)
        getout.render()

        return getout

    # seed = random.randint(0, 100000000)
    # print(seed)
    getout = create_getout_instance(args)
    viewer = setup_image_viewer(getout)

    # frame rate limiting
    fps = 10
    target_frame_duration = 1 / fps
    last_frame_time = 0

    num_epi = 1
    max_epi = 20
    total_reward = 0
    epi_reward = 0
    current_reward = 0
    step = 0
    last_explaining = None
    scores = []
    if args.log:
        log_f = open(args.logfile, "w+")
        writer = csv.writer(log_f)

        if args.alg == 'logic':
            head = ['episode', 'step', 'reward', 'average_reward', 'logic_state', 'probs']
        elif args.alg == 'ppo' or args.alg == 'random':
            head = ['episode', 'step', 'reward', 'average_reward']
        elif args.alg == 'human':
            head = ['episode', 'reward']
        writer.writerow(head)

    while num_epi <= max_epi:
        # control framerate
        current_frame_time = time.time()
        # limit frame rate
        if last_frame_time + target_frame_duration > current_frame_time:
            sl = (last_frame_time + target_frame_duration) - current_frame_time
            time.sleep(sl)
            continue
        last_frame_time = current_frame_time  # save frame start time for next iteration
        # step game
        step += 1
        action = []
        if not getout.level.terminated:
            if args.alg == 'logic':
                action, explaining = agent.act(getout)
            elif args.alg == 'ppo':
                action = agent.act(getout)
            elif args.alg == 'human':
                if KEY_a in viewer.pressed_keys or KEY_LEFT in viewer.pressed_keys:
                    action.append(GetoutActions.MOVE_LEFT)
                if KEY_d in viewer.pressed_keys or KEY_RIGHT in viewer.pressed_keys:
                    action.append(GetoutActions.MOVE_RIGHT)
                if (KEY_SPACE in viewer.pressed_keys) or (
                        KEY_w in viewer.pressed_keys) or KEY_UP in viewer.pressed_keys:
                    action.append(GetoutActions.MOVE_UP)
                if KEY_s in viewer.pressed_keys:
                    action.append(GetoutActions.MOVE_DOWN)
            elif args.alg == 'random':
                action = agent.act(getout)
        else:
            getout = create_getout_instance(args)
            # print("epi_reward: ", round(epi_reward, 2))
            # print("--------------------------     next game    --------------------------")
            print(f"Episode {num_epi}")
            print(f"==========")
            if args.alg == 'human':
                data = [(num_epi, round(epi_reward, 2))]
                # writer.writerows(data)
            total_reward += epi_reward
            epi_reward = 0
            action = 0
            # average_reward = round(total_reward / num_epi, 2)
            num_epi += 1
            step = 0

        reward = getout.step(action)
        score = getout.get_score()
        current_reward += reward
        average_reward = round(current_reward / num_epi, 2)
        disp_text = ""
        if args.alg == 'logic':
            if last_explaining is None or (explaining != last_explaining and repeated > 4):
                print(explaining)
                last_explaining = explaining
                disp_text = explaining
                repeated = 0
            repeated += 1

        if args.log:
            if args.alg == 'logic':
                probs = agent.get_probs()
                logic_state = agent.convert_state(getout)
                data = [(num_epi, step, reward, average_reward, logic_state, probs)]
                writer.writerows(data)
            elif args.alg == 'ppo' or args.alg == 'random':
                data = [(num_epi, step, reward, average_reward)]
                writer.writerows(data)
        # print(reward)
        epi_reward += reward

        if args.render:
            screen = getout.camera.screen
            ImageDraw.Draw(screen).text((40, 60), disp_text, (120, 20, 20), font=font)
            np_img = np.asarray(screen)
            viewer.show(np_img[:, :, :3])
            if args.record:
                screen.save(f"renderings/{step:03}.png")

        # terminated = getout.level.terminated
        # if terminated:
        #    break
        if viewer.is_escape_pressed:
            break

        if getout.level.terminated:
            if args.record:
                exit()
            step = 0
            print(num_epi)
            print('reward: ' + str(round(score, 2)))
            scores.append(epi_reward)
        if num_epi > 100:
            break

    df = pd.DataFrame({'reward': scores})
    # df.to_csv(f"logs/{envname}/random_{envname}_log_{args.seed}.csv", index=False)
    df.to_csv(f"logs/{envname}/{args.alg}_{envname}_log_{args.seed}.csv", index=False)
    print(f"saved in logs/{envname}/{args.alg}_{envname}_log_{args.seed}.csv")
