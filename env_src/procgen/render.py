import ast
import csv
import io
import os
import sys
import time

import cv2
import gym3
import pandas as pd
from PIL import ImageFont, Image, ImageDraw

from env_src.procgen.procgen import ProcgenGym3Env

font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
font = ImageFont.truetype(font_path, size=40)


def run(env, nb_games=20):
    """
    Display a window to the user and loop until the window is closed
    by the user.
    """
    prev_time = env._renderer.get_time()
    env._renderer.start()
    env._draw()
    env._renderer.finish()

    old_stdout = sys.stdout  # Memorize the default stdout stream
    sys.stdout = buffer = io.StringIO()

    # whatWasPrinted = buffer.getvalue()  # Return a str containing the entire contents of the buffer.
    while buffer.getvalue().count("final info") < nb_games:
        now = env._renderer.get_time()
        dt = now - prev_time
        prev_time = now
        if dt < env._sec_per_timestep:
            sleep_time = env._sec_per_timestep - dt
            time.sleep(sleep_time)

        keys_clicked, keys_pressed = env._renderer.start()
        if "O" in keys_clicked:
            env._overlay_enabled = not env._overlay_enabled
        env._update(dt, keys_clicked, keys_pressed)
        env._draw()
        env._renderer.finish()
        if not env._renderer.is_open:
            break
    sys.stdout = old_stdout  # Put the old stream back in place
    all_summaries = [line for line in buffer.getvalue().split("\n") if line.startswith("final")]
    return all_summaries


def get_values(summaries, key_str, stype=float):
    all_values = []
    for line in summaries:
        dico = ast.literal_eval(line[11:])
        all_values.append(stype(dico[key_str]))
    return all_values


def render_loot(agent, args):
    envname = args.env
    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array", rand_seed=args.seed, start_level=args.seed)

    if args.log:
        log_f = open(args.logfile, "w+")
        writer = csv.writer(log_f)
        if args.alg == 'logic':
            head = ['episode', 'step', 'reward', 'average_reward', 'logic_state', 'probs']
            writer.writerow(head)
        elif args.alg == 'ppo' or args.alg == 'random':
            head = ['episode', 'step', 'reward', 'average_reward']
            writer.writerow(head)

    if agent == "human":
        ia = gym3.Interactive(env, info_key="rgb", height=768*2, width=768*2)
        all_summaries = run(ia, 20)

        scores = get_values(all_summaries, "episode_return")
        df = pd.DataFrame({'reward': scores})
        df.to_csv(args.logfile, index=False)
    else:
        if args.render:
            env = gym3.ViewerWrapper(env, info_key="rgb")
        reward, obs, done = env.observe()
        scores = []
        last_explaining = None
        nsteps = 0
        for epi in range(20):
            print(f"Episode {epi}")
            print(f"==========")
            total_r = 0
            step = 0
            while True:
                step += 1
                nsteps += 1
                if args.alg == 'logic':
                    # print(obs['positions'])
                    action, explaining = agent.act(obs)
                else:
                    action = agent.act(obs)
                env.act(action)
                rew, obs, done = env.observe()
                total_r += rew[0]
                if args.alg == 'logic':
                    if last_explaining is None or (explaining != last_explaining and repeated > 2):
                        # print(explaining)
                        last_explaining = explaining
                        disp_text = explaining
                        repeated = 0
                    repeated += 1

                if args.record:
                    screen = Image.fromarray(env._get_image())
                    ImageDraw.Draw(screen).text((40, 60), disp_text, (20, 170, 20), font=font)
                    screen.save(f"renderings/{nsteps:03}.png")

                # if args.log:
                #     if args.alg == 'logic':
                #         probs = agent.get_probs()
                #         logic_state = agent.get_state(obs)
                #         data = [(epi, step, rew[0], average_r, logic_state, probs)]
                #         writer.writerows(data)
                #     else:
                #         data = [(epi, step, rew[0], average_r)]
                #         writer.writerows(data)

                if done:
                    # step = 0
                    print("episode: ", epi)
                    print("return: ", total_r)
                    scores.append(total_r)
                    break

                if epi > 100:
                    break

            df = pd.DataFrame({'reward': scores})
            df.to_csv(f"logs/{envname}/{args.alg}_{envname}_log_{args.seed}.csv", index=False)


def render_ecoinrun(agent, args):
    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array", seed=args.seed)
    if agent == "human":
        ia = gym3.Interactive(env, info_key="rgb", height=768, width=768)
        ia.run()
    else:
        env = gym3.ViewerWrapper(env, info_key="rgb")
        reward, obs, done = env.observe()
        i = 0
        while True:
            action = agent.act(obs)
            env.act(action)
            rew, obs, done = env.observe()
            # if i % 40 == 0:
            #     print("\n" * 50)
            #     print(obs["positions"])
            i += 1


def render_threefish(agent, args):
    envname = args.env
    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array", rand_seed=args.seed, start_level=args.seed)

    if args.log:
        log_f = open(args.logfile, "w+")
        writer = csv.writer(log_f)

        if args.alg == 'logic':
            head = ['episode', 'step', 'reward', 'average_reward', 'logic_state', 'probs']
            writer.writerow(head)
        elif args.alg == 'ppo' or args.alg == 'random':
            head = ['episode', 'step', 'reward', 'average_reward']
            writer.writerow(head)

    if agent == "human":
        ia = gym3.Interactive(env, info_key="rgb", height=768, width=768)
        all_summaries = run(ia, 10)

        df_scores = get_values(all_summaries, "episode_return")
        data = {'reward': df_scores}
        # convert list to df_scores
        # pd.to_csv(df_scores, f"{player_name}_scores.csv")
        df = pd.DataFrame(data)
        df.to_csv(args.logfile, index=False)

    else:
        if args.render:
            env = gym3.ViewerWrapper(env, info_key="rgb", height=600, width=900)
        reward, obs, done = env.observe()
        scores = []
        last_explaining = None
        for epi in range(20):
            print(f"Episode {epi}")
            print(f"==========")
            total_r = 0
            step = 0
            while True:
                step += 1
                if args.alg == 'logic':
                    action, explaining = agent.act(obs)
                else:
                    action = agent.act(obs)
                env.act(action)
                rew, obs, done = env.observe()
                total_r += rew[0]
                if args.alg == 'logic':
                    if last_explaining is None or (explaining != last_explaining and repeated > 4):
                        print(explaining)
                        last_explaining = explaining
                        disp_text = explaining
                        repeated = 0
                    repeated += 1
                if done:
                    step = 0
                    print("episode: ", epi)
                    print("return: ", total_r)
                    scores.append(total_r)
                    if args.record:
                        exit()
                    break
                if epi > 100:
                    break
                if args.record:
                    screen = Image.fromarray(env._get_image())
                    ImageDraw.Draw(screen).text((40, 60), disp_text, (120, 20, 20), font=font)
                    screen.save(f"renderings/{step:03}.png")


            df = pd.DataFrame({'reward': scores})
            df.to_csv(f"logs/{envname}/{args.alg}_{envname}_log_{args.seed}.csv", index=False)
