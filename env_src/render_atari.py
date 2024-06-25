import numpy as np
# from ocatari import OCAtari
from hackatari import HackAtari

def render_atari(agent, args):
    # gamename =
    rdr_mode = "human" if args.render else "rgb_array"
    env = HackAtari(env_name=args.env.capitalize(), render_mode=rdr_mode, mode="revised")
    # env = OCAtari(env_name=args.env.capitalize(), render_mode=rdr_mode, mode="revised")
    obs = env.reset()
    try:
        agent.nb_actions = env.nb_actions
    except:
        pass
    scores = []
    nb_epi = 20
    for epi in range(nb_epi):
        total_r = 0
        step = 0
        print(f"Episode {epi}")
        print(f"==========")
        while True:
            # action = random.randint(0, env.nb_actions-1)
            if args.alg == 'logic':
                action, explaining = agent.act(env.objects)
                print(action, explaining)
            elif args.alg == 'random':
                action = np.random.randint(env.nb_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            step += 1
            # if step % 10 == 0:
            #     import matplotlib.pyplot as plt
            #     plt.imshow(env._get_obs())
            #     plt.show()
            if terminated:
                print("episode: ", epi)
                print("return: ", total_r)
                scores.append(total_r)
                env.reset()
                step = 0
                break
    print(np.average(scores))
