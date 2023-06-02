import random
from argparse import ArgumentParser
import pathlib
import pickle

from src.getout.getout.getout import CJA_NOOP
from imageviewer import ImageViewer

from src.getout.getout.getout import ParameterizedLevelGenerator
from src.getout.getout.getout import Getout

import os


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
    parser.add_argument("-r", "--recording_dir", dest="recording_dir", default="recordings")
    args = parser.parse_args()

    return args

def record_run(recording):
    actions = recording["actions"]
    actions_count = len(actions)

    coin_jump = create_getout_instance(recording['meta']['level']['seed'])
    frames = []

    for step in range(actions_count+1):
        frames.append(coin_jump.camera.screen.copy().convert('RGB'))

        # step game
        action = actions[step] if step != actions_count else CJA_NOOP
        reward = coin_jump.step(action)

    return frames

def run():
    args = parse_args()
    recording_folder = pathlib.Path(args.recording_dir)

    output_folder = recording_folder.parent / (recording_folder.name + "_render")
    output_folder.mkdir(exist_ok=True)

    temp_folder = output_folder / "temp"
    temp_folder.mkdir(exist_ok=True)

    for replay_file in recording_folder.iterdir():
        # clear temp
        for f in temp_folder.glob("*"):
            f.unlink()

        # data = {
        #    'actions': actions, =[ACTION,, ...]
        #    'meta': getout1.level.get_representation(),
        #    'score': getout1.score
        # }
        with open(replay_file, 'rb') as f:
            data = pickle.load(f)
        frames = record_run(data)

        for i, frame in enumerate(frames):
            frame.save(temp_folder / f"{i:04d}.png")

        output_filename = output_folder / (replay_file.name + ".mp4")

        #os.system(f'ffmpeg -r 10 -i {temp_folder}/%04d.png -pix_fmt yuv420p -r 10 -y {output_filename}')
        os.system(f'ffmpeg -r 10 -i {temp_folder}/%04d.png -vcodec libx264 -acodec aac -r 10 -y {output_filename}')


if __name__ == "__main__":
    run()

