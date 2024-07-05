from nudge.renderer import Renderer
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("-g", "--game", type=str, default="seaquest")
parser.add_argument("-a", "--agent_path", type=str, default="out/runs/game/logic/")
parser.add_argument("-np", "--no_predicates", action="store_true")

def find_latest_trained_checkpoint(agent_path):
    agent_path = Path(agent_path)
    for with_datestamp in sorted(agent_path.glob("*"), reverse=True):
        if len(list(with_datestamp.glob("checkpoints/*"))) > 2:
            print("Using checkpoints of", with_datestamp.absolute())
            return with_datestamp
    print(f"Did not find any trained checkpoints in {agent_path}.")
    exit(1)


if __name__ == "__main__":
    args = parser.parse_args()
    agent_path = args.agent_path.replace("game", args.game)
    trained_checkpoint = find_latest_trained_checkpoint(agent_path)
    renderer = Renderer(agent_path=trained_checkpoint,
                        fps=100,
                        deterministic=False,
                        env_kwargs=dict(render_oc_overlay=True),
                        render_predicate_probs=not(args.no_predicates))
    renderer.run()
