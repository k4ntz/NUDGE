from nudge.renderer import Renderer
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--agent_path", type=str, default="out/runs/seaquest/logic/")




if __name__ == "__main__":
    renderer = Renderer(agent_path="out/runs/seaquest/logic/",
                        fps=15,
                        deterministic=False,
                        env_kwargs=dict(render_oc_overlay=True),
                        render_predicate_probs=True)
    renderer.run()
