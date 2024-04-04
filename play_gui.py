from nudge.renderer import Renderer

if __name__ == "__main__":
    renderer = Renderer(agent_path="out/runs/seaquest/logic/24-04-04-16-56",
                        fps=15,
                        deterministic=False,
                        env_kwargs=dict(render_oc_overlay=True),
                        render_predicate_probs=True)
    renderer.run()
