import time

from src.getout.getout.getout import Getout
from src.getout.getout.getout import DummyGenerator
from src.getout.getout.getout import GetoutActions

def run():
    coin_jump = Getout(render=False)
    level_generator = DummyGenerator()
    level_generator.generate(coin_jump, generate_enemies=False)

    num_iterations = 100000
    action_left = True
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        # control framerate

        # step game
        if action_left:
            action = [GetoutActions.MOVE_LEFT]
        else:
            action = [GetoutActions.MOVE_RIGHT]
        reward = coin_jump.step(action)
        representation = coin_jump.get_representation()
        print(representation)
        break

    end_time = time.perf_counter()
    duration_total = (end_time - start_time)
    duration_step = duration_total / num_iterations
    print(f"{duration_step*1000}ms per step (total {duration_total}s with {num_iterations} steps)")

    print("Maze terminated")


if __name__ == "__main__":
    run()

