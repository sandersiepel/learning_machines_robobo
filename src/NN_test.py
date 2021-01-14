from __future__ import print_function
from TrainQLearning import Environment
from NN_Controller import ECEnvironment
from Population import Individual


FILENAME = "results_EC/best_weights_40.pickle"
SIMULATIONS = 5
ITERATIONS = 1_000


class TestEnvironment:
    def __init__(self, env):
        self.env = env  # Main environment for all of the functions etc.
        env.MAX_STEPS = 1000

    def run(self):
        # This function can be used to test a Q-table. It will simply run the environment with a deterministic policy
        # based on the Q-table (so it always chooses its best action).
        # self.env.q_table = self.env.read_q_table(FILENAME)

        rewards = []

        ind = Individual()
        ind.read_weights(FILENAME)

        for i in range(SIMULATIONS):
            self.env.rob.wait_for_ping()
            self.env.rob.play_simulation()

            simulation_reward = self.env.eval_ind(ind)

            # Simulation has ended, collect rewards and print stats
            rewards.append(simulation_reward)
            self.env.rob.stop_world()
            print(f"Simulation {i+1} of {SIMULATIONS} ended, total collected reward over {ITERATIONS} iterations: {simulation_reward}")

        print(f"Average reward over {SIMULATIONS} simulations * {ITERATIONS} iterations: {sum(rewards) / len(rewards)}")


def main():
    env = ECEnvironment()
    test_env = TestEnvironment(env)
    test_env.run()


if __name__ == "__main__":
    main()
