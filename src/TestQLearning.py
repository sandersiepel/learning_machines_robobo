from __future__ import print_function
from TrainQLearning import Environment


FILENAME = "results/train_week1/q_table_data_100_500_train_week1_5.pickle"
M_FNAME = FILENAME = "results/ChangedEpsilon/q_table_data_100_500_ChangedEpsilon_"
SIMULATIONS = 5
ITERATIONS = 1_000
EXPERIMENT = True


class TestEnvironment:
    def __init__(self, env):
        self.env = env  # Main environment for all of the functions etc.

    def run(self):
        # This function can be used to test a Q-table. It will simply run the environment with a deterministic policy
        # based on the Q-table (so it always chooses its best action).
        self.env.q_table = self.env.read_q_table(FILENAME)

        rewards = []
        collisions = []
        for i in range(SIMULATIONS):
            simulation_reward = 0
            simulation_collisions = 0
            self.env.rob.wait_for_ping()
            self.env.rob.play_simulation()
            for _ in range(ITERATIONS):
                # Choose rob's best action (deterministic) and keep track of rewards.
                curr_state = self.env.handle_state()  # Check in what state rob is, return tuple e.g. (0, 0, 0, 1, 0)

                # Choose its best action (deterministic).
                best_action = self.env.best_action_for_state(curr_state)

                # Given our selected action (whether best or random), perform this action and update the Q-table.
                _, reward = self.env.handle_action(best_action)
                simulation_reward += reward

                # Keep track of collisions
                collision = self.env.physical_collision()
                if collision:
                    simulation_collisions += 1

            # Simulation has ended, collect rewards and print stats
            rewards.append(simulation_reward)
            collisions.append(simulation_collisions)

            self.env.rob.stop_world()
            print(f"Simulation {i+1} of {SIMULATIONS} ended, reward over {ITERATIONS} iterations: {simulation_reward}. Collisions: {simulation_collisions}")

        print(f"Average reward over {SIMULATIONS} simulations * {ITERATIONS} iterations: {sum(rewards) / len(rewards)}. Avg collisions: {sum(collisions) / len(collisions)}")


def main(experiment):
    env = Environment()
    if not experiment:
        test_env = TestEnvironment(env)
        test_env.run()
    else:
        for i in range(5):
            global FILENAME
            FILENAME = M_FNAME + str(i+1) + ".pickle"
            test_env = TestEnvironment(env)
            test_env.run()


if __name__ == "__main__":
    main(EXPERIMENT)
