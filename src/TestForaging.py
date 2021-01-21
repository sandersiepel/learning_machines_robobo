from __future__ import print_function
from TrainForaging import Environment
import vrep
import random

FILENAME = "results/CRS_ellipse_draw4/q_table_data_50_250_CRS_ellipse_draw4_3.pickle"
M_FILENAME = "results/ChangedEpsilon/q_table_data_100_500_ChangedEpsilon_"
SIMULATIONS = 20
ITERATIONS = 250
EXPERIMENT = False


# noinspection PyProtectedMember
class TestEnvironment:
    def __init__(self, env):
        self.env = env  # Main environment for all of the functions etc.
        self.iteration_counter = 0
        self.env.q_table = self.env.read_q_table(FILENAME)

    def run(self):
        # This function can be used to test a Q-table. It will simply run the environment with a deterministic policy
        # based on the Q-table (so it always chooses its best action).
        rewards = []
        food_eaten = []
        steps = []
        # self.place_food()

        for i in range(SIMULATIONS):
            simulation_reward = 0
            self.env.rob.wait_for_ping()
            self.env.rob.play_simulation()

            while self.env.rob.collected_food() < 6 and self.iteration_counter < ITERATIONS:
                # Choose rob's best action (deterministic) and keep track of rewards.
                curr_state = self.env.handle_state()  # Check in what state rob is, return tuple e.g. (0, 0, 0, 1, 0)

                # Choose its best action (deterministic).
                best_action = self.env.best_action_for_state(curr_state)

                # Given our selected action (whether best or random), perform this action and update the Q-table.
                _, reward = self.env.handle_action(best_action, curr_state)

                # Update counters
                simulation_reward += reward
                self.iteration_counter += 1

            # Simulation has ended, collect rewards, reset counters and print stats
            rewards.append(simulation_reward)
            food_eaten.append(self.env.rob.collected_food())
            steps.append(self.iteration_counter)

            print(f"Simulation {i + 1} of {SIMULATIONS} ended, reward over {self.iteration_counter} steps: {simulation_reward}. Food eaten: {self.env.rob.collected_food()}")
            self.iteration_counter = 0
            self.env.rob.stop_world()

        print(f"Average reward over {SIMULATIONS} simulations: {sum(rewards) / len(rewards)}\n"
              f"Average amount of steps taken: {sum(steps) / len(steps)}\n"
              f"Average amount of foods eaten over 20 simulations: {sum(food_eaten) / len(food_eaten)}")

    def place_food(self):
        _, handlewallXone = vrep.simxGetObjectHandle(self.env.rob._clientID, '80cmHighWall200cm', vrep.simx_opmode_blocking)
        _, handlewallXtwo = vrep.simxGetObjectHandle(self.env.rob._clientID, '80cmHighWall200cm4', vrep.simx_opmode_blocking)
        _, posXone = vrep.simxGetObjectPosition(self.env.rob._clientID, handlewallXone, -1, vrep.simx_opmode_blocking)
        _, posXtwo = vrep.simxGetObjectPosition(self.env.rob._clientID, handlewallXtwo, -1, vrep.simx_opmode_blocking)
        _, handlewallYone = vrep.simxGetObjectHandle(self.env.rob._clientID, '80cmHighWall200cm3', vrep.simx_opmode_blocking)
        _, handlewallYtwo = vrep.simxGetObjectHandle(self.env.rob._clientID, '80cmHighWall200cm7', vrep.simx_opmode_blocking)
        _, posYone = vrep.simxGetObjectPosition(self.env.rob._clientID, handlewallYone, -1, vrep.simx_opmode_blocking)
        _, posYtwo = vrep.simxGetObjectPosition(self.env.rob._clientID, handlewallYtwo, -1, vrep.simx_opmode_blocking)

        for i in range(0, 9):
            _, handle = vrep.simxGetObjectHandle(self.env.rob._clientID, 'Food' + str(i), vrep.simx_opmode_blocking)
            x_pos = random.uniform(posXone[0] + 0.3, posXtwo[0] - 0.3)
            y_pos = random.uniform(posYone[1] - 0.3, posYtwo[1] + 0.3)
            vrep.simxSetObjectPosition(self.env.rob._clientID, handle, -1, [x_pos, y_pos, 0.04], vrep.simx_opmode_oneshot)


def main():
    env = Environment()
    if not EXPERIMENT:
        test_env = TestEnvironment(env)
        test_env.run()
    else:
        for i in range(5):
            global FILENAME
            FILENAME = M_FILENAME + str(i+1) + ".pickle"
            test_env = TestEnvironment(env)
            test_env.run()


if __name__ == "__main__":
    main()
