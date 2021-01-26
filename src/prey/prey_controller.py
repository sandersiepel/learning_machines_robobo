import random
import threading
import numpy as np


class Direction:
    LEFT = (-5, 5, 300)  # Action: 0, left
    RIGHT = (5, -5, 300)  # Action: 1, right
    FORWARD = (25, 25, 300)  # Action: 2, forward
    RRIGHT = (-15, 15, 300)  # Action: 3, strong right
    LLEFT = (15, -15, 300)  # Action: 4, strong left


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
        regularly for the stopped() condition."""

    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop_event = threading.Event()

    def stop(self):
        # TODO: Save q-table
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class Prey(StoppableThread):
    LEARNING_RATE = .1
    DISCOUNT_FACTOR = .8
    EPSILON = 0.9  # Start epsilon value.


    def __init__(self, robot, seed=42, log=None, level=2):
        super(Prey, self).__init__()
        self._log = log
        self._robot = robot
        # seed for the random function -> make reproducible the experiment
        self._seed = seed
        # default level is 2 -> medium
        self._level = level

    def _sensor_better_reading(self, sensors_values):
        """
        Normalising simulation sensor reading due to reuse old code
        :param sensors_values:
        :return:
        """
        old_min = 0
        old_max = 0.20
        new_min = 20000
        new_max = 0
        return [0 if value is False else (((value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min for value in sensors_values]

    def run(self):
        # TODO: read q-table
        while not self.stopped():
            curr_state = self.handle_state()

            # Do we perform random action (due to epsilon < 1) or our best possible action?
            best_action = self.determine_action(curr_state)

            # Given our selected action (whether best or random), perform this action and update the Q-table.
            reward = self.update_q_table(best_action, curr_state)


            # self._robot.move(left=20.0, right=-10.0, millis=200)
            # sensors = self._sensor_better_reading(self._robot.read_irs())
            # print('moved, sensors: ', sensors)

    def handle_state(self):
        # This function should return the values with which we can index our q_table, in tuple format.
        # So, it should take the last 5 sensor inputs (current state), transform each of them into a bucket where
        # the bucket size is already determined by the shape of the q_table.
        try:
            sensor_values = np.log(np.array(self._robot.read_irs())) / 10  #
        except:
            sensor_values = [0, 0, 0, 0, 0, 0, 0, 0]

        sensor_values = np.where(sensor_values == -np.inf, 0, sensor_values)  # Remove the infinite values.
        sensor_values = (
                                    sensor_values - -0.65) / 0.65  # Scales all variables between [0, 1] where 0 is close proximity.

        # Check what the actual sensor_values are (between [0, 1]) and determine their state
        indices = []
        for sensor_value in sensor_values:
            if sensor_value >= 0.8:  # No need for action, moving forward is best.
                indices.append(0)
            elif 0.65 <= sensor_value < 0.8:
                indices.append(1)
            elif sensor_value < 0.65:  # We see an object, but not really close yet.
                indices.append(2)

        # Return the values in tuple format, with which we can index our Q-table. This tuple is a representation
        # of the current state our robot is in (i.e. what does the robot see with its sensors).
        return tuple(indices)

    def update_q_table(self, best_action, curr_state):
        # This function updates the Q-table accordingly to the current state of rob.
        # First, we determine the new state we end in if we would play our current best action, given our current state.
        new_state, reward = self.handle_action(best_action)

        # Then we calculate the reward we would get in this new state.
        max_future_q = np.amax(self.q_table[new_state])

        # Check what Q-value our current action has.
        current_q = self.q_table[curr_state][best_action]

        # Calculate the new Q-value with the common formula
        new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (reward + self.DISCOUNT_FACTOR * max_future_q)

        # And lastly, update the value in the Q-table.
        self.q_table[curr_state][best_action] = new_q

        return reward

    def handle_action(self, action):
        # This function should accept an action (0, 1, 2...) and move the robot accordingly (left, right, forward).
        # It returns two things: new_state, which is the state (in tuple format) after this action has been performed.
        # and reward, which is the reward from this action.
        collision = self.collision()  # Do we collide, returns either "nothing", "far" or "close"
        collision2 = self.physical_collision()

        if collision2:
            self.physical_collision_counter += 1

        reward = self.determine_reward(collision, action)

        if action == 0:
            left, right, duration = Direction.LEFT  # Left, action 0
        elif action == 1:
            left, right, duration = Direction.RIGHT  # Right, action 1
        elif action == 3:
            left, right, duration = Direction.RRIGHT  # Extreme right, action 3
        elif action == 4:
            left, right, duration = Direction.LLEFT  # Extreme left, action 4
        else:
            left, right, duration = Direction.FORWARD  # Forward, action 2

        self._robot.move(left, right, duration)
        return self.handle_state(), reward # New_state, reward

    def collision(self):
        # This function checks whether rob is close to something or not. It returns the "distance", either "close", "far" or "nothing".
        # It also keeps track of the collision counter. If this counter exceeds its threshold (COLLISION_THRESHOLD)
        # then the environment should reset (to avoid rob getting stuck).
        try:
            sensor_values = self._robot.read_irs()[3:]  # Should be absolute values (no log or anything).
        except:
            sensor_values = [0, 0, 0, 0, 0]

        collision_far = any([0.13 <= i < 0.2 for i in sensor_values])
        collision_close = any([0 < i < 0.13 for i in sensor_values])

        if collision_close:
            self.collision_counter += 1
            return "close"
        elif collision_far:
            return "far"
        else:
            self.collision_counter = 0
            return "nothing"

    @staticmethod
    def determine_reward(collision, action):
        # This function determines the reward an action should get, depending on whether or not rob is about to
        # collide with an object within the environment.
        reward = 0

        if action in [0, 1]:  # Action is moving either left or right.
            if collision == "nothing":
                reward -= 1
            elif collision == "far":
                reward += 1
            elif collision == "close":
                reward -= 1
        elif action in [3, 4]:
            if collision == "nothing":
                reward -= 1
            elif collision == "far":
                reward -= 1
            elif collision == "close":
                reward += 2
        elif action == 2:  # Action is moving forward.
            if collision == "far":
                reward -= 3
            elif collision == "close":
                reward -= 5
            elif collision == "nothing":
                reward += 3

        return reward

    def determine_action(self, curr_state):
        return random.choice(self.action_space) if random.random() < (1 - self.EPSILON_LOW) else self.best_action_for_state(curr_state)


