import random
import threading
import numpy as np
import vrep


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
    DISCOUNT_FACTOR = .9
    EPSILON = 0.9  # Start epsilon value.
    action_space = [0, 1, 2, 3, 4]
    physical_collision_counter = 0
    collision_counter = 0


    def __init__(self, robot, q_table=None, seed=42, log=None, level=2):
        super(Prey, self).__init__()
        self.q_table = q_table
        self._log = log
        self._robot = robot
        # seed for the random function -> make reproducible the experiment
        self._seed = seed
        # default level is 2 -> medium
        self._level = level
        _, self.collision_handle = vrep.simxGetCollisionHandle(self._robot._clientID, 'Hitbox0',
                                                               vrep.simx_opmode_blocking)
        self.reward = 0

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
            # print(self.q_table.shape)
            curr_state = self.handle_state()

            # Do we perform random action (due to epsilon < 1) or our best possible action?
            best_action = self.determine_action(curr_state)

            # Given our selected action (whether best or random), perform this action and update the Q-table.
            self.reward = self.update_q_table(best_action, curr_state)

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
        sensor_values = (sensor_values - -0.65) / 0.65  # Scales all variables between [0, 1] where 0 is close proximity.

        # Check what the actual sensor_values are (between [0, 1]) and determine their state
        indices = []
        for sensor_value in sensor_values:
            if sensor_value >= 0.8:  # No need for action, moving forward is best.
                indices.append(0)
            elif -0.9 <= sensor_value < 0.8:
                indices.append(1)

        # Return the values in tuple format, with which we can index our Q-table. This tuple is a representation
        # of the current state our robot is in (i.e. what does the robot see with its sensors).
        return tuple(indices)

    def get_reward(self):
        return self.reward

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
        collision_front, collision_back = self.collision()  # Do we collide, returns either "nothing", "far" or "close"
        collision2 = self.physical_collision()

        if collision2:
            self.physical_collision_counter += 1

        reward = self.determine_reward(collision_front, collision_back, action)

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

    @staticmethod
    def determine_reward(collision_front, collision_back, action):
        # This function determines the reward an action should get, depending on whether or not rob is about to
        # collide with an object within the environment.
        reward = 0
        print("collision front: " + str(collision_front))
        print("collision back: " + str(collision_back))
        if action in [0, 1]:  # Action is moving either left or right.
            if collision_front:
                reward += 3
        elif action in [3, 4]:
            if collision_front:
                reward += 1
        elif action == 2:  # Action is moving forward.
            if not collision_front:
                reward += 5

        print("action: " + str(action))
        print("reward: " + str(reward) + "\n")
        return reward

    def determine_action(self, curr_state):
        return random.choice(self.action_space) if random.random() < (1 - self.EPSILON) else self.best_action_for_state(curr_state)

    def collision(self):
        # This function checks whether rob is close to something or not. It returns the "distance", either "close", "far" or "nothing".
        # It also keeps track of the collision counter. If this counter exceeds its threshold (COLLISION_THRESHOLD)
        # then the environment should reset (to avoid rob getting stuck).
        try:
            sensor_values = self._robot.read_irs()  # Should be absolute values (no log or anything).
        except:
            sensor_values = [0, 0, 0, 0, 0, 0, 0, 0]


        print(str(sensor_values))
        collision_front = any([0 < i < 0.2 for i in sensor_values[3:]])
        collision_back = any([0 < i < 0.2 for i in sensor_values[:3]])
        # collision_far = any([0.13 <= i < 0.2 for i in sensor_values])
        # collision_close = any([0 < i < 0.13 for i in sensor_values])

        return collision_front, collision_back

    def best_action_for_state(self, state):
        # Given a state (tuple format), what is the best action we take, i.e. for which action is the Q-value highest?
        q_row = self.q_table[state]

        print(q_row)
        max_val_indices = [i for i, j in enumerate(q_row) if j == max(q_row)]
        best_action = random.choice(max_val_indices) if len(max_val_indices) > 1 else np.argmax(q_row)

        return best_action

    def physical_collision(self):
        # This function checks for physical collision that is not based on the sensors, but on an object around the Robot
        # in V-REP.
        [_, collision_state] = vrep.simxReadCollision(self._robot._clientID, self.collision_handle, vrep.simx_opmode_streaming)

        return collision_state

