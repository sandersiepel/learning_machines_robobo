import random
import threading


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
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class Prey(StoppableThread):
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
        while not self.stopped():
            self._robot.move(left=20.0, right=-10.0, millis=200)
            sensors = self._sensor_better_reading(self._robot.read_irs())
            print('moved, sensors: ', sensors)
