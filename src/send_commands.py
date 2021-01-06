#!/usr/bin/env python3
from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey
import random
import pickle


def store_q_table(q_table):
    with open('q_table', 'wb') as fp:
        pickle.dump(q_table, fp)


def read_q_table():
    with open('q_table', 'rb') as fp:
        q_table = pickle.load(fp)
    return q_table


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def initialize_state():
    q_table = np.random.uniform(low=0, high=0, size=([2, 2, 2, 2, 2] + [3]))
    return q_table

def get_prox(rob):
    sensors = np.log(np.array(rob.read_irs())) / 10
    sensors = np.where(sensors[-5:] == -np.inf, 0, sensors[-5:])  # remove the infinite
    sensors = (sensors - -0.65) / 0.65
    prox = []
    is_hit = False
    for sensor in sensors:
        if sensor < 0.6:
            d = 0
            if sensor < 0.3:
                is_hit = True
        else:
            d = 1
        prox.append(d)
    return prox, is_hit


def move_left(rob):
    rob.move(-10, 10, 250)


def move_right(rob):
    rob.move(10, -10, 250)


def move_forward(rob):
    rob.move(10, 10, 250)


def reset_world(rob):
    rob.pause_simulation()
    rob.stop_world()
    rob.play_simulation()


def main():
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.95

    signal.signal(signal.SIGINT, terminate_program)

    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    rob = robobo.SimulationRobobo().connect(address='192.168.1.3', port=19997)

    rob.play_simulation()

    done = False

    # q_table = initialize_state()
    q_table = read_q_table()
    previous_prox = np.array([1, 1, 1, 1, 1])

    for i in range(500):
        previous_q_value = np.amax(q_table[previous_prox[0]][previous_prox[1]][previous_prox[2]][previous_prox[3]][previous_prox[4]])

        prox, is_hit = get_prox(rob)
        q_values = q_table[prox[0]][prox[1]][prox[2]][prox[3]][prox[4]]

        # print(q_values)
        max_q = np.argmax(q_values)

        q_array = np.where(q_values == np.amax(q_values))[0]

        # print(q_array)

        if len(q_array > 1):
            best_q = q_array[random.randint(0,len(q_array)-1)]
        else:
            best_q = q_array

        if random.randint(0,10) < 2:
            best_q = random.randint(0,2)

        if best_q == 0:
            move_left(rob)
        elif best_q == 1:
            move_right(rob)
        elif best_q == 2:
            move_forward(rob)
            # print("nice")

        if is_hit:
            print("HIT!!")
            store_q_table(q_table)
            reward = -10
            reset_world(rob)
        elif (not is_hit) and best_q == 2:
            reward = 5
        else:
            reward = 1

        new_q = previous_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_q - previous_q_value)
        q_table[previous_prox[0]][previous_prox[1]][previous_prox[2]][previous_prox[3]][previous_prox[4]][best_q] = new_q
        previous_prox = prox
        # print("previous: " + str(previous_q_value) + " new: " + str(new_q))

    print(q_table)
    store_q_table(q_table)

    # pause the simulation and read the collected food
    # rob.pause_simulation()
    
    # Stopping the simualtion resets the environment
    rob.stop_world()


if __name__ == "__main__":
    main()
