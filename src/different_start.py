from __future__ import print_function

import time
import numpy as np
import vrep
import robobo
import cv2
import sys
import signal
import prey
import vrep
import random

ClientID = '192.168.0.185'


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def main():
    signal.signal(signal.SIGINT, terminate_program)

    rob = robobo.SimulationRobobo().connect(address= ClientID, port=19997)

    errorCode, robot = vrep.simxGetObjectHandle(rob._clientID, 'Robobo', vrep.simx_opmode_blocking)
    errorCode, handle0 = vrep.simxGetObjectHandle(rob._clientID, 'Start#0', vrep.simx_opmode_blocking)
    errorCode, handle1 = vrep.simxGetObjectHandle(rob._clientID, 'Start#1', vrep.simx_opmode_blocking)
    errorCode, handle2 = vrep.simxGetObjectHandle(rob._clientID, 'Start#2', vrep.simx_opmode_blocking)
    errorCode, handle3 = vrep.simxGetObjectHandle(rob._clientID, 'Start#3', vrep.simx_opmode_blocking)
    errorCode, handle4 = vrep.simxGetObjectHandle(rob._clientID, 'Start#4', vrep.simx_opmode_blocking)
    errorCode, handle5 = vrep.simxGetObjectHandle(rob._clientID, 'Start#5', vrep.simx_opmode_blocking)
    errorCode, handle6 = vrep.simxGetObjectHandle(rob._clientID, 'Start#6', vrep.simx_opmode_blocking)
    list_starting = [handle0,handle1,handle2,handle3,handle4,handle5,handle6]
    random_handle = random.choice(list_starting)
    #print(random_handle, robot)
    error , start_pos = vrep.simxGetObjectPosition(rob._clientID, random_handle, -1, vrep.simx_opmode_blocking)
    print(start_pos)
    start_pos = [start_pos[0], start_pos[1],0.04]
    vrep.simxSetObjectPosition(rob._clientID,robot,-1 ,start_pos, vrep.simx_opmode_oneshot)
    time.sleep(5)
    rob.play_simulation()

    print("robobo is at {}".format(rob.position()))

    for i in range(3):
        print("robobo is at {}".format(rob.position()))
        rob.move(10, 10, 2000)


    time.sleep(0.1)

    # IR reading

    # pause the simulation and read the collected food

    # Stopping the simualtion resets the environment
    rob.stop_world()


if __name__ == "__main__":
    main()