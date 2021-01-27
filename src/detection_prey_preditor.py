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

ClientID = '192.168.0.185'


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def main():
    signal.signal(signal.SIGINT, terminate_program)

    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    rob = robobo.SimulationRobobo().connect(address= ClientID, port=19997)

    rob.play_simulation()

    print("robobo is at {}".format(rob.position()))
    rob.sleep(1)
    #collision detection
    _, handle = vrep.simxGetCollisionHandle(rob._clientID, 'Collision', vrep.simx_opmode_blocking)
    _, handletest = vrep.simxGetCollisionHandle(rob._clientID, 'Collision_walls', vrep.simx_opmode_blocking)
    _, prey = vrep.simxGetObjectHandle(rob._clientID, 'Robobo#0', vrep.simx_opmode_blocking)
    _, preditor = vrep.simxGetObjectHandle(rob._clientID, 'Robobo', vrep.simx_opmode_blocking)
    collision = 0
    for i in range(10):
        print("robobo is at {}".format(rob.position()))
        rob.move(10, 10, 2000)

        _, collisionState = vrep.simxReadCollision(rob._clientID, handle, vrep.simx_opmode_streaming)
        error, collisionState_walls = vrep.simxReadCollision(rob._clientID, handletest, vrep.simx_opmode_streaming)
        print(error, collisionState_walls)
        if collisionState != False:
            print('Robobo caught the prey')
        if collisionState_walls != False:
            print('Prey hit the wall')
        #distance measures
        _, pos_prey = vrep.simxGetObjectPosition(rob._clientID, prey, -1, vrep.simx_opmode_blocking)
        _, pos_preditor = vrep.simxGetObjectPosition(rob._clientID, preditor, -1, vrep.simx_opmode_blocking)
        distance = np.sqrt((pos_prey[0]-pos_preditor[0])**2+(pos_prey[1]-pos_preditor[1])**2)
        print(distance)

    #end collision detection
    rob.sleep(1)


    time.sleep(0.1)

    # IR reading

    # pause the simulation and read the collected food
    rob.pause_simulation()

    # Stopping the simualtion resets the environment
    rob.stop_world()


if __name__ == "__main__":
    main()