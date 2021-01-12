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
    errorCode, handle = vrep.simxGetCollisionHandle(rob._clientID, 'Collision', vrep.simx_opmode_blocking)
    collision = 0
    for i in range(10):
        print("robobo is at {}".format(rob.position()))
        rob.move(10, 10, 2000)

        [collidingObjectHandles, collisionState] = vrep.simxReadCollision(rob._clientID, handle, vrep.simx_opmode_streaming)
        print(collisionState)
        if collisionState != False:
            collision +=1
            print('collisions:' + str(collision))
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