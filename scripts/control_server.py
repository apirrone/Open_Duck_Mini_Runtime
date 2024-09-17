import socket
import time
import pickle
import numpy as np
import pygame
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--controller",
    action="store_true",
    default=False,
    help="if not set, uses keyboard",
)
args = parser.parse_args()

X_RANGE = [0, 0.12]
Y_RANGE = [0, 0.0]
YAW_RANGE = [-0.7, 0.7]

if args.controller:
    pygame.init()
    _p1 = pygame.joystick.Joystick(0)
    _p1.init()
    print(f"Loaded joystick with {_p1.get_numaxes()} axes.")
else:
    pygame.init()
    # open a blank pygame window
    screen = pygame.display.set_mode((100, 100))
    pygame.display.set_caption("Press arrow keys to move robot")

commands = [0, 0, 0]


def get_command():
    global commands
    # commands = [0, 0, 0]
    if args.controller:
        for event in pygame.event.get():
            lin_vel_x = -1 * _p1.get_axis(1)
            # lin_vel_y = -1 * _p1.get_axis(3)
            ang_vel = -1 * _p1.get_axis(0)
            if lin_vel_x >= 0:
                lin_vel_x *= np.abs(X_RANGE[1])
            else:
                lin_vel_x *= np.abs(X_RANGE[0])

            # if lin_vel_y >= 0:
            #     lin_vel_y *= np.abs(Y_RANGE[1])
            # else:
            #     lin_vel_y *= np.abs(Y_RANGE[0])

            if ang_vel >= 0:
                ang_vel *= np.abs(YAW_RANGE[1])
            else:
                ang_vel *= np.abs(YAW_RANGE[0])

            commands[0] = lin_vel_x
            commands[1] = 0
            commands[2] = ang_vel
    else:
        keys = pygame.key.get_pressed()
        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel = 0
        if keys[pygame.K_z]:
            lin_vel_x = X_RANGE[1]
        elif keys[pygame.K_s]:
            lin_vel_x = X_RANGE[0]

        if keys[pygame.K_q]:
            ang_vel = YAW_RANGE[1]
        elif keys[pygame.K_d]:
            ang_vel = YAW_RANGE[0]

        commands[0] = lin_vel_x
        commands[1] = lin_vel_y
        commands[2] = ang_vel

    pygame.event.pump()  # process event queue

    print(commands)
    return commands


host = "0.0.0.0"
port = 1234

server_socket = socket.socket()
server_socket.bind((host, port))

while True:
    server_socket.listen(1)
    conn, address = server_socket.accept()  # accept new connection
    print("Connection from: " + str(address))
    try:
        while True:
            data = get_command()
            data = pickle.dumps(data)
            conn.send(data)  # send data to the client
            time.sleep(1 / 10)
    except:
        pass

conn.close()  # close the connection
