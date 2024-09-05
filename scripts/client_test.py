import socket
import json


def client_program():
    host = "192.168.89.246"  # as both code is running on same pc
    port = 1234  # socket server port number

    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server

    while True:
        data = client_socket.recv(1024).decode()  # receive response
        data = json.loads(data)

        print("Received from server: " + data)  # show in terminal

    client_socket.close()  # close the connection


if __name__ == "__main__":
    client_program()
