import socket
import time
import pickle
from queue import Queue
from threading import Thread


class CommandsClient:
    def __init__(self, host, port=1234, freq=10):
        self.host = host
        self.port = port
        self.freq = 10
        self.client_socket = socket.socket()
        self.client_socket.connect((self.host, self.port))
        self.commands_queue = Queue(maxsize=1)
        self.last_command = [0, 0, 0]

        Thread(target=self.commands_worker, daemon=True).start()

    def commands_worker(self):
        while True:
            data = self.client_socket.recv(1024)  # receive response
            data = pickle.loads(data)

            self.commands_queue.put(data)

            time.sleep(1 / self.freq)

    def get_command(self):
        try:
            self.last_command = self.commands_queue.get(False)
        except:
            pass
        return self.last_command


if __name__ == "__main__":
    client = CommandsClient("192.168.89.246")

    while True:
        print(client.get_command())
        time.sleep(1 / 30)
