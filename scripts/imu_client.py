import socket
import time
import numpy as np
import pickle
from queue import Queue
from threading import Thread
from scipy.spatial.transform import Rotation as R
from FramesViewer.viewer import Viewer

from mini_bdx_runtime.rl_utils import quat_rotate_inverse


class IMUClient:
    def __init__(self, host, port=1234, freq=30):
        self.host = host
        self.port = port
        self.freq = freq
        self.client_socket = socket.socket()
        self.connected = False
        while not self.connected:
            try:
                self.client_socket.connect((self.host, self.port))
                self.connected = True
            except Exception as e:
                print(e)
                time.sleep(0.5)
        self.imu_queue = Queue(maxsize=1)
        self.last_imu = [0, 0, 0, 0]

        Thread(target=self.imu_worker, daemon=True).start()

    def imu_worker(self):
        while True:
            try:
                data = self.client_socket.recv(1024)  # receive response
                data = pickle.loads(data)

                self.imu_queue.put(data)
            except:
                print("missed imu")

            time.sleep(1 / self.freq)

    def get_imu(self):
        try:
            self.last_imu = self.imu_queue.get(False)
        except:
            pass

        return self.last_imu


if __name__ == "__main__":

    client = IMUClient("192.168.248.253")
    fv = Viewer()
    fv.start()
    pose = np.eye(4)
    pose[:3, 3] = [0.1, 0.1, 0.1]
    projected_gravities = []
    try:
        while True:
            quat, gyro = client.get_imu()
            try:
                print(gyro)
                rot_mat = R.from_quat(quat).as_matrix()
                pose[:3, :3] = rot_mat
                fv.pushFrame(pose, "aze")

                # euler = R.from_quat(quat).as_euler("xyz")
                # euler[2] = 0
                # quat = R.from_euler("xyz", euler).as_quat()

                projected_gravity = quat_rotate_inverse(quat, [0, 0, -1])
                projected_gravities.append(projected_gravity)
            except Exception as e:
                print("error", e)
                pass
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass

    # plot projected_gravities [[x, y, z], [x, y, z], ...]

    # from matplotlib import pyplot as plt

    # projected_gravities = np.array(projected_gravities)
    # plt.plot(projected_gravities)
    # plt.show()
