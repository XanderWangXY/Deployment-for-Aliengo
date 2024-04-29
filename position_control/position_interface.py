#!/usr/bin/python

import sys
import time
import math

#import numpy as np

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# low cmd
TARGET_PORT = 8007
LOCAL_PORT = 8082
TARGET_IP = "192.168.123.10"  # target IP address

LOW_CMD_LENGTH = 610
LOW_STATE_LENGTH = 771

HIGHLEVEL = 0x00
LOWLEVEL = 0xff


def jointLinearInterpolation(initPos, targetPos, rate):
    #rate = np.fmin(np.fmax(rate, 0.0), 1.0)
    if rate > 1.0:
        rate = 1.0
    elif rate < 0.0:
        rate = 0.0

    p = initPos * (1 - rate) + targetPos * rate
    return p


class Position:
    def __init__(self):
        self.d = {'FR_0': 0, 'FR_1': 1, 'FR_2': 2,
                  'FL_0': 3, 'FL_1': 4, 'FL_2': 5,
                  'RR_0': 6, 'RR_1': 7, 'RR_2': 8,
                  'RL_0': 9, 'RL_1': 10, 'RL_2': 11}

        self.PosStopF = math.pow(10, 9)
        self.VelStopF = 16000.0

        self.sin_mid_q = [0.0, 1.2, -2.0, 0.0, 1.2, -2.0, 0.0, 1.2, -2.0, 0.0, 1.2, -2.0]

        self.dt = 0.002
        self.qInit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.qDes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.sin_count = 0
        self.rate_count = 0
        self.Kp = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        self.Kd = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

        self.udp = sdk.UDP(LOCAL_PORT, TARGET_IP, TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH, -1)
        # udp = sdk.UDP(8082, "192.168.123.10", 8007, 610, 771)
        self.safe = sdk.Safety(sdk.LeggedType.Aliengo)
        self.cmd = sdk.LowCmd()
        self.state = sdk.LowState()
        self.udp.InitCmdData(self.cmd)
        self.cmd.levelFlag = LOWLEVEL
        self.Tpi = 0
        self.motiontime = 0

    def receive_state(self):
        self.udp.Recv()
        self.udp.GetRecv(self.state)

    def pos_init(self):
        self.time_counter()
        self.receive_state()

        j=0
        for i in self.d:
            self.qInit[j] = self.state.motorState[self.d[i]].q
            j+=1

        #self.Kp = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        #self.Kd = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.send_pos(self.qInit)

    def time_counter(self):
        time.sleep(self.dt)
        self.motiontime += 1

    def send_pos(self, qDes):
        #self.udp.Recv()
        #self.udp.GetRecv(self.state)
        j=0
        for i in self.d:
            self.cmd.motorCmd[self.d[i]].q = qDes[j]
            self.cmd.motorCmd[self.d[i]].dq = 0
            self.cmd.motorCmd[self.d[i]].Kp = self.Kp[j]
            self.cmd.motorCmd[self.d[i]].Kd = self.Kd[j]
            if (j % 3 == 0):
                self.cmd.motorCmd[self.d[i]].tau = -1.6
            else:
                self.cmd.motorCmd[self.d[i]].tau = 0
            j+=1

        self.udp.SetSend(self.cmd)
        self.udp.Send()

    def test(self):
        self.time_counter()
        self.receive_state()

        if (self.motiontime >= 50 and self.motiontime < 400):
            self.rate_count += 1
            rate = self.rate_count / 200.0
            for i in range(12):
                self.qDes[i] = jointLinearInterpolation(self.qInit[i], self.sin_mid_q[i], rate)
        freq_Hz = 1
        freq_rad = freq_Hz * 2 * math.pi
        t = self.dt * self.sin_count
        if (self.motiontime >= 400):
            self.sin_count += 1
            sin_joint = [0, 0.6 * math.sin(t * freq_rad), -0.9 * math.sin(t * freq_rad)]
            for i in range(12):
                self.qDes[i] = self.sin_mid_q[i] + sin_joint[i % 3]

        self.send_pos(self.qDes)


if __name__ == '__main__':
    position = Position()
    for i in range(50):
        position.pos_init()
    while True:
        position.test()
