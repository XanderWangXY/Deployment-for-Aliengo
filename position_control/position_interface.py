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

        self.sin_mid_q = [0.0, 1.2, -2.0]

        self.dt = 0.002
        self.qInit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.qDes = [0, 0, 0]
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
        self.qInit[0] = self.state.motorState[self.d['FR_0']].q
        self.qInit[1] = self.state.motorState[self.d['FR_1']].q
        self.qInit[2] = self.state.motorState[self.d['FR_2']].q
        self.qInit[3] = self.state.motorState[self.d['FR_3']].q
        self.qInit[4] = self.state.motorState[self.d['FR_4']].q
        self.qInit[5] = self.state.motorState[self.d['FR_5']].q
        self.qInit[6] = self.state.motorState[self.d['FR_6']].q
        self.qInit[7] = self.state.motorState[self.d['FR_7']].q
        self.qInit[8] = self.state.motorState[self.d['FR_8']].q
        self.qInit[9] = self.state.motorState[self.d['FR_9']].q
        self.qInit[10] = self.state.motorState[self.d['FR_10']].q
        self.qInit[11] = self.state.motorState[self.d['FR_11']].q

        #self.Kp = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        #self.Kd = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    def time_counter(self):
        time.sleep(self.dt)
        self.motiontime += 1

    def send_pos(self, qDes):
        self.udp.Recv()
        self.udp.GetRecv(self.state)

        self.cmd.motorCmd[self.d['FR_0']].q = qDes[0]
        self.cmd.motorCmd[self.d['FR_0']].dq = 0
        self.cmd.motorCmd[self.d['FR_0']].Kp = self.Kp[0]
        self.cmd.motorCmd[self.d['FR_0']].Kd = self.Kd[0]
        self.cmd.motorCmd[self.d['FR_0']].tau = -1.6

        self.cmd.motorCmd[self.d['FR_1']].q = qDes[1]
        self.cmd.motorCmd[self.d['FR_1']].dq = 0
        self.cmd.motorCmd[self.d['FR_1']].Kp = self.Kp[1]
        self.cmd.motorCmd[self.d['FR_1']].Kd = self.Kd[1]
        self.cmd.motorCmd[self.d['FR_1']].tau = 0

        self.cmd.motorCmd[self.d['FR_2']].q = qDes[2]
        self.cmd.motorCmd[self.d['FR_2']].dq = 0
        self.cmd.motorCmd[self.d['FR_2']].Kp = self.Kp[2]
        self.cmd.motorCmd[self.d['FR_2']].Kd = self.Kd[2]
        self.cmd.motorCmd[self.d['FR_2']].tau = 0

        self.cmd.motorCmd[self.d['FR_3']].q = qDes[3]
        self.cmd.motorCmd[self.d['FR_3']].dq = 0
        self.cmd.motorCmd[self.d['FR_3']].Kp = self.Kp[3]
        self.cmd.motorCmd[self.d['FR_3']].Kd = self.Kd[3]
        self.cmd.motorCmd[self.d['FR_3']].tau = -1.6

        self.cmd.motorCmd[self.d['FR_4']].q = qDes[4]
        self.cmd.motorCmd[self.d['FR_4']].dq = 0
        self.cmd.motorCmd[self.d['FR_4']].Kp = self.Kp[4]
        self.cmd.motorCmd[self.d['FR_4']].Kd = self.Kd[4]
        self.cmd.motorCmd[self.d['FR_4']].tau = 0

        self.cmd.motorCmd[self.d['FR_5']].q = qDes[5]
        self.cmd.motorCmd[self.d['FR_5']].dq = 0
        self.cmd.motorCmd[self.d['FR_5']].Kp = self.Kp[5]
        self.cmd.motorCmd[self.d['FR_5']].Kd = self.Kd[5]
        self.cmd.motorCmd[self.d['FR_5']].tau = 0

        self.cmd.motorCmd[self.d['FR_6']].q = qDes[6]
        self.cmd.motorCmd[self.d['FR_6']].dq = 0
        self.cmd.motorCmd[self.d['FR_6']].Kp = self.Kp[6]
        self.cmd.motorCmd[self.d['FR_6']].Kd = self.Kd[6]
        self.cmd.motorCmd[self.d['FR_6']].tau = -1.6

        self.cmd.motorCmd[self.d['FR_7']].q = qDes[7]
        self.cmd.motorCmd[self.d['FR_7']].dq = 0
        self.cmd.motorCmd[self.d['FR_7']].Kp = self.Kp[7]
        self.cmd.motorCmd[self.d['FR_7']].Kd = self.Kd[7]
        self.cmd.motorCmd[self.d['FR_7']].tau = 0

        self.cmd.motorCmd[self.d['FR_8']].q = qDes[8]
        self.cmd.motorCmd[self.d['FR_8']].dq = 0
        self.cmd.motorCmd[self.d['FR_8']].Kp = self.Kp[8]
        self.cmd.motorCmd[self.d['FR_8']].Kd = self.Kd[8]
        self.cmd.motorCmd[self.d['FR_8']].tau = 0

        self.cmd.motorCmd[self.d['FR_9']].q = qDes[9]
        self.cmd.motorCmd[self.d['FR_9']].dq = 0
        self.cmd.motorCmd[self.d['FR_9']].Kp = self.Kp[9]
        self.cmd.motorCmd[self.d['FR_9']].Kd = self.Kd[9]
        self.cmd.motorCmd[self.d['FR_9']].tau = -1.6

        self.cmd.motorCmd[self.d['FR_10']].q = qDes[10]
        self.cmd.motorCmd[self.d['FR_10']].dq = 0
        self.cmd.motorCmd[self.d['FR_10']].Kp = self.Kp[10]
        self.cmd.motorCmd[self.d['FR_10']].Kd = self.Kd[10]
        self.cmd.motorCmd[self.d['FR_10']].tau = 0

        self.cmd.motorCmd[self.d['FR_11']].q = qDes[11]
        self.cmd.motorCmd[self.d['FR_11']].dq = 0
        self.cmd.motorCmd[self.d['FR_11']].Kp = self.Kp[11]
        self.cmd.motorCmd[self.d['FR_11']].Kd = self.Kd[11]
        self.cmd.motorCmd[self.d['FR_11']].tau = 0

        self.udp.SetSend(self.cmd)
        self.udp.Send()

    def test(self):
        self.time_counter()
        self.receive_state()

        if (self.motiontime >= 10 and self.motiontime < 400):
            self.rate_count += 1
            rate = self.rate_count / 200.0