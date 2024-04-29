#!/usr/bin/python

import sys
import time
import math

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# high cmd
TARGET_PORT = 8082
LOCAL_PORT = 8081
TARGET_IP = "192.168.123.220"   # target IP address

HIGH_CMD_LENGTH = 113
HIGH_STATE_LENGTH = 244

class Custom():
    def __init__(self):
        self.HIGHLEVEL = 0x00
        self.LOWLEVEL = 0xff

        # udp = sdk.UDP(8080, "192.168.123.161", 8082, 129, 1087, False, sdk.RecvEnum.nonBlock)
        # udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)

        self.cmd = sdk.HighCmd()
        self.state = sdk.HighState()

        self.udp = sdk.UDP(LOCAL_PORT, TARGET_IP, TARGET_PORT, HIGH_CMD_LENGTH, HIGH_STATE_LENGTH, -1)

        self.udp.InitCmdData(self.cmd)
        self.motiontime = 0
        self.dt = 0.002

    def UDPRecv(self):
        self.udp.Recv()

    def UDPSend(self):
        self.udp.Send()

    def ChangeLevel(self):
        self.motiontime+=2
        self.udp.GetRecv(self.state)

        self.cmd.velocity=[0.0, 0.0]
        self.cmd.position=[0.0, 0.0]
        self.cmd.yawSpeed=0.0

        self.cmd.mode=0
        self.cmd.rpy=[0.0, 0.0, 0.0]
        self.cmd.gaitType=0
        self.cmd.dBodyHeight=0
        self.cmd.dFootRaiseHeight=0

        if(self.motiontime==2):
            print("begin sending commands.")
        if(self.motiontime>10 and self.motiontime<100):
            self.cmd.levelFlag=0xf0
        if(self.motiontime==100):
            print("Aliengo sport mode trigger sent !")
        self.udp.SetSend(self.cmd)

    def example_walk(self):
        time.sleep(self.dt)
        self.motiontime += 1

        self.udp.Recv()
        self.udp.GetRecv(self.state)

        # print(motiontime, state.motorState[0].q, state.motorState[1].q, state.motorState[2].q)
        print(self.state.imu.rpy[2])

        self.cmd.mode = 0  # 0:idle, default stand      1:forced stand     2:walk continuously
        self.cmd.gaitType = 0
        self.cmd.speedLevel = 0
        self.cmd.dFootRaiseHeight = 0
        self.cmd.dBodyHeight = 0
        self.cmd.rpy = [0, 0, 0]
        self.cmd.velocity = [0, 0]
        self.cmd.yawSpeed = 0.0
        self.cmd.reserve = 0

        # cmd.mode = 2
        # cmd.gaitType = 1
        # # cmd.position = [1, 0]
        # # cmd.position[0] = 2
        # cmd.velocity = [-0.2, 0] # -1  ~ +1
        # cmd.yawSpeed = 0
        # cmd.bodyHeight = 0.1

        if (self.motiontime > 0 and self.motiontime < 1000):
            self.cmd.mode = 1
            self.cmd.rpy = [-0.3, 0, 0]

        if (self.motiontime > 1000 and self.motiontime < 2000):
            self.cmd.mode = 1
            self.cmd.rpy = [0.3, 0, 0]

        if (self.motiontime > 2000 and self.motiontime < 3000):
            self.cmd.mode = 1
            self.cmd.rpy = [0, -0.2, 0]

        if (self.motiontime > 3000 and self.motiontime < 4000):
            self.cmd.mode = 1
            self.cmd.rpy = [0, 0.2, 0]

        if (self.motiontime > 4000 and self.motiontime < 5000):
            self.cmd.mode = 1
            self.cmd.rpy = [0, 0, -0.2]

        if (self.motiontime > 5000 and self.motiontime < 6000):
            self.cmd.mode = 1
            self.cmd.rpy = [0.2, 0, 0]

        if (self.motiontime > 6000 and self.motiontime < 7000):
            self.cmd.mode = 1
            self.cmd.dBodyHeight = -0.2

        if (self.motiontime > 7000 and self.motiontime < 8000):
            self.cmd.mode = 1
            self.cmd.dBodyHeight = 0.1

        if (self.motiontime > 8000 and self.motiontime < 9000):
            self.cmd.mode = 1
            self.cmd.dBodyHeight = 0.0

        if (self.motiontime > 9000 and self.motiontime < 11000):
            self.cmd.mode = 5

        if (self.motiontime > 11000 and self.motiontime < 13000):
            self.cmd.mode = 6

        if (self.motiontime > 13000 and self.motiontime < 14000):
            self.cmd.mode = 0

        if (self.motiontime > 14000 and self.motiontime < 18000):
            self.cmd.mode = 2
            self.cmd.gaitType = 2
            self.cmd.velocity = [0.4, 0]  # -1  ~ +1
            self.cmd.yawSpeed = 2
            self.cmd.dFootRaiseHeight = 0.1
            # printf("walk\n")

        if (self.motiontime > 18000 and self.motiontime < 20000):
            self.cmd.mode = 0
            self.cmd.velocity = [0, 0]

        if (self.motiontime > 20000 and self.motiontime < 24000):
            self.cmd.mode = 2
            self.cmd.gaitType = 1
            self.cmd.velocity = [0.2, 0]  # -1  ~ +1
            self.cmd.dBodyHeight = 0.1
            # printf("walk\n")
        self.udp.SetSend(self.cmd)
        self.udp.Send()



if __name__ == '__main__':
    custom=Custom()
    print("Communication level is set to HIGH-level.")
    print("WARNING: Make sure the robot is standing on the ground.")
    print("Press Enter to continue...")
    input()

    for i in range(60):
        custom.ChangeLevel()
        custom.UDPSend()
        custom.UDPRecv()
        time.sleep(100*custom.dt)
        print(i)

    custom.motiontime=0

    while(True):
        custom.example_walk()