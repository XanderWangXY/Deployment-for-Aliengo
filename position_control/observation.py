import time
import torch
import numpy as np

from position_control import Position
import robot_interface as sdk


class Observation:
    def __init__(self, position: Position):
        self.state = position.state
        self.d = position.d

        self.roll = self.state.imu.rpy[0]
        self.pitch = self.state.imu.rpy[1]
        self.yaw = self.state.imu.rpy[2]

        self.gyroscope = self.state.imu.gyroscope

        self.x_vel = 0
        self.y_vel = 0
        self.z_vel = 0

        self.x_pos=0
        self.y_pos = 0
        self.z_pos = 0

        self.goals=[[2,5,0],
                   [5,2,0],
                   [7,3,0],
                   [9,1,0]] # 目标点
        self.cur_goals=torch.Tensor(self.goals[0])
        self.next_goals = torch.Tensor(self.goals[0])

    def get_obs_buf1(self):
        """
        [self.base_ang_vel  * self.obs_scales.ang_vel]
        躯体角速度          *  0.25（标准化系数）

        :return:
        obs_buf[0:3]
        """
        ang_vel = 0.25
        return torch.Tensor(self.gyroscope * ang_vel)

    def get_imu_obs(self):
        """
        [roll,pitch]

        :return:
        obs_buf[3:5]
        """
        roll = torch.Tensor([[self.roll]])
        pitch = torch.Tensor([[self.pitch]])

        return torch.cat((roll, pitch), 1)

    def get_delta_yaw(self):
        """
        [0*delta_yaw,delta_yaw,delta_next_yaw]

        :return:
        obs_buf[5:8]
        """
        pos=torch.Tensor([self.x_pos,self.y_pos,self.z_pos])

        target_pos_rel = self.cur_goals[:2] - pos[:2]
        norm = torch.norm(target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = target_pos_rel / (norm + 1e-5)
        target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        delta_yaw = target_yaw - self.yaw

        next_target_pos_rel = self.next_goals[:2] - pos[:2]
        norm = torch.norm(next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = next_target_pos_rel / (norm + 1e-5)
        next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        delta_next_yaw = next_target_yaw - self.yaw

        return [0, delta_yaw, delta_next_yaw]

    def get_commands(self):
        """
        [0*commands[0:2],commands[0:1]]
        commands[4]
        commands[x_vel, y_vel, yaw_vel, heading]

        :return:
        obs_buf[8:11]
        """
        heading = 0
        commands = [self.x_vel, self.y_vel, self.gyroscope[2], heading]
        commands3 = [0 * commands[0], 0 * commands[1], commands[1]]
        return torch.Tensor(commands3)

    def get_dof_pos(self):
        """
        self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos)

        :return:
        obs_buf[13:25]
        """
        dof_pos = []
        for i in range(12):
            dof_pos.append(self.state.motorState[i].q)
        return torch.Tensor(dof_pos)

    def get_dof_vel(self):
        """

        :return:
        obs_buf[25:37]
        """
        dof_vel = []
        for i in range(12):
            dof_vel.append(self.state.motorState[i].dq * 0.05)
        return torch.Tensor(dof_vel)

    def get_action_history_buf(self):
        """

        :return:
        obs_buf[37:49]
        """
        pass

    def get_contact_filt(self):
        """
        是否着地
        :return:
        obs_buf[49:53]
        """
        contact_filt = [0, 0, 0, 0]
        for i in range(4):
            if (self.state.footForce[i] <= 5):
                contact_filt[i] = 0 - 0.5
            else:
                contact_filt[i] = 1 - 0.5
        return torch.Tensor(contact_filt)

    def get_priv_explicit(self):
        lin_vel = 2.0
        base_lin_vel = [0.0, 0.0, 0.0]

        lin_vel = torch.Tensor([[lin_vel]])
        base_lin_vel = torch.Tensor(base_lin_vel)
        priv_explicit = torch.cat((base_lin_vel * lin_vel,
                                   (0.0 * base_lin_vel).unsqueeze(0),
                                   (0.0 * base_lin_vel).unsqueeze(0)), dim=-1)
        return priv_explicit

    def get_priv_latent(self):
        """
        mass:?
        friction_coeffs:0.6~2.0


        :return:
        """
        mass_params = [0, 0, 0, 0]
        friction_coeffs = [1.2]
        motor_strength = torch.ones(2, 12)

        priv_latent = torch.cat((torch.Tensor(mass_params),
                                 torch.Tensor(friction_coeffs),
                                 motor_strength[0] - 1,
                                 motor_strength[1] - 1), dim=-1)
        return priv_latent

    def observation(self):
        print(self.state.imu)


if __name__ == '__main__':
    position = Position()
    while True:
        position.receive_state()
        time.sleep(0.002)
        observation = Observation(position)
        print(observation.get_priv_latent())
