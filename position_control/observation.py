import time
import torch
import numpy as np

from position_control import Position
import robot_interface as sdk


class Observation:
    def __init__(self, position: Position):
        self.state = position.state
        self.d = position.d

        self.roll = torch.Tensor([[self.state.imu.rpy[0]]])
        self.pitch = torch.Tensor([[self.state.imu.rpy[1]]])
        self.yaw = torch.Tensor([[self.state.imu.rpy[2]]])

        self.gyroscope = torch.Tensor(self.state.imu.gyroscope)

    def get_obs_buf1(self):
        """
        [self.base_ang_vel  * self.obs_scales.ang_vel]
        躯体角速度          *  0.25（标准化系数）

        :return:
        obs_buf[0:3]
        """
        return self.gyroscope

    def get_imu_obs(self):
        """
        [roll,pitch]

        :return:
        obs_buf[3:5]
        """
        return torch.cat((self.roll, self.pitch), 1)

    def get_delta_yaw(self):
        """
        [0*delta_yaw,delta_yaw,delta_next_yaw]

        :return:
        obs_buf[5:8]
        """
        target_vec_norm = self.cur_goals[:, :2] - self.root_states[:, :2]
        target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        delta_yaw = target_yaw - self.yaw

        target_vec_norm = self.next_goals[:, :2] - self.root_states[:, :2]
        next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        delta_next_yaw = next_target_yaw - self.yaw

        return [0, delta_yaw, delta_next_yaw]

    def get_commands(self):
        """

        :return:
        obs_buf[8:11]
        """
        pass

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
            dof_vel.append(self.state.motorState[i].q * 0.05)
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

    def observation(self):
        print(self.state.imu)


if __name__ == '__main__':
    position = Position()
    while True:
        position.receive_state()
        time.sleep(0.002)
        observation = Observation(position)
        print(observation.get_priv_explicit())
