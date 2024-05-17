import time
import torch
import numpy as np
import sys, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from position_control.position_control import Position
from position_control.Estimator import KF
#from position_control import Position
#from Estimator import KF

#import robot_interface as sdk


#@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower


#@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


#@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


#@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


class Observation:
    def __init__(self, position: Position):
        self.state = position.state
        #self.d = position.d

        self.roll = self.state.imu.rpy[0]
        self.pitch = self.state.imu.rpy[1]
        self.yaw = self.state.imu.rpy[2]

        self.quaternion = self.state.imu.quaternion
        self.quaternion = torch.Tensor([self.quaternion])

        self.gyroscope = self.state.imu.gyroscope
        # 世界坐标系线速度
        #self.x_vel = vel[0]
        #self.y_vel = vel[1]
        #self.z_vel = vel[2]

        # 世界坐标系位置
        #self.x_pos = pos[0]
        #self.y_pos = pos[1]
        #self.z_pos = pos[2]


        self.goals = [[2, 5, 0],
                      [5, 2, 0],
                      [7, 3, 0],
                      [9, 1, 0]]  # 目标点
        self.cur_goals = torch.Tensor([self.goals[0]])
        self.next_goals = torch.Tensor([self.goals[0]])

        #self.device = torch.device('cpu')
        self.device = 'cpu'

        self.height_samples = torch.Tensor(np.loadtxt(BASE+'/position_control/data'))
        self.episode_length_buf = torch.zeros(1)
        self.obs_history_buf = torch.zeros(1, 10, 53)


    def get_pos(self, pos=torch.zeros(1, 3)):
        self.pos = pos

    def get_vel(self, vel=torch.zeros(1, 3)):
        self.vel = vel

    def get_obs_buf1(self):
        """
        [self.base_ang_vel  * self.obs_scales.ang_vel]
        躯体角速度          *  0.25（标准化系数）
        ###########世界坐标系角速度？

        :return:
        obs_buf[0:3]
        """
        ang_vel = 0.25
        return torch.Tensor([[self.gyroscope[0], self.gyroscope[1], self.gyroscope[2]]]) * ang_vel

    def get_imu_obs(self):
        """
        [roll,pitch]

        :return:
        obs_buf[3:5]
        """
        roll = torch.Tensor([[self.roll]])
        pitch = torch.Tensor([[self.pitch]])

        return torch.cat((roll, pitch), dim=-1)

    def get_delta_yaw(self):
        """
        [0*delta_yaw,delta_yaw,delta_next_yaw]

        :return:
        obs_buf[5:8]
        """
        #pos = torch.Tensor([[self.x_pos, self.y_pos, self.z_pos]])  #狗的位置

        target_pos_rel = self.cur_goals[:, :2] - self.pos[:, :2]
        norm = torch.norm(target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = target_pos_rel / (norm + 1e-5)
        target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        delta_yaw = target_yaw - self.yaw

        next_target_pos_rel = self.next_goals[:, :2] - self.pos[:, :2]
        norm = torch.norm(next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = next_target_pos_rel / (norm + 1e-5)
        next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        delta_next_yaw = next_target_yaw - self.yaw

        return torch.Tensor([[0 * delta_yaw, delta_yaw, delta_next_yaw]])

    def get_commands(self):
        """
        [0*commands[0:2],commands[0:1]]
        commands[4]
        commands[lin_vel_x, lin_vel_y, ang_vel_yaw, heading]

        :return:
        obs_buf[8:11]
        """
        heading = 0

        commands = torch.cat((self.vel[0][0].unsqueeze(-1),
                              self.vel[0][1].unsqueeze(-1),
                              torch.Tensor([self.gyroscope[2]]),
                              torch.Tensor([heading])), dim=-1)
        commands3 = torch.cat(((0 * commands[0]).unsqueeze(-1),
                               (0 * commands[1]).unsqueeze(-1),
                               commands[0].unsqueeze(-1)), dim=-1)
        return commands3.unsqueeze(0)

    def get_dof_pos(self):
        """
        self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos)
        ###############要看一下具体值，default_dof_pos_all需要设置
        :return:
        obs_buf[13:25]
        """
        dof_pos = []
        for i in range(12):
            dof_pos.append(self.state.motorState[i].q)
        return torch.Tensor([dof_pos])

    def get_dof_vel(self):
        """

        :return:
        obs_buf[25:37]
        """
        dof_vel = []
        for i in range(12):
            dof_vel.append(self.state.motorState[i].dq * 0.05)
        return torch.Tensor([dof_vel])

    def get_action_history_buf(self, action_history=torch.zeros(1,12)):
        """

        :return:
        obs_buf[37:49]
        """
        self.action_history=action_history
        return self.action_history

    def get_contact_filt(self):
        """
        是否着地
        :return:
        obs_buf[49:53]
        """
        contact_filt = [0, 0, 0, 0]
        for i in range(4):
            if self.state.footForce[i] <= 5:
                contact_filt[i] = 0 - 0.5
            else:
                contact_filt[i] = 1 - 0.5
        return torch.Tensor([contact_filt])

    def get_obs_buf(self):
        self.obs_buf = torch.cat((self.get_obs_buf1(),
                                  self.get_imu_obs(),
                                  self.get_delta_yaw(),
                                  self.get_commands(),
                                  torch.Tensor([[0, 1]]),
                                  self.get_dof_pos(),
                                  self.get_dof_vel(),
                                  self.get_action_history_buf(),
                                  self.get_contact_filt()
                                  ), dim=-1)

    def get_obs_history_buf(self):
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.obs_buf] * 10, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                self.obs_buf.unsqueeze(1)
            ], dim=1)
        )

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
        return torch.t(priv_latent.unsqueeze(-1))

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05,
                             1.2]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        measure_horizontal_noise = 0

        y = torch.tensor(measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        num_height_points = grid_x.numel()
        points = torch.zeros(1, num_height_points, 3, device=self.device, requires_grad=False)
        for i in range(1):
            offset = torch_rand_float(-measure_horizontal_noise, measure_horizontal_noise, (num_height_points, 2),
                                      device=self.device).squeeze()
            xy_noise = torch_rand_float(-measure_horizontal_noise, measure_horizontal_noise, (num_height_points, 2),
                                        device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
        return points

    def get_heights(self):
        border_size = 5
        vertical_scale = 0.005
        horizontal_scale = 0.05
        height_points = self._init_height_points()

        points = quat_apply_yaw(self.quaternion.repeat(1, 132), height_points) + (
            self.pos[:, :3]).unsqueeze(1)
        points += border_size
        points = (points / horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(1, -1) * vertical_scale

    def get_observation(self):
        self.get_obs_buf()
        self.get_obs_history_buf()
        self.observation = torch.cat((self.obs_buf,
                                      self.get_heights(),
                                      self.get_priv_explicit(),
                                      self.get_priv_latent(),
                                      self.obs_history_buf.view(1, -1)
                                      ), dim=-1)


if __name__ == '__main__':
    position = Position()
    kf = KF()
    observation = Observation(position)

    for i in range(50):
        position.pos_init()
    while True:
        position.receive_state()
        time.sleep(0.002)
        imu_rpy = position.state.imu.rpy
        imu_acc = position.state.imu.accelerometer
        imu_gyo = position.state.imu.gyroscope
        imu_quat = position.state.imu.quaternion

        kf.run_KF(imu_acc, imu_gyo, imu_rpy)

        #observation.get_imu(imu_rpy, imu_gyo, imu_quat)
        observation.get_pos(kf.x[:3].reshape(1,3))
        observation.get_vel(kf.x[3:6].reshape(1,3))

        observation.episode_length_buf = observation.episode_length_buf+1

        observation.get_observation()
        print(observation.observation.shape)
        time.sleep(0.5)