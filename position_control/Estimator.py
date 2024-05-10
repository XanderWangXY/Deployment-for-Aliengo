import torch


class InitEuler:
    def __init__(self, a_x, a_y, a_z, m_x, m_y, m_z):
        self.g = torch.tensor([0, 0, -9.81])
        self.a_imu = torch.Tensor([a_x, a_y, a_z])
        self.m_imu = torch.Tensor([m_x, m_y, m_z])

    def get_Euler(self):
        self.roll = torch.asin(self.a_imu[1] / torch.sqrt(self.a_imu[0] ** 2 + self.a_imu[1] ** 2 + self.a_imu[2] ** 2))
        self.pitch = torch.atan(2 * (-self.a_imu[0] / self.a_imu[2]))
        m_xn = self.m_imu[0] * torch.cos(self.pitch) + self.m_imu[2] * torch.sin(self.pitch)
        m_yn = (self.m_imu[0] * torch.sin(self.pitch) * torch.sin(self.roll)
                + self.m_imu[1] * torch.cos(self.roll)
                - self.m_imu[2] * torch.cos(self.pitch) * torch.sin(self.roll))
        self.yaw = torch.atan2(m_yn, m_xn)

    def euler_to_quat(self):
        cy = torch.cos(self.yaw * 0.5)
        sy = torch.sin(self.yaw * 0.5)
        cr = torch.cos(self.roll * 0.5)
        sr = torch.sin(self.roll * 0.5)
        cp = torch.cos(self.pitch * 0.5)
        sp = torch.sin(self.pitch * 0.5)

        qw = cy * cr * cp + sy * sr * sp
        qx = cy * sr * cp - sy * cr * sp
        qy = cy * cr * sp + sy * sr * cp
        qz = sy * cr * cp - cy * sr * sp

        self.quat = torch.stack([qx, qy, qz, qw], dim=-1)

    def get_rot_mat(self):
        self.rot_mat = torch.ones(3, 3)
        return self.rot_mat

    def get_a_world(self):
        self.a_world = self.rot_mat * self.a_imu + self.g
        return self.a_world


class KF:
    def __init__(self):
        self.A_c = torch.zeros(18, 18)
        self.A_c[0][3] = 1
        self.A_c[1][4] = 1
        self.A_c[2][5] = 1

        self.B_c = torch.zeros(18, 3)
        self.B_c[3][0] = 1
        self.B_c[4][1] = 1
        self.B_c[5][2] = 1

        eye3 = torch.eye(3, 3)
        zero3 = torch.zeros(3, 3)
        C0 = torch.cat((-eye3, zero3, eye3, zero3, zero3, zero3), dim=-1)
        C1 = torch.cat((-eye3, zero3, zero3, eye3, zero3, zero3), dim=-1)
        C2 = torch.cat((-eye3, zero3, zero3, zero3, eye3, zero3), dim=-1)
        C3 = torch.cat((-eye3, zero3, zero3, zero3, zero3, eye3), dim=-1)
        C4 = torch.cat((zero3, -eye3, zero3, zero3, zero3, zero3), dim=-1)
        C5 = torch.cat((zero3, -eye3, zero3, zero3, zero3, zero3), dim=-1)
        C6 = torch.cat((zero3, -eye3, zero3, zero3, zero3, zero3), dim=-1)
        C7 = torch.cat((zero3, -eye3, zero3, zero3, zero3, zero3), dim=-1)
        C8 = torch.zeros(4, 18)
        C8[0][8] = 1
        C8[1][11] = 1
        C8[2][14] = 1
        C8[3][17] = 1
        self.C = torch.cat((C0, C1, C2, C3, C4, C5, C6, C7, C8))

        self.p_b = torch.zeros(3, 1)  # 机身位置
        self.v_b = torch.zeros(3, 1)  # 机身速度
        self.p_0 = torch.zeros(3, 1)  # 足端位置
        self.p_1 = torch.zeros(3, 1)  # 足端位置
        self.p_2 = torch.zeros(3, 1)  # 足端位置
        self.p_3 = torch.zeros(3, 1)  # 足端位置

        self.x=torch.cat((self.p_b,self.v_b,self.p_0,self.p_1,self.p_2,self.p_3))

        self.Q = 0  # 过程噪声 w~(0,Q)
        self.R = 0  # 测量噪声 w~(0,R)

        self.dt = 0.002

    def con_to_dis(self):
        self.A = torch.eye(self.A_c.shape[0]) + self.dt * self.A_c
        self.B = self.dt * self.B_c

    def kalman_filter(self, x_0, P_0, u, y):
        x_est = self.A * x_0 + self.B * u
        P_est = self.A * P_0 * torch.transpose(self.A, dim0=0, dim1=1) + self.Q

        K = P_est * torch.transpose(self.C, dim0=0, dim1=1) * torch.inverse(
            self.C * P_est * torch.transpose(self.C, dim0=0, dim1=1) + self.R)
        x = x_est + K * (y - self.C * x_est)
        P = (torch.eye((K * self.C).shape[0]) - K * self.C) * P_est
        return x, P


kf = KF()
print(kf.C)
