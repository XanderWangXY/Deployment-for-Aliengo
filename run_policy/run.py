import cv2
import torch
import time
import sys, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from realsense.realsense import RealSense
from load_policy import depth_rec, policy
from position_control.position_control import Position
from position_control.observation import Observation
from position_control.Estimator import KF

if __name__ == '__main__':
    device = torch.device('cpu')
    #realsense = RealSense()
    position = Position()
    kf = KF()
    observation = Observation(position)
    for i in range(50):
        position.pos_init()
    try:
        while True:
            # 1.处理图像
            #realsense.get_depth()
            #depth_image = torch.Tensor(realsense.depth_image_clip).unsqueeze(0)
            depth_image = torch.ones(1, 58, 87)

            # 2.体感数据
            position.receive_state()
            time.sleep(0.002)
            imu_rpy = position.state.imu.rpy
            imu_acc = position.state.imu.accelerometer
            imu_gyo = position.state.imu.gyroscope
            imu_quat = position.state.imu.quaternion

            kf.run_KF(imu_acc, imu_gyo, imu_rpy)

            observation.get_pos(kf.x[:3].reshape(1, 3))
            observation.get_vel(kf.x[3:6].reshape(1, 3))
            observation.episode_length_buf = observation.episode_length_buf + 1
            observation.get_observation()

            # 3.运行策略
            depth_latent = depth_rec(depth_image, observation.obs_buf)
            depth_latent = depth_latent[:,:32]
            obs_input = observation.observation.to(device=device)
            actions=policy(obs_input, depth_latent)
            print(actions)
            observation.get_action_history_buf(actions)

            # 4.执行动作
            send_actions = actions.tolist()
            position.send_pos(send_actions[0])

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    #except:
        #realsense.stop()
    finally:
        pass
