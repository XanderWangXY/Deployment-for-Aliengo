import glob
import torch

def load_policy():
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
n_priv_explicit = 3 + 3 + 3             # self.base_lin_vel * self.obs_scales.lin_vel+0 * self.base_lin_vel+0 * self.base_lin_vel
n_priv_latent = 4 + 1 + 12 + 12         # priv_latent=mass_params+friction_coeffs+motor_strength[0] - 1+motor_strength[1] - 1
num_scan = 132                          # height
num_actions = 12                        # motor position

# depth_buffer_len = 2
depth_resized = (87, 58)
n_proprio = 3 + 2 + 3 + 4 + 36 + 4 + 1  # history buf
history_len = 10
num_envs = 1

dirs = glob.glob(f"../policy/*")
logdir = sorted(dirs)
print(logdir)
model=torch.jit.load(logdir[0])
#vision_weight=torch.jit.load(logdir[1])

obs_input = torch.ones(num_envs, n_proprio + num_scan + n_priv_explicit + n_priv_latent + history_len*n_proprio, device=device)
depth_latent = torch.ones(1, 32, device=device)

test = model(obs_input, depth_latent)
print(test)
#print(model.eval())