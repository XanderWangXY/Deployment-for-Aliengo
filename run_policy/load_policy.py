import glob
import torch
import torch.nn as nn


class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone

        self.combination_mlp = nn.Sequential(
                nn.Linear(32 + 53, 128),
                activation,
                nn.Linear(128, 32)
        )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
            nn.Linear(512, 32 + 2),
            last_activation
        )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        # depth_latent = self.base_backbone(depth_image)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))

        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()


class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, scandots_output_dim=32, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent


class DepthImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_frames = 1
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, 32)
        )



if __name__ == '__main__':
    depth_conv = DepthOnlyFCBackbone58x87()
    depth_rec = RecurrentDepthBackbone(depth_conv)

    dirs = glob.glob(f"../policy/*")
    logdir = sorted(dirs)
    print(logdir)
    policy = torch.jit.load(logdir[0])
    vision_weight = torch.load(logdir[1])
    vision_weight = vision_weight['depth_encoder_state_dict']
    depth_rec.load_state_dict(vision_weight)
    image = torch.ones(1, 58, 87)
    proprioception = torch.ones(1, 53)
    test_depth=depth_rec(image, proprioception)

############
    device = torch.device('cpu')
    n_priv_explicit = 3 + 3 + 3  # self.base_lin_vel * self.obs_scales.lin_vel+0 * self.base_lin_vel+0 * self.base_lin_vel
    n_priv_latent = 4 + 1 + 12 + 12  # priv_latent=mass_params+friction_coeffs+motor_strength[0] - 1+motor_strength[1] - 1
    num_scan = 132  # height
    num_actions = 12  # motor position

    # depth_buffer_len = 2
    depth_resized = (87, 58)
    n_proprio = 3 + 2 + 3 + 4 + 36 + 4 + 1  # history buf
    history_len = 10
    num_envs = 1


    obs_input = torch.ones(num_envs, n_proprio + num_scan + n_priv_explicit + n_priv_latent + history_len * n_proprio,
                           device=device)
    depth_latent = torch.ones(1, 32, device=device)

    test_policy = policy(obs_input, depth_latent)

    print(test_depth)
    print(test_policy)

