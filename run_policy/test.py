from load_policy import depth_rec, policy
import torch

image = torch.ones(1, 58, 87)
proprioception = torch.ones(1, 53)
depth_latent = torch.ones(1, 32)
obs_input = torch.ones(1, 753)

test_depth=depth_rec(image, proprioception)
depth_latent = torch.ones(1, 32)
test_policy = policy(obs_input, depth_latent)

print(test_depth.shape)
print(test_policy.shape)