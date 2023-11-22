#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yhq
@contact:1318231697@qq.com
@version: 1.0.0
@file: edit.py
@time: 2022/6/22 20:19
"""
import numpy as np
import torch
import torch.nn as nn
# from WGAN import Generator,return_index
class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 6, 1, 0),  # img: 21*21
            self._block(features_g * 2, channels_img, (10,1), 1, 0),  # img: 30*21
            # nn.ConvTranspose2d(
            #     features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            # ),
            # Output: N x channels_img x 30*21
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
def return_index(one_hot_coding):
    # one_hot = one_hot_coding.numpy()
    index = np.argwhere(one_hot_coding == 1)
    return index[:, -1].reshape(-1, 30)
def linear_interpolate(latent_code,
                       boundary,
                       start_distance=-3.0,
                       end_distance=3.0,
                       steps=10):
  """Manipulates the given latent code with respect to a particular boundary.
  Basically, this function takes a latent code and a boundary as inputs, and
  outputs a collection of manipulated latent codes. For example, let `steps` to
  be 10, then the input `latent_code` is with shape [1, latent_space_dim], input
  `boundary` is with shape [1, latent_space_dim] and unit norm, the output is
  with shape [10, latent_space_dim]. The first output latent code is
  `start_distance` away from the given `boundary`, while the last output latent
  code is `end_distance` away from the given `boundary`. Remaining latent codes
  are linearly interpolated.
  Input `latent_code` can also be with shape [1, num_layers, latent_space_dim]
  to support W+ space in Style GAN. In this case, all features in W+ space will
  be manipulated same as each other. Accordingly, the output will be with shape
  [10, num_layers, latent_space_dim].
  NOTE: Distance is sign sensitive.
  Args:
    latent_code: The input latent code for manipulation.
    boundary: The semantic boundary as reference.
    start_distance: The distance to the boundary where the manipulation starts.
      (default: -3.0)
    end_distance: The distance to the boundary where the manipulation ends.
      (default: 3.0)
    steps: Number of steps to move the latent code from start position to end
      position. (default: 10)
  """
  assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
          len(boundary.shape) == 2 and
          boundary.shape[1] == latent_code.shape[-1])

  linspace = np.linspace(start_distance, end_distance, steps)
  if len(latent_code.shape) == 2:
    linspace = linspace - latent_code.dot(boundary.T)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    return latent_code + linspace * boundary
  if len(latent_code.shape) == 3:
    linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
    return latent_code + linspace * boundary.reshape(1, 1, -1)
  raise ValueError(f'Input `latent_code` should be with shape '
                   f'[1, latent_space_dim] or [1, N, latent_space_dim] for '
                   f'W+ space in Style GAN!\n'
                   f'But {latent_code.shape} is received.')
def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    props=props.cpu().detach().numpy()
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return torch.FloatTensor(b)
print('loading...')
# laten_code = np.load("rand_noise2.npy")
# laten_code = np.empty_like(laten_code)
# laten_code.fill(500)
# laten_code2 = np.load("rand_noise2.npy")
# laten_code = (laten_code + laten_code2)
boundary = np.load('boundary_a.npy')
# laten_code = torch.zeros(1,100).numpy()
# np.save("rand_noise3.npy",laten_code)
laten_code = torch.randn(1, 100).numpy()
np.save("rand_noise3.npy",laten_code)
norm = np.linalg.norm(laten_code, axis=1, keepdims=True)
laten_code = laten_code / norm * np.sqrt(100)
# laten_code = laten_code / norm
# laten_code = np.ones_like(laten_code)

noise = linear_interpolate(laten_code,boundary)
# print(a.shape)
IMAGE_SIZE = (30,21)
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 50
FEATURES_CRITIC = 16
FEATURES_GEN = 16
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN)
gen.load_state_dict(torch.load('../../WGAN/gan_amp_49.pth', map_location={'cuda:1':'cpu'})['model'])
noise = torch.tensor(noise)
noise = noise.unsqueeze(dim = 2)
noise = noise.unsqueeze(dim = 2)
# label1_gen=np.empty((20,1))
# label1_gen.fill(1)
# # label2=label2.squeeze().squeeze()
# label1_gen=torch.LongTensor(label1_gen)
# label1_gen = label1_gen.unsqueeze(1)
# label2_gen = np.ones((10000,1))
# label2=label2.squeeze().squeeze()
# label2_gen=torch.LongTensor(label2_gen)
# label2_gen = label2_gen.unsqueeze(1).to(device)
print(noise.shape)
# print(label1_gen.shape)
fake = gen(noise)
# fake = gen(noise)
fake=fake.clone().detach()
fake=fake.squeeze(1)
for i in range(fake.shape[0]):
    fake[i]=props_to_onehot(fake[i])
fake=fake.cpu().detach().numpy()
np.save('gen_wgan_edit.npy', fake)
wgan_index = np.load('gen_wgan_edit.npy')
np.save('gen_wgan_edit.npy',return_index(wgan_index))