#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getProjectionMatrix, focal2fov
import time
import numpy as np


class View(nn.Module):
    def __init__(self, R, T, focal_x, focal_y, width, height, data_device = "cuda" ):
        super(View, self).__init__()


        self.R = R
        self.T = T
        self.focal_y = focal_x
        self.focal_x = focal_y
        self.image_width = width
        self.image_height = height
        self.FoVx = focal2fov(focal_x,width)
        self.FoVy = focal2fov(focal_y,height)

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.zfar = 100.0
        self.znear = 0.01
        
        self.getWorldView(R,T)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def getWorldView(self, R, T):
        self.world_view_transform = np.zeros((4, 4))
        self.world_view_transform[3, 3] = 1.0
        t_inv = np.matmul(-R.transpose(),T)
        self.world_view_transform[:3,:3]=R.transpose()
        self.world_view_transform[:3,3] = t_inv
        self.world_view_transform = torch.tensor(np.float32(self.world_view_transform)).transpose(0, 1).cuda()
        

def render_view(camera_model, rotation, translation, gaussians, pipeline, background, kernel_size, idx):
    view = View(rotation, translation, camera_model["fl_x"], camera_model["fl_y"], camera_model["w"], camera_model["h"])
    rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
    rendering = rendering[:3, :, :]
    # image_path=os.path.join("/workspace/gaussian-opacity-fields/outputs", '{0:05d}'.format(idx) + ".png")
    # torchvision.utils.save_image(rendering, image_path)
    # os.chmod(image_path, 0o777)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    camera_model = { "fl_x": 914.7086181640625,
	"fl_y": 912.5759887695312,
	"w": 1280,
	"h": 720}

    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    #Load model
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, load_cameras=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size
    R = np.array([[-0.9439490687833039, -0.012658517779477973, 0.32984832494763394], [-0.32913601383730196, -0.039867014542148096, -0.9434405681052662], [0.025092627172631866, -0.9991248085595327, 0.03346605716920529]])
    T = np.array([-12.574966256566968, 38.94981550209616, -0.06905938699614564])
    start=time.clock_gettime_ns(time.CLOCK_THREAD_CPUTIME_ID)
    for _ in range(30):
        render_view(camera_model, R, T, gaussians,pipeline, background, kernel_size, 1)
    end=time.clock_gettime_ns(time.CLOCK_THREAD_CPUTIME_ID)
    print("Mean rendering time (ms): {} FPS: {}".format((end-start)*10**-6/30,(1/((end-start)*10**-9)*30)))