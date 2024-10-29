# general
import argparse
import pathlib
import time
import datetime
import copy
import imageio
import tqdm
import pdb
import curses
import sys
import os
import random
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

# matplotlib.use("Agg")
import matplotlib
from skimage import color, io

from habitat_sim.utils.common import d3_40_colors_rgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors


import torch
import torch.nn.functional as F
from lpips import LPIPS
import cv2

sys.path.append("perception/nerfacc")
from nerfacc.estimators.occ_grid import OccGridEstimator

# nerfacc
sys.path.append("perception/models")
from datasets.utils import Rays
from utils import (
    render_image_with_occgrid,
    render_image_with_occgrid_test,
    render_image_with_occgrid_with_depth_guide,
    render_probablistic_image_with_occgrid_test,
)
from radiance_fields.ngp import NGPRadianceField

# rotorpy
sys.path.append("planning/rotorpy")
sys.path.append("planning")
from planning_funcs import *
from online_gd import online_gd, online_gd_changing_lr


# habitat simulator
sys.path.append("simulator")
from sim import HabitatSim

# data processing
sys.path.append("perception/data_proc")
from habitat_to_data import Dataset

import yaml
import pickle

import seaborn as sns

sns.set_theme()
fsz = 32
plt.rc("font", size=fsz)
plt.rc("axes", titlesize=fsz)
plt.rc("axes", labelsize=fsz)
plt.rc("xtick", labelsize=fsz)
plt.rc("ytick", labelsize=fsz)
plt.rc("legend", fontsize=fsz)
plt.rc("figure", titlesize=fsz)
plt.rc("pdf", fonttype=42)
plt.rc("lines", linewidth=4)
sns.set_style("ticks", rc={"axes.grid": True})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sem-num",
        type=int,
        default=0,
        help="number of semantics classes",
    )
    parser.add_argument(
        "--habitat-scene",
        type=str,
        default="102344250",
        help="habitat scene",
    )
    parser.add_argument(
        "--habitat-config-file",
        type=str,
        default=str(
            pathlib.Path.cwd()
            / "data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"
        ),
        help="scene_dataset_self.config_file",
    )
    parser.add_argument(
        "--ogd-change-lr",
        type=bool,
        default=True,
        help="online gradient descent with changing learning rate",
    )
    parser.add_argument(
        "--online-gd",
        type=bool,
        default=False,
        help="online gradient descent for improving info gain prediction",
    )
    return parser.parse_args()


class ActiveNeRFMapper:
    def __init__(self, args) -> None:
        print("Parameters Loading")
        # initialize radiance field, estimator, optimzer, and dataset
        with open(f"scripts/config_" + args.habitat_scene + ".yaml", "r") as f:
            self.config_file = yaml.safe_load(f)

        self.save_path = (
            self.config_file["save_path"]
            + "/"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        self.learning_rate_lst = []

        # scene parameters
        self.aabb = torch.tensor(
            self.config_file["aabb"], device=self.config_file["cuda"]
        )

        grid_size_scele = [1, 1, 1.5, 1.5] * self.config_file["n_ensembles"]

        neuron_scale = [1, 0.25, 1, 0.5] * self.config_file["n_ensembles"]
        layer_scale = [1, 1.5, 1, 1.5] * self.config_file["n_ensembles"]
        sem_neuron_scale = [1, 2, 2, 2] * self.config_file["n_ensembles"]

        # model parameters
        self.main_grid_resolution = [
            (
                (
                    (self.aabb.cpu().numpy()[3:] - self.aabb.cpu().numpy()[:3])
                    / (self.config_file["main_grid_size"] * gs)
                )
                .astype(int)
                .tolist()
            )
            for gs in grid_size_scele
        ]

        self.cost_map = np.full(
            (self.main_grid_resolution[0][0], self.main_grid_resolution[0][2]), 0.5
        )
        self.visiting_map = np.zeros(self.cost_map.shape)

        self.minor_grid_resolution = (
            (
                (self.aabb.cpu().numpy()[3:] - self.aabb.cpu().numpy()[:3])
                / self.config_file["minor_grid_size"]
            )
            .astype(int)
            .tolist()
        )

        self.trajector_uncertainty_list = [
            [] for _ in range(self.config_file["planning_step"])
        ]

        self.policy_type = "uncertainty"  # "uncertainty", "random", "spatial"

        if self.policy_type == "random":
            self.config_file["num_traj"] = 1

        self.estimators = []
        self.radiance_fields = []
        self.optimizers = []
        self.grad_scalers = []
        self.schedulers = []
        self.binary_grid = None
        self.train_dataset = None
        self.test_dataset = None
        self.errors_hist = []

        self.sem_ce_ls = []

        self.sim_step = 0
        self.viz_save_path = self.save_path + "/viz/"

        for i in range(self.config_file["n_ensembles"]):
            estimator = OccGridEstimator(
                roi_aabb=self.aabb,
                resolution=self.main_grid_resolution[i],
                levels=self.config_file["main_grid_nlvl"],
            ).to(self.config_file["cuda"])

            radiance_field = NGPRadianceField(
                aabb=estimator.aabbs[-1],
                neurons=int(self.config_file["main_neurons"] * neuron_scale[i]),
                layers=int(self.config_file["main_layer"] * layer_scale[i]),
                num_semantic_classes=args.sem_num,
                sem_neuron_scale=sem_neuron_scale[i],
            ).to(self.config_file["cuda"])

            optimizer = torch.optim.Adam(
                radiance_field.parameters(),
                lr=1e-3,
                eps=1e-8,
                weight_decay=self.config_file["weight_decay"],
            )

            self.estimators.append(estimator)
            self.grad_scalers.append(torch.cuda.amp.GradScaler(2**10))
            self.radiance_fields.append(radiance_field)
            self.optimizers.append(optimizer)
            self.schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.CyclicLR(
                            optimizer,
                            base_lr=1e-4,
                            max_lr=1e-3,
                            step_size_up=15,  # int(self.config_file["training_steps"] / 4),
                            mode="exp_range",
                            gamma=1.0,  # 0.9999,
                            cycle_momentum=False,
                        )
                    ]
                )
            )

        self.lpips_net = LPIPS(net="vgg").to(self.config_file["cuda"])
        self.lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1

        self.focal = (
            0.5 * self.config_file["img_w"] / np.tan(self.config_file["hfov"] / 2)
        )

        cmap = plt.cm.tab20
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap1 = plt.cm.tab20b
        cmaplist1 = [cmap1(i) for i in range(cmap1.N)]

        cmaplist = (
            cmaplist
            + [cmaplist1[0]]
            + [cmaplist1[1]]
            + [cmaplist1[4]]
            + [cmaplist1[5]]
            + [cmaplist1[8]]
            + [cmaplist1[9]]
            + [cmaplist1[12]]
            + [cmaplist1[13]]
            + [cmaplist1[16]]
            + [cmaplist1[17]]
        )
        self.custom_cmap = matplotlib.colors.ListedColormap(cmaplist)

        r = np.arctan(np.linspace(0.5, 319.5, 320) / 320).tolist()
        r.reverse()
        l = np.arctan(-np.linspace(0.5, 319.5, 320) / 320).tolist()
        self.align_angles = np.array(r + l)

        self.global_origin = np.array(self.config_file["global_origin"])

        self.current_pose = np.array(self.config_file["global_origin"])

        self.sim = HabitatSim(
            args.habitat_scene,
            args.habitat_config_file,
            img_w=self.config_file["img_w"],
            img_h=self.config_file["img_h"],
        )

        self.online_gd_clr = online_gd_changing_lr()
        self.online_gd = online_gd()

        print("Parameters Loaded")

    def initialization(self):
        print("initialization Started")
        sampled_poses_quat = []
        sampled_poses_mat = []
        r = R.from_quat(self.global_origin[3:])
        g_pose = self.global_origin.copy()
        initial_sample = 39
        for i in range(initial_sample):
            angles = r.as_euler("zyx", degrees=True)
            angles[1] = (angles[1] + 9 * i) % 360
            pose = g_pose.copy()
            pose[3:] = R.from_euler("zyx", angles, degrees=True).as_quat()

            pose[:3] = pose[:3] + np.random.uniform([-0.2, -0.2, -0.2], [0.2, 0.2, 0.2])
            sampled_poses_quat.append(pose)

            T = np.eye(4)
            T[:3, :3] = R.from_quat(pose[3:]).as_matrix()
            T[:3, 3] = pose[:3]
            sampled_poses_mat.append(T)

        (
            sampled_images,
            sampled_depth_images,
            sampled_sem_images,
        ) = self.sim.sample_images_from_poses(sampled_poses_quat)

        for i, d_img in enumerate(sampled_depth_images):
            d_points = d_img[int(d_img.shape[0] / 2)]
            # st()
            R_m = sampled_poses_mat[i][:3, :3]
            euler = R.from_matrix(R_m).as_euler("yzx")
            d_angles = (self.align_angles + euler[0]) % (2 * np.pi)
            w_loc = sampled_poses_mat[i][:3, 3]
            grid_loc = np.array(
                (w_loc - self.aabb.cpu().numpy()[:3])
                // self.config_file["main_grid_size"],
                dtype=int,
            )
            self.cost_map, visiting_map = update_cost_map(
                cost_map=self.cost_map,
                depth=d_points,
                angle=d_angles,
                g_loc=grid_loc,
                w_loc=w_loc,
                aabb=self.aabb.cpu().numpy(),
                resolution=self.config_file["main_grid_size"],
            )
            self.visiting_map += visiting_map

        sampled_images = sampled_images[:, :, :, :3]

        sampled_poses_mat = np.array(sampled_poses_mat)

        self.train_dataset = Dataset(
            training=True,
            save_fp=self.save_path + "/train/",
            num_rays=self.config_file["init_batch_size"],
            num_models=self.config_file["n_ensembles"],
            device=self.config_file["cuda"],
        )

        self.train_dataset.update_data(
            sampled_images,
            sampled_depth_images,
            sampled_sem_images,
            sampled_poses_mat,
        )

        test_loc = self.config_file["test_loc"]

        test_quat = self.config_file["test_quat"]

        test_samples = []

        for loc in test_loc:
            for quat in test_quat:
                test_samples.append(np.array(loc + quat))

        test_sampled_poses_mat = []
        for p in test_samples:
            T = np.eye(4)
            T[:3, :3] = R.from_quat(p[3:]).as_matrix()
            T[:3, 3] = p[:3]
            test_sampled_poses_mat.append(T)

        (
            test_sampled_images,
            test_sampled_depth_images,
            test_sampled_sem_images,
        ) = self.sim.sample_images_from_poses(test_samples)

        test_sampled_images = test_sampled_images[:, :, :, :3]

        self.test_dataset = Dataset(
            training=False,
            save_fp=self.save_path + "/test/",
            num_models=self.config_file["n_ensembles"],
            device=self.config_file["cuda"],
        )

        self.test_dataset.update_data(
            test_sampled_images,
            test_sampled_depth_images,
            test_sampled_sem_images,
            np.array(test_sampled_poses_mat),
        )

        self.sem_ce_ls = [[] for _ in range(self.test_dataset.size)]
        self.psnrs_lst = [[] for _ in range(self.test_dataset.size)]
        self.lpips_lst = [[] for _ in range(self.test_dataset.size)]
        self.mse_dep_lst = [[] for _ in range(self.test_dataset.size)]

        print("Initialization Finished")

    def training_single_image(self, image_idx, steps=30):
        def occ_eval_fn(x):
            density = radiance_field.query_density(x)
            return density * self.config_file["render_step_size"]

        curr_device = self.config_file["cuda"]

        # st()

        for step in range(steps):
            # train and record the models in the ensemble

            # training each model
            for model_idx, (
                radiance_field,
                estimator,
                optimizer,
                scheduler,
                grad_scaler,
            ) in enumerate(
                zip(
                    self.radiance_fields,
                    self.estimators,
                    self.optimizers,
                    self.schedulers,
                    self.grad_scalers,
                )
            ):

                radiance_field.train()
                estimator.train()

                if step % 2 == 0:
                    i = image_idx
                else:
                    curr_idx = self.train_dataset.bootstrap(0)
                    curr_idx = curr_idx[:image_idx]
                    i = np.random.choice(curr_idx, 1).item()

                data = self.train_dataset[i]
                render_bkgd = data["color_bkgd"].to(curr_device)
                ry = data["rays"]
                rays = Rays(
                    origins=ry.origins.to(curr_device),
                    viewdirs=ry.viewdirs.to(curr_device),
                )
                pixels = data["pixels"].to(curr_device)
                dep = data["dep"].to(curr_device)
                sem = data["sem"].to(curr_device)

                # update occupancy grid
                estimator.update_every_n_steps(
                    step=step,
                    occ_eval_fn=occ_eval_fn,
                    occ_thre=1e-2,
                )

                (
                    rgb,
                    acc,
                    depth,
                    semantic,
                    n_rendering_samples,
                ) = render_image_with_occgrid_with_depth_guide(
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=self.config_file["near_plane"],
                    render_step_size=self.config_file["render_step_size"],
                    render_bkgd=render_bkgd,
                    cone_angle=self.config_file["cone_angle"],
                    alpha_thre=self.config_file["alpha_thre"],
                    depth=dep,
                )

                if n_rendering_samples == 0:
                    continue

                if self.config_file["target_sample_batch_size"] > 0:
                    # dynamic batch size for rays to keep sample batch size constant.
                    num_rays = len(pixels)
                    num_rays = int(
                        num_rays
                        * (
                            self.config_file["target_sample_batch_size"]
                            / float(n_rendering_samples)
                        )
                    )
                    self.train_dataset.update_num_rays(min(4000, num_rays))

                # compute loss
                loss_rgb = F.smooth_l1_loss(rgb, pixels)
                loss_dep = F.smooth_l1_loss(depth, dep.unsqueeze(1))
                loss_sem = F.cross_entropy(semantic, sem)

                loss = loss_rgb * 10 + loss_dep / 5 + loss_sem / 2

                optimizer.zero_grad()
                loss.backward()

                flag = False
                for name, param in radiance_field.named_parameters():
                    if torch.sum(torch.isnan(param.grad)) > 0:
                        flag = True
                        break

                if flag:
                    optimizer.zero_grad()
                    print(f"step jumped {model_idx}")
                    continue
                else:
                    optimizer.step()
                    scheduler.step()

    def nerf_training(
        self, steps, final_train=False, initial_train=False, planning_step=-1
    ):
        print("Nerf Training Started")

        # if final_train:
        #     self.schedulers = []
        #     for i in range(self.config_file["n_ensembles"]):
        #         optimizer = self.optimizers[i]
        #         self.schedulers.append(
        #             torch.optim.lr_scheduler.MultiStepLR(
        #                 optimizer,
        #                 milestones=[int(steps * 0.3), int(steps * 0.8)],
        #                 gamma=0.1,
        #             )
        #         )

        num_test_images = self.test_dataset.size
        test_idx = np.arange(num_test_images)

        def occ_eval_fn(x):
            density = radiance_field.query_density(x)
            return density * self.config_file["render_step_size"]

        losses = [[], [], []]

        # invisible_idx = self.estimators[1].occs < 0
        # self.estimators[1].occs[invisible_idx] = 1e-2
        # self.estimators[1].mark_invisible_cells(
        #     self.train_dataset.K[None, :, :],
        #     self.train_dataset.camtoworlds,
        #     self.train_dataset.width,
        #     self.train_dataset.height,
        #     depth=self.train_dataset.depths,
        # )

        for step in tqdm.tqdm(range(steps)):
            # train and record the models in the ensemble
            ground_truth_imgs = []
            rendered_imgs = [[] for _ in range(num_test_images)]

            ground_truth_depth = []
            depth_imgs = [[] for _ in range(num_test_images)]

            ground_truth_sem = []
            sem_imgs = []

            # training each model
            for model_idx, (
                radiance_field,
                estimator,
                optimizer,
                scheduler,
                grad_scaler,
            ) in enumerate(
                zip(
                    self.radiance_fields,
                    self.estimators,
                    self.optimizers,
                    self.schedulers,
                    self.grad_scalers,
                )
            ):
                curr_device = (
                    self.config_file["cuda"]
                    if model_idx == 0
                    else self.config_file["cuda"]
                )
                radiance_field.train()
                estimator.train()

                c = np.random.random_sample()

                if c < 0.1 and not final_train and not initial_train:
                    # train with most recent batch of data
                    curr_idx = self.train_dataset.bootstrap(0)
                    curr_idx = curr_idx[-40:]
                    i = np.random.choice(curr_idx, 1).item()
                else:
                    curr_idx = self.train_dataset.bootstrap(0)
                    i = np.random.choice(curr_idx, 1).item()

                data = self.train_dataset[i]
                render_bkgd = data["color_bkgd"].to(curr_device)
                ry = data["rays"]
                rays = Rays(
                    origins=ry.origins.to(curr_device),
                    viewdirs=ry.viewdirs.to(curr_device),
                )
                pixels = data["pixels"].to(curr_device)
                dep = data["dep"].to(curr_device)
                sem = data["sem"].to(curr_device)

                # update occupancy grid
                estimator.update_every_n_steps(
                    step=step,
                    occ_eval_fn=occ_eval_fn,
                    occ_thre=1e-2,
                )

                (
                    rgb,
                    acc,
                    depth,
                    semantic,
                    n_rendering_samples,
                ) = render_image_with_occgrid_with_depth_guide(
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=self.config_file["near_plane"],
                    render_step_size=self.config_file["render_step_size"],
                    render_bkgd=render_bkgd,
                    cone_angle=self.config_file["cone_angle"],
                    alpha_thre=self.config_file["alpha_thre"],
                    depth=dep,
                )

                if n_rendering_samples == 0:
                    continue

                if self.config_file["target_sample_batch_size"] > 0:
                    # dynamic batch size for rays to keep sample batch size constant.
                    num_rays = len(pixels)
                    num_rays = int(
                        num_rays
                        * (
                            self.config_file["target_sample_batch_size"]
                            / float(n_rendering_samples)
                        )
                    )
                    self.train_dataset.update_num_rays(min(2000, num_rays))

                # compute loss
                loss_rgb = F.smooth_l1_loss(rgb, pixels)
                loss_dep = F.smooth_l1_loss(depth, dep.unsqueeze(1))
                loss_sem = F.cross_entropy(semantic, sem)

                loss = loss_rgb * 10 + loss_dep / 5 + loss_sem / 2

                losses[0].append(loss_rgb.detach().cpu().item())
                losses[1].append(loss_dep.detach().cpu().item())
                losses[2].append(loss_sem.detach().cpu().item())

                optimizer.zero_grad()
                loss.backward()

                flag = False
                for name, param in radiance_field.named_parameters():
                    if torch.sum(torch.isnan(param.grad)) > 0:
                        flag = True
                        break

                if flag:
                    optimizer.zero_grad()
                    print(f"step jumped {model_idx}")
                    continue
                else:
                    optimizer.step()
                    scheduler.step()

                if model_idx == 0 and step % 500:
                    self.learning_rate_lst.append(scheduler._last_lr)

                # Evaluation
                if (
                    step == steps + 1
                    and (
                        (planning_step == 0)
                        or ((planning_step + 1) % 2 == 0)
                        or final_train
                    )
                    and model_idx == 0
                ):
                    radiance_field.eval()
                    estimator.eval()

                    psnrs = []
                    lpips = []
                    with torch.no_grad():
                        for i in tqdm.tqdm(range(num_test_images)):
                            data = self.test_dataset[test_idx[i]]
                            render_bkgd = data["color_bkgd"].to(curr_device)
                            ry = data["rays"]
                            rays = Rays(
                                origins=ry.origins.to(curr_device),
                                viewdirs=ry.viewdirs.to(curr_device),
                            )
                            pixels = data["pixels"].to(curr_device)
                            dep = data["dep"].to(curr_device)
                            sem_gt = data["sem"].to(curr_device)

                            # rendering
                            (
                                rgb,
                                acc,
                                depth,
                                sem,
                                _,
                            ) = render_image_with_occgrid_test(
                                1024,
                                # scene
                                radiance_field,
                                estimator,
                                rays,
                                # rendering options
                                near_plane=self.config_file["near_plane"],
                                render_step_size=self.config_file["render_step_size"],
                                render_bkgd=render_bkgd,
                                cone_angle=self.config_file["cone_angle"],
                                alpha_thre=self.config_file["alpha_thre"],
                            )
                            ground_truth_sem.append(sem_gt.cpu().numpy())
                            sem_imgs.append(sem.cpu().numpy())
                            self.sem_ce_ls[i].append(
                                F.cross_entropy(
                                    sem.reshape(
                                        (-1, radiance_field.num_semantic_classes)
                                    ),
                                    sem_gt.flatten(),
                                ).item()
                            )

                            lpips_fn = lambda x, y: self.lpips_net.to(curr_device)(
                                self.lpips_norm_fn(x), self.lpips_norm_fn(y)
                            ).mean()

                            mse = F.mse_loss(rgb, pixels)
                            psnr = -10.0 * torch.log(mse) / np.log(10.0)
                            psnrs.append(psnr.item())
                            # st()
                            lpips.append(lpips_fn(rgb, pixels).item())

                            mse_dep = F.mse_loss(depth, dep.unsqueeze(2))
                            self.mse_dep_lst[i].append(mse_dep.item())
                            ground_truth_imgs.append(pixels.cpu().numpy())
                            rendered_imgs[i].append(rgb.cpu().numpy())

                            ground_truth_depth.append(dep.cpu().numpy())
                            depth_imgs[i].append(depth.cpu().numpy())
                            self.psnrs_lst[i].append(psnr.item())
                            self.lpips_lst[i].append(lpips_fn(rgb, pixels).item())

            ## Save checkpoit for video
            # if final_train:  # (step + 1) % 1000 == 0:
            #     self.render(np.array([self.current_pose]))
            #     if not os.path.exists(self.save_path + "/checkpoints/"):
            #         os.makedirs(self.save_path + "/checkpoints/")

            #     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            #     checkpoint_path = (
            #         self.save_path
            #         + "/checkpoints/"
            #         + "model_"
            #         + str(current_time)
            #         + ".pth"
            #     )
            #     save_dict = {
            #         "occ_grid": self.estimators[0].binaries,
            #         "model": self.radiance_fields[0].state_dict(),
            #         "optimizer_state_dict": self.optimizers[0].state_dict(),
            #     }
            #     torch.save(save_dict, checkpoint_path)
            #     print("Saved checkpoints at", checkpoint_path)

            if step == steps - 1:
                # step == steps + 1 and (
                #     (planning_step == 0) or ((planning_step + 1) % 2 == 0) or final_train
                # ):
                print("start evaluation")

                print("loss")
                print(np.mean(np.array(losses), axis=1))

                eval_path = self.save_path + "/prediction/"
                if not os.path.exists(eval_path):
                    os.makedirs(eval_path)

                psnr_test = np.array(self.psnrs_lst)[:, -1]
                depth_mse_test = np.array(self.mse_dep_lst)[:, -1]
                sem_ce = np.array(self.sem_ce_ls)[:, -1]

                print("Mean PSNR: " + str(np.mean(psnr_test)))
                print("Mean Depth MSE: " + str(np.mean(depth_mse_test)))
                print("Mean Semantic CE: " + str(np.mean(sem_ce)))
                # self.errors_hist.append(
                #     [
                #         planning_step,
                #         np.mean(psnr_test),
                #         np.mean(depth_mse_test),
                #         np.mean(sem_ce),
                #     ]
                # )
                np.save(eval_path + "psnr_test.npy", np.array(self.psnrs_lst))
                np.save(eval_path + "depth_mse_test.npy", np.array(self.mse_dep_lst))
                np.save(eval_path + "sem_ce_test.npy", np.array(self.sem_ce_ls))
                np.save(eval_path + "lpips_test.npy", np.array(self.lpips_lst))

    def entropy_calculation(self, pose, idx):
        """uncertainty of each trajectory"""

        rendered_imgs = [[] for _ in range(self.config_file["n_ensembles"])]
        rendered_imgs_var = [[] for _ in range(self.config_file["n_ensembles"])]
        depth_imgs = [[] for _ in range(self.config_file["n_ensembles"])]
        depth_imgs_var = [[] for _ in range(self.config_file["n_ensembles"])]
        acc_imgs = [[] for _ in range(self.config_file["n_ensembles"])]
        sem_imgs = [[] for _ in range(self.config_file["n_ensembles"])]
        num_sample = 40  # self.config_file["sample_disc"] + 5
        for model_idx, (radiance_field, estimator) in enumerate(
            zip(self.radiance_fields, self.estimators)
        ):
            curr_device = (
                self.config_file["cuda"] if model_idx == 0 else self.config_file["cuda"]
            )

            radiance_field.eval()
            estimator.eval()

            with torch.no_grad():
                scale = 0.1
                (
                    rgb,
                    rgb_var,
                    depth,
                    depth_var,
                    acc,
                    sem,
                ) = Dataset.render_probablistic_image_from_pose(
                    radiance_field,
                    estimator,
                    np.array([pose]),
                    self.config_file["img_w"],
                    self.config_file["img_h"],
                    self.focal,
                    self.config_file["near_plane"],
                    self.config_file["render_step_size"],
                    scale,
                    self.config_file["cone_angle"],
                    self.config_file["alpha_thre"],
                    4,
                    curr_device,
                )

                rendered_imgs[model_idx].append(rgb[0])
                rendered_imgs_var[model_idx].append(rgb_var[0])
                depth_imgs[model_idx].append(depth[0])
                depth_imgs_var[model_idx].append(depth_var[0])
                acc_imgs[model_idx].append(acc[0])
                sem_imgs[model_idx].append(sem[0])

                # if model_idx == 0:
                #     if not os.path.exists(self.viz_save_path + "rgb/"):
                #         os.makedirs(self.viz_save_path + "rgb/")
                #         os.makedirs(self.viz_save_path + "depth/")
                #         os.makedirs(self.viz_save_path + "sem/")
                #         os.makedirs(self.viz_save_path + "acc/")

                #     cv2.imwrite(
                #         self.viz_save_path + "rgb/" + str(idx) + ".png",
                #         cv2.cvtColor(np.float32(rgb[0] * 255), cv2.COLOR_RGB2BGR),
                #     )

                #     cv2.imwrite(
                #         self.viz_save_path + "depth/" + str(idx) + ".png",
                #         np.clip(depth[0] * 25, 0, 255),
                #     )

                #     sem_argmax = np.argmax(sem[0], axis=2)
                #     sem_pd = d3_40_colors_rgb[sem_argmax.flatten()].reshape(
                #         sem_argmax.shape[0], sem_argmax.shape[1], 3
                #     )
                #     cv2.imwrite(
                #         self.viz_save_path + "sem/" + str(idx) + ".png",
                #         cv2.cvtColor(np.float32(sem_pd), cv2.COLOR_RGB2BGR),
                #     )

                #     cv2.imwrite(
                #         self.viz_save_path + "acc/" + str(idx) + ".png",
                #         np.clip(acc[0] * 255, 0, 255),
                #     )

        rendered_imgs = np.array(rendered_imgs)
        rendered_imgs_var = np.array(rendered_imgs_var)
        depth_imgs = np.array(depth_imgs)
        depth_imgs_var = np.array(depth_imgs_var)
        acc_imgs = np.array(acc_imgs)
        sem_imgs = np.array(sem_imgs)

        # rgb predictive information
        rgb_conditional_entropy = (
            np.log(2 * np.pi * np.e * rendered_imgs_var + 1e-4) / 2
        )
        rgb_entropy = np.mean(rgb_conditional_entropy, axis=0)

        # depth predictive information
        depth_conditional_entropy = np.log(2 * np.pi * np.e * depth_imgs_var + 1e-4) / 2
        depth_entropy = np.mean(depth_conditional_entropy, axis=0)

        # semantic entropy
        sem_p = F.softmax(torch.from_numpy(sem_imgs), dim=-1).numpy()
        sem_conditional_entropy = -np.sum(
            (sem_p + 1e-4) * np.log(sem_p + 1e-4), axis=-1
        )
        sem_entropy = np.mean(sem_conditional_entropy, axis=0)

        # occupancy entropy
        occ_conditional_entropy = -(acc_imgs + 1e-4) * np.log(acc_imgs + 1e-4) - (
            1 - acc_imgs + 1e-4
        ) * np.log(1 - acc_imgs + 1e-4)
        occ_entropy = np.mean(occ_conditional_entropy, axis=0)

        rgb_entropy = np.nan_to_num(rgb_entropy, nan=0)
        depth_entropy = np.nan_to_num(depth_entropy, nan=0)
        sem_entropy = np.nan_to_num(sem_entropy, nan=0)
        occ_entropy = np.nan_to_num(occ_entropy, nan=0)

        return (rgb_entropy, depth_entropy, sem_entropy, occ_entropy)

    def probablistic_uncertainty(self, trajectory, step):
        """uncertainty of each trajectory"""
        rendered_imgs = [[] for _ in range(self.config_file["n_ensembles"])]
        rendered_imgs_var = [[] for _ in range(self.config_file["n_ensembles"])]
        depth_imgs = [[] for _ in range(self.config_file["n_ensembles"])]
        depth_imgs_var = [[] for _ in range(self.config_file["n_ensembles"])]
        acc_imgs = [[] for _ in range(self.config_file["n_ensembles"])]
        sem_imgs = [[] for _ in range(self.config_file["n_ensembles"])]
        num_sample = 40  # self.config_file["sample_disc"] + 5
        for model_idx, (radiance_field, estimator) in enumerate(
            zip(self.radiance_fields, self.estimators)
        ):
            curr_device = (
                self.config_file["cuda"] if model_idx == 0 else self.config_file["cuda"]
            )

            radiance_field.eval()
            estimator.eval()

            with torch.no_grad():
                scale = 0.1
                a = np.linspace(0, len(trajectory) - 20, 20)
                b = np.linspace(len(trajectory) - 20, len(trajectory) - 1, 20)
                unc_idx = np.hstack((a, b)).astype(int)
                (
                    rgb,
                    rgb_var,
                    depth,
                    depth_var,
                    acc,
                    sem,
                ) = Dataset.render_probablistic_image_from_pose(
                    radiance_field,
                    estimator,
                    trajectory[unc_idx],
                    self.config_file["img_w"],
                    self.config_file["img_h"],
                    self.focal,
                    self.config_file["near_plane"],
                    self.config_file["render_step_size"],
                    scale,
                    self.config_file["cone_angle"],
                    self.config_file["alpha_thre"],
                    4,
                    curr_device,
                )

                rendered_imgs[model_idx].append(rgb[-num_sample:])
                rendered_imgs_var[model_idx].append(rgb_var[-num_sample:])
                depth_imgs[model_idx].append(depth[-num_sample:])
                depth_imgs_var[model_idx].append(depth_var[-num_sample:])
                acc_imgs[model_idx].append(acc[-num_sample:])
                sem_imgs[model_idx].append(sem[-num_sample:])

        rendered_imgs = np.array(rendered_imgs)
        rendered_imgs_var = np.array(rendered_imgs_var)
        depth_imgs = np.array(depth_imgs)
        depth_imgs_var = np.array(depth_imgs_var)
        acc_imgs = np.array(acc_imgs)
        sem_imgs = np.array(sem_imgs)

        # rgb predictive information
        rgb_conditional_entropy = (
            np.log(2 * np.pi * np.e * rendered_imgs_var + 1e-4) / 2
        )
        rgb_mean_conditional_entropy = np.mean(rgb_conditional_entropy, axis=0)

        rgb_ensemble_variance = np.sum(rendered_imgs_var, axis=0) / 2
        rgb_entropy = np.log(2 * np.pi * np.e * rgb_ensemble_variance + 1e-4) / 2

        rgb_mean_conditional_entropy = np.nan_to_num(
            rgb_mean_conditional_entropy, nan=0
        )
        rgb_entropy = np.nan_to_num(rgb_entropy, nan=0)

        rgb_predictive_information = np.mean(rgb_entropy - rgb_mean_conditional_entropy)

        # depth predictive information
        depth_conditional_entropy = np.log(2 * np.pi * np.e * depth_imgs_var + 1e-4) / 2
        depth_mean_conditional_entropy = np.mean(depth_conditional_entropy, axis=0)

        depth_ensemble_variance = np.sum(depth_imgs_var, axis=0) / 2
        depth_entropy = np.log(2 * np.pi * np.e * depth_ensemble_variance + 1e-4) / 2

        depth_mean_conditional_entropy = np.nan_to_num(
            depth_mean_conditional_entropy, nan=0
        )
        depth_entropy = np.nan_to_num(depth_entropy, nan=0)

        depth_predictive_information = np.mean(
            depth_entropy - depth_mean_conditional_entropy
        )

        # semantic entropy
        sem_p = F.softmax(torch.from_numpy(sem_imgs), dim=-1).numpy()
        sem_conditional_entropy = -np.sum(
            (sem_p + 1e-4) * np.log(sem_p + 1e-4), axis=-1
        )
        sem_mean_conditional_entropy = np.mean(sem_conditional_entropy, axis=0)

        sem_ensemble_p = np.mean(sem_p, axis=0)
        sem_entropy = -np.sum(
            (sem_ensemble_p + 1e-4) * np.log(sem_ensemble_p + 1e-4), axis=-1
        )

        sem_mean_conditional_entropy = np.nan_to_num(
            sem_mean_conditional_entropy, nan=0
        )
        sem_entropy = np.nan_to_num(sem_entropy, nan=0)

        sem_predictive_information = np.mean(sem_entropy - sem_mean_conditional_entropy)

        # occupancy entropy
        occ_conditional_entropy = -(acc_imgs + 1e-4) * np.log(acc_imgs + 1e-4) - (
            1 - acc_imgs + 1e-4
        ) * np.log(1 - acc_imgs + 1e-4)
        occ_mean_conditional_entropy = np.mean(occ_conditional_entropy, axis=0)

        occ_ensemble_p = np.mean(acc_imgs, axis=0)
        occ_entropy = -(occ_ensemble_p + 1e-4) * np.log(occ_ensemble_p + 1e-4) - (
            1 - occ_ensemble_p + 1e-4
        ) * np.log(1 - occ_ensemble_p + 1e-4)

        occ_mean_conditional_entropy = np.nan_to_num(
            occ_mean_conditional_entropy, nan=0
        )
        occ_entropy = np.nan_to_num(occ_entropy, nan=0)

        occ_predictive_information = np.mean(occ_entropy - occ_mean_conditional_entropy)

        predictive_information = (
            rgb_predictive_information
            + depth_predictive_information
            + sem_predictive_information * 3
            + occ_predictive_information * 2
        )

        self.trajector_uncertainty_list[step - 1].append(
            [
                rgb_predictive_information,
                depth_predictive_information,
                sem_predictive_information * 3,
                occ_predictive_information * 2,
            ]
        )
        # print(
        #     rgb_predictive_information,
        #     depth_predictive_information,
        #     sem_predictive_information * 3,
        #     occ_predictive_information * 2,
        # )
        # print(predictive_information)
        return (
            predictive_information,
            rgb_entropy - rgb_mean_conditional_entropy,
            depth_entropy - depth_mean_conditional_entropy,
            sem_entropy - sem_mean_conditional_entropy,
            occ_entropy - occ_mean_conditional_entropy,
        )

    def trajector_uncertainty(self, trajectory, step):
        """uncertainty of each trajectory"""
        rendered_imgs = []
        depth_imgs = []
        acc_imgs = []
        sem_imgs = []
        num_sample = 40  # self.config_file["sample_disc"] + 5
        for model_idx, (radiance_field, estimator) in enumerate(
            zip(self.radiance_fields, self.estimators)
        ):
            curr_device = (
                self.config_file["cuda"] if model_idx == 0 else self.config_file["cuda"]
            )

            radiance_field.eval()
            estimator.eval()

            with torch.no_grad():
                scale = 0.1
                a = np.linspace(0, len(trajectory) - 20, 20)
                b = np.linspace(len(trajectory) - 20, len(trajectory) - 1, 20)
                unc_idx = np.hstack((a, b)).astype(int)
                if model_idx == 0:
                    rgb, depth, acc, sem = Dataset.render_image_from_pose(
                        radiance_field,
                        estimator,
                        trajectory[unc_idx],
                        self.config_file["img_w"],
                        self.config_file["img_h"],
                        self.focal,
                        self.config_file["near_plane"],
                        self.config_file["render_step_size"],
                        scale,
                        self.config_file["cone_angle"],
                        self.config_file["alpha_thre"],
                        4,
                        curr_device,
                    )
                    sem_imgs.append(sem[-num_sample:])
                else:
                    rgb, depth, acc = Dataset.render_image_from_pose(
                        radiance_field,
                        estimator,
                        trajectory[unc_idx],
                        self.config_file["img_w"],
                        self.config_file["img_h"],
                        self.focal,
                        self.config_file["near_plane"],
                        self.config_file["render_step_size"],
                        scale,
                        self.config_file["cone_angle"],
                        self.config_file["alpha_thre"],
                        4,
                        curr_device,
                    )

                rendered_imgs.append(rgb[-num_sample:])
                depth_imgs.append(depth[-num_sample:])
                acc_imgs.append(acc[-num_sample:])

        # semantic uncertainty by entropy
        rendered_imgs = np.array(rendered_imgs)
        sem_imgs = np.array(sem_imgs)

        sem_p = F.softmax(torch.from_numpy(sem_imgs), dim=-1).numpy()
        sem_entropy = -np.sum(sem_p * np.log(sem_p + 1e-10), axis=-1)

        depth_imgs = np.array(depth_imgs)

        acc_imgs = np.array(acc_imgs[0]) + 1e-4

        intensity_var = np.mean(np.var(rendered_imgs, axis=0), axis=-1)
        depth_var = np.var(depth_imgs, axis=0)

        # 0 ~ 20
        intensity_var_mean = np.clip(np.mean(intensity_var, axis=(1, 2)) * 4000, 0, 100)
        depth_var_mean = np.clip(np.mean(depth_var, axis=(1, 2)) * 50, 0, 100)
        acc_inv_mean = np.mean(np.clip(1 / acc_imgs - 1, 0, 10000), axis=(1, 2))

        if self.radiance_fields[0].num_semantic_classes > 0:
            sem_entropy_mean = np.clip(
                np.mean(sem_entropy, axis=(0, 2, 3)) * 50, 0, 100
            )
            uncertainty = (
                intensity_var_mean + depth_var_mean + acc_inv_mean + sem_entropy_mean
            )
        else:
            uncertainty = intensity_var_mean + depth_var_mean + acc_inv_mean

        if step == -1:
            max_idx = np.argsort(uncertainty)
            max_idx = np.sort(max_idx)
            uncertainty = np.mean(uncertainty[-11:])
        else:
            max_idx = np.argsort(uncertainty)
            max_idx = np.sort(max_idx)
            uncertainty = np.mean(uncertainty[max_idx])

        if self.radiance_fields[0].num_semantic_classes > 0:
            self.trajector_uncertainty_list[step - 1].append(
                [
                    intensity_var_mean[-num_sample:],
                    depth_var_mean[-num_sample:],
                    acc_inv_mean[-num_sample:],
                    sem_entropy_mean[-num_sample:],
                ]
            )
        else:
            self.trajector_uncertainty_list[step - 1].append(
                [
                    intensity_var_mean[-num_sample:],
                    depth_var_mean[-num_sample:],
                    acc_inv_mean[-num_sample:],
                ]
            )

        return uncertainty, max_idx

    def render(self, traj):
        traj = traj[:1][:1]
        traj1 = np.copy(traj)
        traj2 = np.copy(traj)
        step = self.sim_step

        render_images = np.array(self.sim.render_tpv(traj))
        if not os.path.exists(self.viz_save_path):
            os.makedirs(self.viz_save_path)
        for img in render_images:
            cv2.imwrite(self.viz_save_path + str(self.sim_step) + ".png", img)
            self.sim_step += 1

        render_images = np.array(self.sim.render_top_tpv(traj, self.aabb.cpu().numpy()))
        if not os.path.exists(self.viz_save_path):
            os.makedirs(self.viz_save_path)
        if not os.path.exists(self.viz_save_path + "top/"):
            os.makedirs(self.viz_save_path + "top/")
        for s, img in enumerate(render_images):
            cv2.imwrite(self.viz_save_path + "top/" + str(step + s) + ".png", img)

        fpv_path = self.viz_save_path + "fpv/"
        if not os.path.exists(fpv_path):
            os.makedirs(fpv_path)
            os.makedirs(fpv_path + "gt_rgb/")
            os.makedirs(fpv_path + "gt_dep/")
            os.makedirs(fpv_path + "gt_sem/")
            os.makedirs(fpv_path + "pd_rgb/")
            os.makedirs(fpv_path + "pd_dep/")
            os.makedirs(fpv_path + "pd_occ/")
            os.makedirs(fpv_path + "pd_sem/")

        (
            sampled_images,
            sampled_depth_images,
            sampled_sem_images,
        ) = self.sim.sample_images_from_poses(traj1)

        (
            rgb_predictions,
            depth_predictions,
            acc_predictions,
            sem_predictions,
        ) = Dataset.render_image_from_pose(
            self.radiance_fields[0],
            self.estimators[0],
            traj2,
            self.config_file["img_w"],
            self.config_file["img_h"],
            self.focal,
            self.config_file["near_plane"],
            self.config_file["render_step_size"],
            1,
            self.config_file["cone_angle"],
            self.config_file["alpha_thre"],
            1,
            self.config_file["cuda"],
        )

        for st, (rgb, dep, sem, rgb_pd, dep_pd, acc_pd, sem_pd) in enumerate(
            zip(
                sampled_images,
                sampled_depth_images,
                sampled_sem_images,
                rgb_predictions,
                depth_predictions,
                acc_predictions,
                sem_predictions,
            )
        ):
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(
                fpv_path + "gt_rgb/" + str(step + st) + ".png",
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                fpv_path + "pd_rgb/" + str(step + st) + ".png",
                cv2.cvtColor(np.float32(rgb_pd * 255), cv2.COLOR_RGB2BGR),
            )

            cv2.imwrite(
                fpv_path + "gt_dep/" + str(step + st) + ".png",
                np.clip(dep * 25, 0, 255),
            )
            cv2.imwrite(
                fpv_path + "pd_dep/" + str(step + st) + ".png",
                np.clip(dep_pd * 25, 0, 255),
            )

            sem = d3_40_colors_rgb[sem.flatten()].reshape(sem.shape[0], sem.shape[1], 3)
            cv2.imwrite(
                fpv_path + "gt_sem/" + str(step + st) + ".png",
                cv2.cvtColor(np.float32(sem), cv2.COLOR_RGB2BGR),
            )
            sem_argmax = np.argmax(sem_pd, axis=2)
            sem_pd = d3_40_colors_rgb[sem_argmax.flatten()].reshape(
                sem_argmax.shape[0], sem_argmax.shape[1], 3
            )
            cv2.imwrite(
                fpv_path + "pd_sem/" + str(step + st) + ".png",
                cv2.cvtColor(np.float32(sem_pd), cv2.COLOR_RGB2BGR),
            )

            cv2.imwrite(
                fpv_path + "pd_occ/" + str(step + st) + ".png",
                np.clip(acc_pd * 255, 0, 255),
            )

    def planning(self, steps, training_steps_per_step):
        print("Planning Thread Started")

        current_state = self.global_origin[:3]

        def occ_eval_fn(x):
            density = self.radiance_field.query_density(x)
            return density * self.config_file["render_step_size"]

        sim_step = 0

        loss_orig = []
        loss_improved = []
        loss_improved_clr = []

        feedback_ig = []
        feedback_ogd = []
        feedback_ogd_clr = []
        feedback_preds = []

        step = 0
        flag = True
        while flag and step < self.config_file["planning_step"]:
            print("planning step: " + str(step))
            step += 1

            # get voxel grid
            # filter out invisible cells
            # st()

            invis_map = self.estimators[1].mark_invisible_cells(
                self.train_dataset.K[None, :, :],
                self.train_dataset.camtoworlds,
                self.train_dataset.width,
                self.train_dataset.height,
                depth=self.train_dataset.depths,
            )
            invis_map = (invis_map < 0).view(self.estimators[1].binaries.shape)
            # invis_map = self.estimators[0].binaries

            plt.imsave(
                "invis.jpg",
                (invis_map < 0).view(self.estimators[0].binaries.shape)[0, :, 8].cpu(),
            )
            plt.imsave("binary.jpg", self.estimators[0].binaries[0, :, 8].cpu())
            plt.imsave("binary1.jpg", self.estimators[1].binaries[0, :, 8].cpu())

            voxel_grid = torch.logical_or(
                self.estimators[1].binaries,
                invis_map,  # (invis_map < 0).view(self.estimators[1].binaries.shape),
            )
            voxel_grid = voxel_grid.cpu().numpy()
            vg = np.swapaxes(voxel_grid, 2, 3)

            plt.imsave("combine1.jpg", voxel_grid[0, :, 8])

            voxel_grid1 = torch.logical_or(
                self.estimators[0].binaries,
                invis_map,  # (invis_map < 0).view(self.estimators[0].binaries.shape),
            )
            voxel_grid1 = voxel_grid1.cpu().numpy()
            vg1 = np.swapaxes(voxel_grid1, 2, 3)

            plt.imsave("combine.jpg", voxel_grid1[0, :, 8])

            print("sampling trajectory from: " + str(current_state))

            xyz_state = np.copy(current_state)
            xyz_state[1] = current_state[2]
            xyz_state[2] = current_state[1]

            aabb = np.copy(self.aabb.cpu().numpy())
            aabb[1] = self.aabb[2]
            aabb[2] = self.aabb[1]
            aabb[4] = self.aabb[5]
            aabb[5] = self.aabb[4]

            N_sample_traj_pose = sample_traj(
                voxel_grid=np.array([vg, vg1]),
                current_state=xyz_state,
                N_traj=self.config_file["num_traj"],
                aabb=aabb,
                sim=self.sim,
                cost_map=self.cost_map,
                N_sample_disc=self.config_file["sample_disc"],
                voxel_grid_size=self.config_file["main_grid_size"],
                visiting_map=self.visiting_map,
                save_path=self.save_path,
            )
            copy_traj = N_sample_traj_pose.copy()

            if self.policy_type == "uncertainty":
                uncertainties = []
                rgb_info_gains = []
                depth_info_gains = []
                sem_info_gains = []
                occ_info_gains = []
                for i in tqdm.tqdm(range(self.config_file["num_traj"])):
                    (
                        uncertainty,
                        rgb_info_gain,
                        depth_info_gain,
                        sem_info_gain,
                        occ_info_gain,
                    ) = self.probablistic_uncertainty(N_sample_traj_pose[i], step)

                    rgb_info_gain_copy = np.copy(rgb_info_gain)
                    depth_info_gain_copy = np.copy(depth_info_gain)
                    sem_info_gain_copy = np.copy(sem_info_gain)
                    occ_info_gain_copy = np.copy(occ_info_gain)
                    # OGD PREDICTION
                    info_gain_preds = [[] for i in range(6)]
                    info_gain_preds[0].append(rgb_info_gain_copy[0, :, :, :, 0])
                    info_gain_preds[1].append(rgb_info_gain_copy[0, :, :, :, 1])
                    info_gain_preds[2].append(rgb_info_gain_copy[0, :, :, :, 2])
                    info_gain_preds[3].append(depth_info_gain_copy[0, :, :, :])
                    info_gain_preds[4].append(sem_info_gain_copy[0, :, :, :])
                    info_gain_preds[5].append(occ_info_gain_copy[0, :, :, :])

                    info_gain_preds = np.squeeze(
                        np.clip(np.array(info_gain_preds), a_min=0, a_max=5)
                    )

                    # ogd
                    info_gain_hat = self.online_gd.improve_prediction(
                        np.copy(info_gain_preds)
                    )
                    info_gain_hat = np.clip(info_gain_hat, a_min=0, a_max=5)

                    # ogd clr
                    info_gain_hat_clr = self.online_gd_clr.improve_prediction(
                        np.copy(info_gain_preds)
                    )
                    info_gain_hat_clr = np.clip(info_gain_hat_clr, a_min=0, a_max=5)

                    # if args.online_gd:
                    # if args.online_gd:
                    # print("OGD imrpoving")
                    # uncertainty = (
                    #     np.mean(info_gain_hat_clr[:3])
                    #     + np.mean(info_gain_hat_clr[3])
                    #     + np.mean(info_gain_hat_clr[4]) * 3
                    #     + np.mean(info_gain_hat_clr[5]) * 2
                    # )

                    uncertainties.append(uncertainty)
                    rgb_info_gains.append(rgb_info_gain)
                    depth_info_gains.append(depth_info_gain)
                    sem_info_gains.append(sem_info_gain)
                    occ_info_gains.append(occ_info_gain)

                best_index = np.argmax(np.array(uncertainties))
                rgb_info_gains = rgb_info_gains[best_index]
                depth_info_gains = depth_info_gains[best_index]
                sem_info_gains = sem_info_gains[best_index]
                occ_info_gains = occ_info_gains[best_index]

                a = np.linspace(
                    0,
                    len(N_sample_traj_pose[best_index]) - 20,
                    35,
                )
                b = np.linspace(
                    len(N_sample_traj_pose[best_index]) - 20,
                    len(N_sample_traj_pose[best_index]) - 1,
                    5,
                )
                unc_idx = np.hstack((a, b)).astype(int)

                (
                    sampled_images,
                    sampled_depth_images,
                    sampled_sem_images,
                ) = self.sim.sample_images_from_poses(
                    N_sample_traj_pose[best_index][unc_idx]
                )

                sampled_images = sampled_images[:, :, :, :3]

                self.render(copy_traj[best_index])
                self.current_pose = copy_traj[best_index][-1]

                sampled_poses_mat = []
                for pose in N_sample_traj_pose[best_index][unc_idx]:
                    T = np.eye(4).astype(np.float32)
                    T[:3, :3] = R.from_quat(pose[3:]).as_matrix()
                    T[:3, 3] = pose[:3]
                    sampled_poses_mat.append(T)

                self.train_dataset.update_data(
                    sampled_images,
                    sampled_depth_images,
                    sampled_sem_images,
                    sampled_poses_mat,
                )

                # update visiable voxels
                # invisible_idx = self.estimators[1].occs < 0
                # self.estimators[1].occs[invisible_idx] = 1e-2
                # self.estimators[1].mark_invisible_cells(
                #     self.train_dataset.K[None, :, :],
                #     self.train_dataset.camtoworlds,
                #     self.train_dataset.width,
                #     self.train_dataset.height,
                #     depth=self.train_dataset.depths,
                # )

                info_gain_feedbacks = [[] for i in range(6)]
                for i in tqdm.tqdm(range(40, 0, -1)):
                    pose = N_sample_traj_pose[best_index][unc_idx[-i]]
                    # mat = self.train_dataset[-i]
                    # quat = R.from_matrix(mat[:3, :3]).as_quat()
                    # pose = np.hstack((mat[:3, 3], quat))

                    # print(pose)
                    # st()
                    entropy_old = self.entropy_calculation(pose, i * 2)

                    # train
                    self.training_single_image(-i)

                    entropy_new = self.entropy_calculation(pose, i * 2 - 1)

                    rgb_gain = entropy_old[0] - entropy_new[0]
                    info_gain_feedbacks[0].append(rgb_gain[0, :, :, 0])
                    info_gain_feedbacks[1].append(rgb_gain[0, :, :, 1])
                    info_gain_feedbacks[2].append(rgb_gain[0, :, :, 2])
                    info_gain_feedbacks[3].append((entropy_old[1] - entropy_new[1])[0])
                    info_gain_feedbacks[4].append((entropy_old[2] - entropy_new[2])[0])
                    info_gain_feedbacks[5].append((entropy_old[3] - entropy_new[3])[0])

                # OGD PREDICTION
                info_gain_preds = [[] for i in range(6)]
                info_gain_preds[0].append(rgb_info_gains[0, :, :, :, 0])
                info_gain_preds[1].append(rgb_info_gains[0, :, :, :, 1])
                info_gain_preds[2].append(rgb_info_gains[0, :, :, :, 2])
                info_gain_preds[3].append(depth_info_gains[0, :, :, :])
                info_gain_preds[4].append(sem_info_gains[0, :, :, :])
                info_gain_preds[5].append(occ_info_gains[0, :, :, :])

                info_gain_preds = np.squeeze(
                    np.clip(np.array(info_gain_preds), a_min=0, a_max=5)
                )

                # ogd
                info_gain_hat = self.online_gd.improve_prediction(info_gain_preds)
                info_gain_hat = np.clip(info_gain_hat, a_min=0, a_max=5)

                # ogd clr
                info_gain_hat_clr = self.online_gd_clr.improve_prediction(
                    info_gain_preds
                )
                info_gain_hat_clr = np.clip(info_gain_hat_clr, a_min=0, a_max=5)

                info_gain_feedbacks = np.clip(
                    np.array(info_gain_feedbacks), a_min=0, a_max=5
                )

                self.online_gd.update_d(
                    info_gain_feedbacks, info_gain_preds, info_gain_hat
                )

                self.online_gd_clr.update_d(
                    info_gain_feedbacks, info_gain_preds, info_gain_hat_clr
                )

                loss_orig.append(np.sum(np.abs(info_gain_feedbacks - info_gain_preds)))

                loss_improved.append(
                    np.sum(np.abs(info_gain_feedbacks - info_gain_hat))
                )

                loss_improved_clr.append(
                    np.sum(np.abs(info_gain_feedbacks - info_gain_hat_clr))
                )

                feedback_ig.append(info_gain_feedbacks)
                feedback_preds.append(info_gain_preds)
                feedback_ogd.append(info_gain_hat)
                feedback_ogd_clr.append(info_gain_hat_clr)

                # fig, ax = plt.subplots(6, 3, figsize=(15, 20))

                # type_ls = ["r", "g", "b", "depth", "sem", "occ"]
                # name_ls = ["feedback", "pred", "hat"]
                # for i in range(6):
                #     ax[i, 0].hist(
                #         info_gain_feedbacks[i].flatten(),
                #         bins=50,
                #         alpha=0.8,
                #         label=f"{type_ls[i]} feedback",
                #         log=True,
                #     )

                #     ax[i, 1].hist(
                #         info_gain_preds[i].flatten(),
                #         bins=50,
                #         alpha=0.8,
                #         label=f"{type_ls[i]} pred",
                #         log=True,
                #     )

                #     ax[i, 2].hist(
                #         info_gain_hat[i].flatten(),
                #         bins=50,
                #         alpha=0.8,
                #         label=f"{type_ls[i]} hat",
                #         log=True,
                #     )

                # plt.savefig(self.save_path + f"/info_gain_hist_{step}.png")
                # fig.clf()

                # remove low info images
                # for i in range(40, 0, -1):
                #     low_ig_threshold = 0.1

                #     if np.sum(np.mean(info_gain_feedbacks[:3, -i])) < low_ig_threshold:
                #         self.train_dataset.remove_data(-i)
                #         print(f"remove {-i}th image")

                for i, (mat, d_img) in enumerate(
                    zip(sampled_poses_mat[-6:], sampled_depth_images[-6:])
                ):
                    d_points = d_img[int(d_img.shape[0] / 2)]
                    R_m = mat[:3, :3]
                    euler = R.from_matrix(R_m).as_euler("yzx")
                    d_angles = (self.align_angles + euler[0]) % (2 * np.pi)
                    w_loc = mat[:3, 3]
                    grid_loc = np.array(
                        (w_loc - self.aabb.cpu().numpy()[:3])
                        // self.config_file["main_grid_size"],
                        dtype=int,
                    )

                    self.cost_map, visiting_map = update_cost_map(
                        cost_map=self.cost_map,
                        depth=d_points,
                        angle=d_angles,
                        g_loc=grid_loc,
                        w_loc=w_loc,
                        aabb=self.aabb.cpu().numpy(),
                        resolution=self.config_file["main_grid_size"],
                    )
                    self.visiting_map += visiting_map

                current_state = N_sample_traj_pose[best_index][unc_idx][-1, :3]
            elif self.policy_type == "random":
                uncertainty, mid = self.trajector_uncertainty(
                    N_sample_traj_pose[0], step
                )

                (
                    sampled_images,
                    sampled_depth_images,
                    sampled_sem_images,
                ) = self.sim.sample_images_from_poses(N_sample_traj_pose[0])
                best_index = 0

                sampled_images = sampled_images[:, :, :, :3]

                self.render(N_sample_traj_pose[best_index][1:])

                sampled_poses_mat = []
                for pose in N_sample_traj_pose[best_index]:
                    T = np.eye(4)
                    T[:3, :3] = R.from_quat(pose[3:]).as_matrix()
                    T[:3, 3] = pose[:3]
                    sampled_poses_mat.append(T)

                for i, d_img in enumerate(sampled_depth_images):
                    d_points = d_img[int(d_img.shape[0] / 2)]
                    R_m = sampled_poses_mat[i][:3, :3]
                    euler = R.from_matrix(R_m).as_euler("yzx")
                    d_angles = (self.align_angles + euler[0]) % (2 * np.pi)
                    w_loc = sampled_poses_mat[i][:3, 3]
                    grid_loc = np.array(
                        (w_loc - self.aabb.cpu().numpy()[:3])
                        // self.config_file["main_grid_size"],
                        dtype=int,
                    )
                    self.cost_map = update_cost_map(
                        self.cost_map,
                        d_points,
                        d_angles,
                        grid_loc,
                        self.aabb.cpu().numpy(),
                        self.config_file["main_grid_size"],
                    )

                self.train_dataset.update_data(
                    sampled_images,
                    sampled_depth_images,
                    sampled_sem_images,
                    sampled_poses_mat,
                )

                current_state = N_sample_traj_pose[best_index][-1, :3]

                self.current_pose = N_sample_traj_pose[best_index][-1]

            elif self.policy_type == "spatial":
                (
                    sampled_images,
                    sampled_depth_images,
                    sampled_sem_images,
                ) = None

            print("plan finished at: " + str(current_state))

            self.nerf_training(training_steps_per_step, planning_step=step)

            past_unc = np.array(self.trajector_uncertainty_list[:step]).astype(float)

            unc = np.max(np.mean(past_unc, axis=2), axis=1)

            fig, ax = plt.subplots(figsize=(10, 15))

            ax.plot(np.cumsum(np.array(loss_orig)), label="original")
            ax.plot(np.cumsum(np.array(loss_improved)), label="improved")
            ax.plot(np.cumsum(np.array(loss_improved_clr)), label="improved_clr")
            ax.legend()
            fig.savefig(self.save_path + "/loss.png")
            fig.clf()
            plt.close(fig)
            # if step >= 5:
            #     if (
            #         unc[step - 1] > 0.05
            #         and unc[step - 2] > 0.05
            #         and unc[step - 3] > 0.05
            #         and unc[step - 4] > 0.05
            #         and unc[step - 5] > 0.05
            #     ):
            #         flag = False

        np.save(self.save_path + "/loss.npy", np.array(loss_orig))
        np.save(self.save_path + "/loss_improved.npy", np.array(loss_improved))
        np.save(self.save_path + "/loss_improved_clr.npy", np.array(loss_improved_clr))

        np.save(self.save_path + "/feedback_ig.npy", np.array(feedback_ig))
        np.save(self.save_path + "/feedback_preds.npy", np.array(feedback_preds))
        np.save(self.save_path + "/feedback_ogd.npy", np.array(feedback_ogd))
        np.save(self.save_path + "/feedback_ogd_clr.npy", np.array(feedback_ogd_clr))

    def pipeline(self):
        self.initialization()

        # self.nerf_training(3 * self.config_file["training_steps"])
        self.nerf_training(2000, initial_train=True)

        self.planning(
            self.config_file["planning_step"], int(self.config_file["training_steps"])
        )

        # self.nerf_training(
        #     self.config_file["training_steps"] * 3, final_train=True, planning_step=-10
        # )
        self.nerf_training(2000, final_train=True, planning_step=-10)

        plt.plot(np.arange(len(self.learning_rate_lst)), self.learning_rate_lst)
        if args.online_gd:
            plt.savefig(self.save_path + "/learning_rate_ogd.png")
        else:
            plt.savefig(self.save_path + "/learning_rate.png")

        plt.yscale("log")
        plt.plot(np.arange(len(self.learning_rate_lst)), self.learning_rate_lst)
        plt.savefig(self.save_path + "/learning_rate_log.png")

        # save radiance field, estimator, and optimzer
        print("Saving Models")
        # save_model(radiance_field, estimator, "test")

        self.train_dataset.save()
        self.test_dataset.save()

        if not os.path.exists(self.save_path + "/checkpoints/"):
            os.makedirs(self.save_path + "/checkpoints/")

        # self.trajector_uncertainty_list = np.array(self.trajector_uncertainty_list)
        # np.save(self.save_path + "/uncertainty.npy", self.trajector_uncertainty_list)
        with open(self.save_path + "/uncertainty.pkl", "wb") as f:
            pickle.dump(self.trajector_uncertainty_list, f)

        # self.errors_hist = np.array(self.errors_hist)
        # np.save(self.save_path + "/errors.npy", self.errors_hist)
        with open(self.save_path + "/errors.pkl", "wb") as f:
            pickle.dump(self.errors_hist, f)

        for i, (radiance_field, estimator, optimizer, scheduler) in enumerate(
            zip(self.radiance_fields, self.estimators, self.optimizers, self.schedulers)
        ):
            checkpoint_path = (
                self.save_path + "/checkpoints/" + "model_" + str(i) + ".pth"
            )
            save_dict = {
                "occ_grid": estimator.binaries,
                "model": radiance_field.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(save_dict, checkpoint_path)
            print("Saved checkpoints at", checkpoint_path)


if __name__ == "__main__":
    args = parse_args()

    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)

    mapper = ActiveNeRFMapper(args)
    mapper.pipeline()
