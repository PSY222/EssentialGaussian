import os
import torch
from random import randint
from gaussian_renderer import render, count_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import getWorld2View2
from icecream import ic
import random
import copy
import gc
import numpy as np
from collections import defaultdict

def calculate_v_imp_score(gaussians, imp_list, v_pow,option='Lgs'):
    if option =='Lgs':
        volume = np.prod(gaussians.get_scaling(), axis=1) #scene/gaussian_model.py file
        sorted_volume = np.sort(volume)[::-1]
        kth_largest = sorted_volume[int(len(volume) * 0.9)]
        v_list = (volume / kth_largest) ** v_pow
    #elif : other scoring method
    '''
    :return: A list of adjusted values (v_list) used for pruning.
    
    It is delivered to scene/gaussian_model/prune_gaussian function!!
    '''
    return v_list * imp_list

# def calculate_iou(gaussian1, gaussian2,method='ellipses'):
#     if method =='ellipses':
#     elif method == 'mean_dst':
           
#     return iou

def prune_list(gaussians, scene, pipe, background):
    viewpoint_stack = scene.getTrainCameras().copy()
    gaussian_list, imp_list = None, None
    viewpoint_cam = viewpoint_stack.pop()
    render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
    gaussian_list, imp_list = (
        render_pkg["gaussians_count"],
        render_pkg["important_score"],
    )

    # Initialize prob_list
    prob_list = imp_list / imp_list.sum()

    ic(dataset.model_path)
    for iteration in range(len(viewpoint_stack)):
        # Pick a random Camera
        # prunning
        viewpoint_cam = viewpoint_stack.pop()
        render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gaussians_count, important_score = (
            render_pkg["gaussians_count"].detach(),
            render_pkg["important_score"].detach(),
        )
        gaussian_list += gaussians_count
        imp_list += important_score
        prob_list += (important_score/ important_score.sum())
        gc.collect()
    return gaussian_list, imp_list, prob_list

# Make the function to return gaussian_list,imp_list
# prune and select gaussian ->calculate_importance socre 로 연결되어야 함
# def prune_and_select_gaussians(gaussians, scene, pipe, background, threshold, nyquist_limit, v_pow):
#     # Step 1: Calculate importance scores
#     gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)

#     scores = calculate_importance_score(gaussian_list, imp_list, v_pow)
#     # Initialise probability list
#     prob_list = (scores/scores.sum())
#     gaussians_with_scores = sorted(zip(gaussian_list, scores), key=lambda x: x[1], reverse=True)
#     return aggregate_gaussians(selected_gaussians)

#Find the corresponding part on MS-GS code
def aggregate_gaussians(gaussians):
    lmax = calculate_lmax(gaussians)
    for lm in range(2, lmax + 1):
        coverage = pixel_coverage(gaussians, scale=4**(lm - 1))
        Gsmall = [g for g in gaussians if coverage[g] < ST]
        voxel_dict = {}
        for g in Gsmall:
            voxel_id = get_voxel_id(g, lm)
            voxel_dict.setdefault(voxel_id, []).append(g)
        for voxel_gaussians in voxel_dict.values():
            if voxel_gaussians:
                Gnew = enlarge(average(voxel_gaussians))
                insert_into_scene(Gnew)
    return get_combined_gaussians()

# Placeholder helper functions
def prune_list(gaussians, scene, pipe, background): pass
def calculate_lmax(gaussians): pass
def pixel_coverage(gaussians, scale): pass
def get_voxel_id(gaussian, lm): pass
def average(gaussians): pass
def enlarge(gaussian): pass
def insert_into_scene(gaussian): pass
def get_combined_gaussians(): pass
def calculate_nyquist_limit(scene_complexity, desired_reconstruction_quality): pass
