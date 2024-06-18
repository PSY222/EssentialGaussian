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

def calculate_importance_score(gaussians, imp_list, v_pow,option='Lgs'):
    if option =='Lgs':
        volume = np.prod(gaussians.get_scaling(), axis=1)
        sorted_volume = np.sort(volume)[::-1]
        kth_largest = sorted_volume[int(len(volume) * 0.9)]
        v_list = (volume / kth_largest) ** v_pow
    #other scoring method
    return v_list * imp_list

def calculate_iou(gaussian1, gaussian2,method='ellipses'):
    if method =='ellipses':
    elif method == 'mean_dst':
           
    return iou

def prune_and_select_gaussians(gaussians, scene, pipe, background, threshold, nyquist_limit, v_pow):
    # Step 1: Calculate importance scores
    gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
    scores = calculate_importance_score(gaussian_list, imp_list, v_pow)
    gaussians_with_scores = sorted(zip(gaussian_list, scores), key=lambda x: x[1], reverse=True)

    # Step 2: Suppress Gaussians based on IoU and threshold, ensuring Nyquist limit
    selected_gaussians = []
    for gaussian, score in gaussians_with_scores:
        if all(calculate_iou(gaussian, sg) <= threshold for sg in selected_gaussians):
            selected_gaussians.append(gaussian)
            if len(selected_gaussians) >= nyquist_limit:
                break
    return aggregate_gaussians(selected_gaussians)

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

def main():
    gaussians = load_initial_gaussians()
    iou_threshold = 0.5
    nyquist_limit = calculate_nyquist_limit(scene_complexity, desired_reconstruction_quality)
    v_pow = 2
    final_gaussians = prune_and_select_gaussians(gaussians, scene, pipe, background, iou_threshold, nyquist_limit, v_pow)
    output_final_gaussians(final_gaussians)

# Placeholder helper functions
def bounding_ellipse_iou(gaussian1, gaussian2): pass
def prune_list(gaussians, scene, pipe, background): pass
def calculate_lmax(gaussians): pass
def pixel_coverage(gaussians, scale): pass
def get_voxel_id(gaussian, lm): pass
def average(gaussians): pass
def enlarge(gaussian): pass
def insert_into_scene(gaussian): pass
def get_combined_gaussians(): pass
def load_initial_gaussians(): pass
def output_final_gaussians(gaussians): pass
def calculate_nyquist_limit(scene_complexity, desired_reconstruction_quality): pass
