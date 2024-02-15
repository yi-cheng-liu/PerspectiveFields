import argparse
import copy
import glob
import json
import math
import multiprocessing
import multiprocessing as mp
import os
import pickle
import tempfile
import time
import warnings
from multiprocessing import Pool, Process, Queue

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.data.common import MapDataset
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from tqdm import tqdm

import perspective2d.modeling  # noqa
from perspective2d.config import get_perspective2d_cfg_defaults
from perspective2d.data import PerspectiveMapper
from perspective2d.utils.predictor import VisualizationDemo


# TODO: set args type
def setup_cfg(args):
    """setup model configurations

    Args:
        args (_type_): command-line arguments

    Returns:
        CfgNode: model configurations
    """
    cfgs = []
    configs = args.config_file.split("#")
    weights_id = args.opts.index("MODEL.WEIGHTS") + 1
    weights = args.opts[weights_id].split("#")
    for i, conf in enumerate(configs):
        if len(conf) != 0:
            tmp_opts = copy.deepcopy(args.opts)
            tmp_opts[weights_id] = weights[i]
            cfg = get_cfg()
            get_perspective2d_cfg_defaults(cfg)
            cfg.merge_from_file(conf)
            cfg.merge_from_list(tmp_opts)
            cfg.freeze()
            cfgs.append(cfg)
    return cfgs


def get_parser():
    """
    Returns:
        Argument parser for command-line options
    """
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--dataset", required=True, help="dataset name")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def dict2str(info):
    text = ""
    for key in info.keys():
        text += f"{key}: {np.array2string(info[key], precision=2, floatmode='fixed')}\n"
    return text


def save_vis(demo, img, pred, gt, idx, info, output_folder):
    """save visualization of PerspectiveNet predictions and ground truth perspective field

    Args:
        demo (VisualizationDemo)
        img (np.ndarray): input image
        pred (dict): PerspectiveNet predictions
        gt (dict): ground truth labels
        idx (int): index of current data point
        info (dict): optimized ParamNet predictions
        output_folder (str): folder to output visualization
    """
    pred_vis = demo.draw(
        image=img,
        latimap=pred["pred_latitude_original"],
        gravity=pred["pred_gravity_original"],
        latimap_format=pred["pred_latitude_original_mode"],
    )

    gt_vis = demo.draw(
        image=img,
        latimap=gt["gt_latitude_original"],
        gravity=gt["gt_gravity_original"],
        latimap_format=gt["gt_latitude_original_mode"],
    )
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, str(idx).zfill(4) + "_2input.jpg"), img)
    pred_vis.save(os.path.join(output_folder, str(idx).zfill(4) + "_1pred"))
    gt_vis.save(os.path.join(output_folder, str(idx).zfill(4) + "_0gt"))


def get_r_p_from_rotation(gravity_world, R_world_from_cam):
    """convert camera rotation matrix to roll and pitch

    Args:
        gravity_world (tuple): gravity direction in the world frame
        R_world_from_cam (np.ndarray): camera rotation matrix

    Returns:
        float, float: camera roll and pitch
    """
    gravity_world = gravity_world / np.linalg.norm(gravity_world)
    gravity_world = gravity_world.astype(float)
    rot = R_world_from_cam.T
    # The convention here is that the camera's positive x axis points right, the positive y
    # axis points up, and the positive z axis points away from where the camera is looking.
    up_n = rot @ gravity_world
    # Uprightnet: Z upward, Y right, X backward
    up_n = np.array([up_n[2], up_n[0], up_n[1]])
    pitch = -math.asin(up_n[0])

    sin_roll = up_n[1] / math.cos(pitch)

    roll = math.asin(sin_roll)
    roll = -roll
    return roll, pitch


def vec_angle_err(pred, gt):
    """compute cosine error between two vector fields

    Args:
        pred (np.ndarray): predicted field
        gt (np.ndarray): ground truth field

    Returns:
        np.ndarray: cosine error between pred and gt
    """
    dot = torch.clamp(torch.sum(pred * gt, dim=0), -1, 1)
    angle = torch.acos(dot)
    return angle


def eval_by_idx(cfg, demo, dataset, idx):
    """evaluate image at idx

    Args:
        cfg (CfgNode): model configurations
        demo (VisualizationDemo)
        dataset (dict): dict containing metadata for every image in the dataset

    Returns:
        dict: dict of errors for image at idx
    """
    img_path = dataset[idx]["file_name"]
    img = read_image(img_path, format=cfg.INPUT.FORMAT)
    predictions = demo.run_on_image(img)
    if "mask_on" in dataset[idx].keys() and dataset[idx]["mask_on"]:
        mask = mask_util.decode(dataset[idx]["mask"]).astype(bool)
    else:
        mask = np.ones(img.shape[:2], dtype=bool)
    mat_up_err_rad = vec_angle_err(
        predictions["pred_gravity_original"],
        dataset[idx]["gt_gravity_original"].to(
            predictions["pred_gravity_original"].device
        ),
    )
    mat_up_err_deg = torch.rad2deg(mat_up_err_rad).detach().cpu().numpy()
    perc_up_err_less_5 = ((mat_up_err_deg[mask] <= 5).sum() / mask.sum()).item()
    avg_up_err_deg = mat_up_err_deg[mask].mean()
    med_up_err_deg = np.median(mat_up_err_deg[mask])
    if (
        predictions["pred_latitude_original_mode"] == "deg"
        and dataset[idx]["gt_latitude_original_mode"] == "deg"
    ):
        mat_lati_err_deg = torch.abs(
            predictions["pred_latitude_original"].detach().cpu()
            - dataset[idx]["gt_latitude_original"]
        )
        perc_lati_err_less_5 = ((mat_lati_err_deg[mask] <= 5).sum() / mask.sum()).item()
        avg_lati_err_deg = mat_lati_err_deg.numpy()[mask].mean()
        med_lati_err_deg = np.median(mat_lati_err_deg.numpy()[mask])
    else:
        raise NotImplementedError
    rtn = {
        "pred_gravity_original": predictions["pred_gravity_original"].cpu().numpy(),
        "pred_latitude_original": predictions["pred_latitude_original"].cpu().numpy(),
        "pred_latitude_original_mode": predictions["pred_latitude_original_mode"],
        "avg_up_err_deg": avg_up_err_deg,
        "med_up_err_deg": med_up_err_deg,
        "avg_lati_err_deg": avg_lati_err_deg,
        "med_lati_err_deg": med_lati_err_deg,
        "perc_up_err_less_5": perc_up_err_less_5,
        "perc_lati_err_less_5": perc_lati_err_less_5,
    }
    # save_vis(demo, img, predictions, dataset[idx], f"{str(math.floor(avg_up_err_deg)).zfill(3)}_" + str(idx), rtn, output_folder='/Pool1/users/jinlinyi/public_html_shared/densefields/v06_ours_objectron_crop_mask')
    return rtn


def eval_by_list(cfg, demo, dataset, idx_list, return_dict, multiprocessing=False):
    """evaluate images at each index in idx_list from dataset

    Args:
        cfg (CfgNode): model configurations
        demo (VisualizationDemo)
        dataset (dict): dict containing metadata for every image in the dataset
        idx_list (list[int]): list of indexes to evaluate at
        return_dict (dict): dict containing errors for each image
        multiprocessing (bool, optional): Defaults to False.
    """
    if multiprocessing:
        loop = idx_list
    else:
        loop = tqdm(idx_list)
    for idx in loop:
        rtn = eval_by_idx(cfg, demo, dataset, idx)
        return_dict[idx] = rtn
        # if not multiprocessing:
        #     loop.set_description('Optimizing (loss %.4f)' % rtn['loss'])


class dataset_list(MapDataset):
    def __init__(self, cfg_list, dataset):
        dataloaders = []
        for cfg in cfg_list:
            dataloader = build_detection_test_loader(
                cfg,
                dataset,
                mapper=PerspectiveMapper(cfg, False, dataset_names=(dataset,)),
            )
            dataloaders.append(dataloader.dataset)
        l = len(dataloaders[0])
        for dataloader in dataloaders:
            assert l == len(dataloader)
        self.dataloaders = dataloaders

    def __getitem__(self, idx):
        rtn = {}
        for dataloader in self.dataloaders:
            item = dataloader[idx]
            for key in item.keys():
                if key in rtn.keys():
                    # assert rtn[key] == item[key]
                    continue
                else:
                    rtn[key] = item[key]
        return rtn

    def __len__(self):
        return len(self.dataloaders[0])


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg_list = setup_cfg(args)

    demo = VisualizationDemo(cfg_list=cfg_list)

    cfg = cfg_list[0]
    dataloader = dataset_list(cfg_list, args.dataset)
    return_dict = {}

    np.random.seed(2022)
    idx_list = np.arange(len(dataloader))

    eval_by_list(cfg, demo, dataloader, idx_list, return_dict)



    metrics = np.array([
        ['med(up_avgs)', 'up mean'],
        ['med(up_meds)', 'up median'],
        ['pgp_up<5', 'up %<5'],
        ['med(lat_avgs)', 'lati mean'],
        ['med(lat_meds)', 'lati median'],
        ['pgp_lat<5', 'lati %<5'],
    ])
    up_avgs = np.array([o['avg_up_err_deg'] for o in return_dict.values()])
    lat_avgs = np.array([o['avg_lati_err_deg'] for o in return_dict.values()])
    up_meds =np.array([o['med_up_err_deg'] for o in return_dict.values()])
    lat_meds =np.array([o['med_lati_err_deg'] for o in return_dict.values()])

    eval_metrics = {}
    if 'med(up_avgs)' in metrics[:,0]:
        eval_metrics['med(up_avgs)']= np.median(up_avgs) 
    if 'med(lat_avgs)' in metrics[:,0]:
        eval_metrics['med(lat_avgs)'] = np.median(lat_avgs) 
    if 'pgp_up<5' in metrics[:,0]:
        eval_metrics['pgp_up<5'] = np.average(np.array([o['perc_up_err_less_5'] for o in return_dict.values()])) * 100
    if 'pgp_lat<5' in metrics[:,0]:
        eval_metrics['pgp_lat<5'] = np.average(np.array([o['perc_lati_err_less_5'] for o in return_dict.values()])) * 100
    if 'med(up_meds)' in metrics[:,0]:
        eval_metrics['med(up_meds)']= np.median(up_meds) 
    if 'med(lat_meds)' in metrics[:,0]:
        eval_metrics['med(lat_meds)'] = np.median(lat_meds) 
    for m in metrics:
        print(f"{m[1]}: {eval_metrics[m[0]] :.2f}")
