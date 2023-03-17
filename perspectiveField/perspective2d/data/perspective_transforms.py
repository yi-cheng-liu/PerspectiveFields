import copy
import json
import logging
import numpy as np
import os
import torch
import pickle
import random
import cv2
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import (
    BitMasks,
    Boxes, 
    BoxMode, 
    Instances, 
    PolygonMasks,
    polygons_to_bitmask,
)

from PIL import Image
import torchvision.transforms as transforms
from perspective2d.utils.panocam import PanoCam
from sklearn.preprocessing import normalize
from detectron2.data import transforms as T
import albumentations as A

from .latitude_transform import LatitudeTransform
from .gravity_transform import GravityTransform
from ..utils import general_vfov
import scipy.optimize 


__all__ = ["PerspectiveMapper"]


def fun(h, *args):
    focal, d_cx, d_cy, target_cos_FoV = args
    
    p_sqr = (focal/h) ** 2 + d_cx ** 2 + (d_cy + 0.5) ** 2
    q_sqr = (focal/h) ** 2 + d_cx ** 2 + (d_cy - 0.5) ** 2
    cos_FoV = (p_sqr + q_sqr - 1) / 2 / np.sqrt(p_sqr) / np.sqrt(q_sqr)
    return cos_FoV - target_cos_FoV



class PerspectiveMapper:
    """
    A callable which takes a dict produced by the detection dataset, and applies transformations,
    including image resizing and flipping. The transformation parameters are parsed from cfg file
    and depending on the is_train condition.
    Note that for our existing models, mean/std normalization is done by the model instead of here.

    According to cfg.MODEL.PIXEL_MEAN, the output image is 0-255
    """

    def __init__(self, cfg, is_train=True, dataset_names=None):
        # self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        self.cfg = cfg
        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.online_crop    = cfg.INPUT.ONLINE_CROP
        self.resize         = cfg.DATALOADER.RESIZE
        self.gravity_on     = cfg.MODEL.GRAVITY_ON
        self.latitude_on    = cfg.MODEL.LATITUDE_ON
        self.height_on      = cfg.MODEL.HEIGHT_ON
        self.center_on      = cfg.MODEL.CENTER_ON
        self.focal_on       = cfg.MODEL.RECOVER_PP
        self.pp_on          = cfg.MODEL.RECOVER_PP
        self.debug_on       = cfg.DEBUG_ON
        self.overfit_on        = cfg.OVERFIT_ON
            
        # fmt: on
        self.is_train = is_train
        self.init_aug()
        self.init_color_aug()

        assert dataset_names is not None
        assert self.cfg.MODEL.PIXEL_MEAN[0] > 10 # ensure output image is 0-255
        if self.gravity_on:
            self.gravity_transform = GravityTransform(cfg, is_train)
        if self.latitude_on:
            self.latitude_transform = LatitudeTransform(cfg, is_train)
        if self.height_on:
            self.height_transform = HeightTransform(cfg, is_train)

        self.init_pseudo_uniform_counter()

    def init_pseudo_uniform_counter(self):
        self.rel_focal_range = (0.5, 2, 61)
        self.sampled_rel_f_count = np.zeros(self.rel_focal_range[2])
        self.general_vfov_range = (20, 100, 81)
        self.sampled_general_vfov_count = np.zeros(self.general_vfov_range[2])
        
    def sample_general_vfov(self, max_vfov):
        
        max_bin_id = self.encode_param_bin(max_vfov, self.general_vfov_range)
        selected_bin = np.random.choice(np.where(np.min(self.sampled_general_vfov_count[:max_bin_id]) == self.sampled_general_vfov_count)[0], 1)[0]

        bin_size = (self.general_vfov_range[1] - self.general_vfov_range[0]) / self.general_vfov_range[2]
        self.sampled_general_vfov_count[selected_bin] += 1
        vfov = min(max_vfov, self.decode_param_bin(selected_bin, self.general_vfov_range) + np.random.uniform(-bin_size / 2, bin_size / 2))
        return vfov.item()

    def sample_rel_f(self, min_rel_f):
        min_bin_id = self.encode_param_bin(min_rel_f, self.rel_focal_range)
        selected_bin = np.random.choice(np.where(np.min(self.sampled_rel_f_count[min_bin_id:]) == self.sampled_rel_f_count)[0], 1)[0]

        bin_size = (self.rel_focal_range[1] - self.rel_focal_range[0]) / self.rel_focal_range[2]
        self.sampled_rel_f_count[selected_bin] += 1
        rel_f = max(min_rel_f, self.decode_param_bin(selected_bin, self.rel_focal_range) + np.random.uniform(-bin_size / 2, bin_size / 2))
        return rel_f

    def encode_param_bin(self, value, param_range):
        """
        value to bin id. 
        """
        bin_size = (param_range[1] - param_range[0]) / param_range[2]
        boundaries = torch.arange(param_range[0], param_range[1], bin_size)[1:]
        binmap = torch.bucketize(value, boundaries)
        return binmap.type(torch.LongTensor)


    def decode_param_bin(self, bin_id, param_range):
        """
        decode bin to value
        """
        bin_size = (param_range[1] - param_range[0]) / param_range[2]
        bin_centers = torch.arange(param_range[0], param_range[1], bin_size) + bin_size / 2
        value = bin_centers[bin_id]
        return value



    def __call__(self, dataset_dict):
        """
        Transform the dataset_dict according to the configured transformations.
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a new dict that's going to be processed by the model.
                It currently does the following:
                1. Read the image from "file_name"
                2. Transform the image and annotations
                3. Prepare the annotations to :class:`Instances`
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        assert os.path.exists(dataset_dict["file_name"]), dataset_dict["file_name"]
        img = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, img)
        if self.focal_on:
            if self.is_train and not self.overfit_on:
                assert 'focal_length' not in dataset_dict.keys()
                img_height = img.shape[0]
                focal_length = img_height / 2 / np.tan(np.radians(dataset_dict['vfov'] / 2))
            else:
                assert 'vfov' not in dataset_dict.keys() # should test on cropped images?
                focal_length = dataset_dict['focal_length']
        else:
            focal_length = None

        if self.latitude_on:
            gt_latitude_original, gt_latitude_original_mode = self.latitude_transform.get_input_label(dataset_dict)
        else:
            gt_latitude_original, gt_latitude_original_mode = None, None

        if self.gravity_on:
            gt_absvvp, gt_gfield_original = self.gravity_transform.get_input_label(dataset_dict)
        else:
            gt_absvvp, gt_gfield_original = None, None
 
        if self.height_on:
            gt_height, gt_height_original, height_validmap = self.height_transform.get_input_label(dataset_dict)
        else:
            gt_height, gt_height_original, height_validmap = None, None, None

        if self.pp_on and (not self.is_train or self.overfit_on):
            pp = np.array(dataset_dict['center'])
        else:
            assert 'pp' not in dataset_dict.keys()
            pp = np.array(img.shape[:2][::-1]) / 2
        if self.debug_on:
            if not self.overfit_on:
                dataset_dict['img_center_original'] = cv2.circle(img.copy(), pp.astype(int), radius=10, color=(0, 0, 255), thickness=-1)
            else:
                original_path = os.path.join('/home/jinlinyi/tmp/google_street_view_191210/manhattan', *(dataset_dict['file_name'].split('/')[-1].split('-', 1)))
                if not os.path.exists(original_path):
                    import pdb;pdb.set_trace()
                    pass
                original_rgb = cv2.imread(original_path)[:,:,::-1]
                original_rgb_center = np.array(original_rgb.shape[:2][::-1]) / 2
                dataset_dict['img_center_original'] = cv2.circle(original_rgb.copy(), original_rgb_center.astype(int), radius=10, color=(0, 0, 255), thickness=-1)

        if self.cfg.DATALOADER.AUGMENTATION_FUN == 'with_up_field':
            img, gt_latitude_aug, gt_gfield_aug = self.augmentation_with_up_field(img, gt_latitude_original, gt_gfield_original)
            pp_aug, focal_length_aug = (0,0), 0
        elif self.cfg.DATALOADER.AUGMENTATION_FUN == 'default' or not self.is_train or self.overfit_on:
            img, gt_latitude_aug, gt_absvvp_aug, gt_height_aug, height_validmap_aug, pp_aug, focal_length_aug = self.augmentation(img, gt_latitude_original, gt_absvvp, gt_height, height_validmap, pp, focal_length)
        elif self.cfg.DATALOADER.AUGMENTATION_FUN == 'uniform_vfov_crop_resize':
            img, gt_latitude_aug, gt_absvvp_aug, gt_height_aug, height_validmap_aug, pp_aug, focal_length_aug = self.uniform_vfov_crop_resize(img, gt_latitude_original, gt_absvvp, gt_height, height_validmap, pp, focal_length)
        elif self.cfg.DATALOADER.AUGMENTATION_FUN == 'uniform_rel_f_crop_resize':
            img, gt_latitude_aug, gt_absvvp_aug, gt_height_aug, height_validmap_aug, pp_aug, focal_length_aug = self.uniform_rel_f_crop_resize(img, gt_latitude_original, gt_absvvp, gt_height, height_validmap, pp, focal_length)
        else:
            raise NotImplementedError
        rel_focal = focal_length_aug / img.shape[0] if self.focal_on else None
        pp_cropped = np.array(img.shape[:2][::-1]) / 2
        rel_pp = (pp_aug - pp_cropped) / np.array(img.shape[:2][::-1])

        if self.gravity_on:
            if self.cfg.DATALOADER.AUGMENTATION_FUN == 'with_up_field':
                dataset_dict['gt_gravity'] = self.gravity_transform.to_tensor_from_field(im_h=img.shape[0], im_w=img.shape[1], gfield=gt_gfield_aug)    
            else:
                dataset_dict['gt_gravity'] = self.gravity_transform.to_tensor(im_h=img.shape[0], im_w=img.shape[1], absvvp=gt_absvvp_aug)

        if self.latitude_on:
            dataset_dict['gt_latitude'] = self.latitude_transform.to_tensor(gt_latitude_aug)

        if self.height_on:
            dataset_dict['gt_height'], dataset_dict['height_validmap'] = self.height_transform.to_tensor(gt_height_aug, height_validmap_aug)

        img = torch.as_tensor(img.transpose(2, 0, 1).astype("float32"))
        dataset_dict['image'] = img
        dataset_dict['pp'] = pp_aug
        dataset_dict['rel_pp'] = rel_pp
        dataset_dict['rel_cx'] = rel_pp[0]
        dataset_dict['rel_cy'] = rel_pp[1]
        dataset_dict['rel_focal'] = rel_focal
        if rel_focal is not None:
            dataset_dict['general_vfov'] = general_vfov(rel_pp[0], rel_pp[1], 1, rel_focal, degree=True)
        
        if not self.is_train:
            if self.gravity_on:
                # Get gfield for original image
                gt_gfield_original = torch.as_tensor(gt_gfield_original.transpose(2, 0, 1).astype("float32"))
                dataset_dict['gt_gravity_original'] = gt_gfield_original
            if self.latitude_on:
                gt_latitude_original = torch.as_tensor(gt_latitude_original.astype("float32"))
                dataset_dict['gt_latitude_original'] = gt_latitude_original
                dataset_dict['gt_latitude_original_mode'] = gt_latitude_original_mode
                
            if self.height_on:
                gt_height_original = torch.as_tensor(gt_height_original.astype("float32"))
                dataset_dict['gt_height_original'] = gt_height_original
        
        return dataset_dict

    def augmentation_with_up_field(self, img, latimap, gfield):
        img = self.color_aug(image=img)['image']
        aug_list = []
        if self.is_train:
            aug_list.append(
                A.RandomResizedCrop(height=self.resize[0], width=self.resize[1])
            )
        else:
            aug_list.append(
                A.Resize(self.resize[0], self.resize[1])
            )

        aug = A.Compose(
            aug_list, 
            keypoint_params=A.KeypointParams(
                format='xy', remove_invisible=False),
            additional_targets={
                'latimap': 'mask',
                'upmap': 'mask',
                'corner': 'keypoints',
            }
        )
        h, w, _ = img.shape
        transformed = aug(
            image=img, 
            latimap=latimap,
            keypoints=[(0,0)],
            upmap=gfield,
            corner=[(0, 0), (w, 0), (0, h), (w, h)],
        )
        img_aug = transformed['image']
        transformed['corner'] = np.array(transformed['corner'])
        transformed_h = transformed['corner'][2,1] - transformed['corner'][0,1]
        transformed_w = transformed['corner'][1,0] - transformed['corner'][0,0]


        scale = np.array([[transformed_w / w, transformed_h / h]])
        gfield_aug = transformed['upmap'] * scale
        gfield_aug = normalize(gfield_aug.reshape(-1, 2)).reshape(transformed['upmap'].shape)
        latimap_aug = transformed['latimap']
        return img_aug, latimap_aug, gfield_aug
        


    def augmentation(self, img, latimap, absvvp, heightmap, height_validmap, pp, focal_length):
        # im_w, im_h, _ = img.shape
        # absvvp_center = absvvp[:2] - np.array([im_w / 2 - 0.5, im_h / 2 - 0.5]) if absvvp is not None else None
        # transformed = self.aug(
        #     image=img, 
        #     latimap=latimap if latimap is not None else np.ones_like(img),
        #     keypoints=[tuple(absvvp_center)] if absvvp is not None else [(0,0)],
        #     heightmap=heightmap if heightmap is not None else np.zeros_like(img),
        #     height_validmap=height_validmap if height_validmap is not None else np.zeros_like(img),
        # )
        # img_aug = transformed['image']
        # im_h_aug, im_w_aug, _ = img_aug.shape
        # latimap_aug = transformed['latimap'] if latimap is not None else None
        # absvvp_aug = np.array([transformed['keypoints'][0][0], transformed['keypoints'][0][1], absvvp[2]]) + np.array([im_w_aug / 2 - 0.5, im_h_aug / 2 - 0.5, 0]) if absvvp is not None else None
        # heightmap_aug = transformed['heightmap'] if heightmap is not None else None
        # height_validmap_aug = transformed['height_validmap'] if height_validmap is not None else None
        # return img_aug, latimap_aug, absvvp_aug, heightmap_aug, height_validmap_aug
        h, w, _ = img.shape
        transformed = self.aug(
            image=img, 
            latimap=latimap if latimap is not None else np.ones_like(img),
            keypoints=[tuple(absvvp[:2])] if absvvp is not None else [(0,0)],
            heightmap=heightmap if heightmap is not None else np.zeros_like(img),
            height_validmap=height_validmap if height_validmap is not None else np.zeros_like(img),
            pp=[tuple(pp)], # x, y
            corner=[(0, 0), (w, 0), (0, h), (w, h)],
        )
        img_aug = transformed['image']
        latimap_aug = transformed['latimap'] if latimap is not None else None
        absvvp_aug = np.array([transformed['keypoints'][0][0], transformed['keypoints'][0][1], absvvp[2]]) if absvvp is not None else None
        heightmap_aug = transformed['heightmap'] if heightmap is not None else None
        height_validmap_aug = transformed['height_validmap'] if height_validmap is not None else None
        pp_aug = np.array(transformed['pp'][0]) # [cx, cy]
        if focal_length is not None:
            transformed['corner'] = np.array(transformed['corner'])
            focal_length_aug = focal_length * (transformed['corner'][2] - transformed['corner'][0])[1] / h
        else:
            focal_length_aug = None
        return img_aug, latimap_aug, absvvp_aug, heightmap_aug, height_validmap_aug, pp_aug, focal_length_aug
        

    def uniform_vfov_crop_resize(self, image, latimap, absvvp, heightmap, height_validmap, pp, focal_length):
        image = self.color_aug(image=image)['image']
        H, W, _ = image.shape
        assert H == W
        rel_cx = np.random.uniform(-0.4, 0.4)
        rel_cy = np.random.uniform(-0.4, 0.4)
        original_size = H
        crop_size_max = min(W / (1-2*rel_cx), W / (1 + 2*rel_cx), H / (1-2*rel_cy), H / (1 + 2*rel_cy))
        
        max_vfov = general_vfov(rel_cx, rel_cy, 1, focal_length / crop_size_max, degree=True)
        vfov = self.sample_general_vfov(max_vfov)
        h = scipy.optimize.fsolve(fun, crop_size_max, args=(focal_length, rel_cx, rel_cy, np.cos(np.radians(vfov))))[0]
        h = int(h)
        w = h
        cx = rel_cx * w + 0.5 * W
        cy = rel_cy * h + 0.5 * H
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        max_crop_size = w
        transformed = {
            'image': image[y:y+max_crop_size, x:x+max_crop_size, :],
            'latimap': latimap[y:y+max_crop_size, x:x+max_crop_size] if latimap is not None else None,
            'absvvp': np.array(absvvp) - np.array([x, y, 0]) if absvvp is not None else np.array([0,0,0]),
            'heightmap': heightmap[y:y+max_crop_size, x:x+max_crop_size]  if heightmap is not None else None,
            'height_validmap': height_validmap[y:y+max_crop_size, x:x+max_crop_size] if height_validmap is not None else None,
            'pp': np.array(pp) - np.array([x, y]),
            'focal_length': focal_length,
        }
        # scale
        assert self.resize[0] == self.resize[1]
        scale_factor = self.resize[0] / w
        img_aug = cv2.resize(transformed['image'], self.resize)
        latimap_aug = cv2.resize(transformed['latimap'], self.resize) if latimap is not None else None
        absvvp_aug = transformed['absvvp'] * np.array([scale_factor, scale_factor, 1])
        heightmap_aug = cv2.resize(transformed['heightmap'], self.resize)  if heightmap is not None else None
        height_validmap_aug = cv2.resize(transformed['height_validmap'], self.resize) if height_validmap is not None else None
        pp_aug = transformed['pp'] * scale_factor
        focal_length_aug = transformed['focal_length'] * scale_factor
        return img_aug, latimap_aug, absvvp_aug, heightmap_aug, height_validmap_aug, pp_aug, focal_length_aug

    def uniform_rel_f_crop_resize(self, image, latimap, absvvp, heightmap, height_validmap, pp, focal_length):
        image = self.color_aug(image=image)['image']
        H, W, _ = image.shape
        assert H == W
        rel_cx = np.random.uniform(-0.4, 0.4)
        rel_cy = np.random.uniform(-0.4, 0.4)
        original_size = H
        crop_size_max = min(W / (1-2*rel_cx), W / (1 + 2*rel_cx), H / (1-2*rel_cy), H / (1 + 2*rel_cy))
        min_rel_f = focal_length / crop_size_max
        rel_f = self.sample_rel_f(min_rel_f)
        # rel_f = max(np.random.uniform(min_rel_f, 2), min_rel_f)
        # rel_f = focal_length / crop_size_max

        h = int(focal_length / rel_f)
        w = h
        cx = rel_cx * w + 0.5 * W
        cy = rel_cy * h + 0.5 * H
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        max_crop_size = w
        transformed = {
            'image': image[y:y+max_crop_size, x:x+max_crop_size, :],
            'latimap': latimap[y:y+max_crop_size, x:x+max_crop_size],
            'absvvp': np.array(absvvp) - np.array([x, y, 0]),
            'heightmap': heightmap[y:y+max_crop_size, x:x+max_crop_size]  if heightmap is not None else None,
            'height_validmap': height_validmap[y:y+max_crop_size, x:x+max_crop_size] if height_validmap is not None else None,
            'pp': np.array(pp) - np.array([x, y]),
            'focal_length': focal_length,
        }
        # scale
        assert self.resize[0] == self.resize[1]
        scale_factor = self.resize[0] / w
        img_aug = cv2.resize(transformed['image'], self.resize)
        latimap_aug = cv2.resize(transformed['latimap'], self.resize)
        absvvp_aug = transformed['absvvp'] * np.array([scale_factor, scale_factor, 1])
        heightmap_aug = cv2.resize(transformed['heightmap'], self.resize)  if heightmap is not None else None
        height_validmap_aug = cv2.resize(transformed['height_validmap'], self.resize) if height_validmap is not None else None
        pp_aug = transformed['pp'] * scale_factor
        focal_length_aug = transformed['focal_length'] * scale_factor
        return img_aug, latimap_aug, absvvp_aug, heightmap_aug, height_validmap_aug, pp_aug, focal_length_aug

    def init_color_aug(self):
        if self.is_train and not self.overfit_on:
            aug_list = [
                    A.RandomBrightnessContrast(p=0.2),
                    A.HueSaturationValue(p=0.2),
                    A.ToGray(p=0.2),
                    A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
                    
                    A.OneOf([
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
                ]
            self.color_aug = A.Compose(
                aug_list, 
            )
        else:
            self.color_aug = A.Compose([])

    def init_aug(self):
        if self.is_train and not self.overfit_on:
            aug_list = [
                    A.RandomBrightnessContrast(p=0.2),
                    A.HueSaturationValue(p=0.2),
                    A.ToGray(p=0.2),
                    A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
                    
                    A.OneOf([
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
                ]
            if self.cfg.MODEL.RECOVER_RPF:
                if self.cfg.MODEL.RECOVER_PP:
                    aug_list.append(
                        A.RandomResizedCrop(height=self.resize[0], width=self.resize[1], ratio=(1.0,1.0))
                    )
                else:
                    aug_list.append(
                        A.Resize(self.resize[0], self.resize[1])
                    )
            elif self.cfg.DATALOADER.NO_GEOMETRY_AUG: 
                aug_list.append(
                    A.Resize(self.resize[0], self.resize[1])
                )
            else:
                aug_list.extend(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.2),
                        A.RandomRotate90(p=0.5),
                        A.RandomResizedCrop(height=self.resize[0], width=self.resize[1]),
                    ]
                )

            self.aug = A.Compose(
                aug_list, 
                keypoint_params=A.KeypointParams(
                    format='xy', remove_invisible=False),
                additional_targets={
                    'latimap': 'mask',
                    'heightmap': 'mask',
                    'height_validmap': 'mask',
                    'pp': 'keypoints',
                    'corner': 'keypoints',
                }
            )
        else:
            self.aug = A.Compose([
                A.Resize(self.resize[0], self.resize[1])
                ], 
                keypoint_params=A.KeypointParams(
                    format='xy', remove_invisible=False),
                additional_targets={
                    'latimap': 'mask',
                    'heightmap': 'mask',
                    'height_validmap': 'mask',
                    'pp': 'keypoints',
                    'corner': 'keypoints',
                }
            )


def vis_surface_normal(normal):
    
    normal_vis = (normal + 1.0) / 2.0 * 255.0
    normal_vis = normal_vis.astype(np.uint8)
    return normal_vis


if __name__=='__main__':
    from perspective2d.data import tmp_data_dir
    src_path = '/home/jinlinyi/datasets/gsv_test_crop_uniform.tar'
    tmp_data_dir.TempSSDPath(src_path, 'gsv_test_crop_uniform', job_dir='log', logging=True)
    from detectron2.config import get_cfg
    from perspective2d.config import get_perspective2d_cfg_defaults
    from detectron2.data import DatasetCatalog
    import cv2
    import perspective2d.data
    from tqdm import tqdm
    import seaborn as sns
    import matplotlib.pyplot as plt
    from panocam import PanoCam

    from perspective2d.utils import decode_bin, draw_vector_field, draw_up_field

    random.seed(2021)

    debug_folder = './debug'
    os.makedirs(debug_folder, exist_ok=True)
    # Load cfg
    cfg = get_cfg()
    get_perspective2d_cfg_defaults(cfg)
    cfg.merge_from_file('./configs/recover_rpf/e05_recover_rpgfpp_cls_pretrain.yaml')
    dataloader = PerspectiveMapper(cfg, is_train=True, dataset_names=cfg.DATASETS.TRAIN)
    dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
    rel_pps, rel_focals, general_vfovs, original_vfovs, d_lats = [], [], [], [], []
    for i in tqdm(range(1000)):
        d = dataloader(dataset_dicts[i])
        rel_pps.append(d['rel_pp'])
        rel_focals.append(d['rel_focal'])
        d_lats.append(torch.rad2deg(torch.arcsin(d['gt_latitude'][0, 0, 160])) - torch.rad2deg(torch.arcsin(d['gt_latitude'][0, 319, 160])))
        # d['general_vfov']
        
        # import pdb;pdb.set_trace()
        general_vfovs.append(d['general_vfov'])
        # original_vfovs.append(d['vfov'])
        if np.abs(d_lats[-1] - general_vfovs[-1]) > 30:
            import pdb;pdb.set_trace()
        # up_general = torch.as_tensor(PanoCam.get_up_general(
        #     d['rel_focal'], 
        #     d['image'].shape[2],
        #     d['image'].shape[1],
        #     np.radians(d['pitch']),
        #     np.radians(d['roll']),
        #     d['rel_cx'],
        #     d['rel_cy'],
        # ).transpose(2, 0, 1).astype("float32"))
        # lat_general = torch.sin(torch.deg2rad(torch.as_tensor(PanoCam.get_lat_general(
        #     d['rel_focal'], 
        #     d['image'].shape[2],
        #     d['image'].shape[1],
        #     np.radians(d['pitch']),
        #     np.radians(d['roll']),
        #     d['rel_cx'],
        #     d['rel_cy'],
        # ))))

        # pred = draw_up_field(
        #     img_rgb=d['image'].numpy().transpose(1,2,0).astype(np.uint8)[:,:,::-1], 
        #     vector_field=d['gt_gravity'],
        #     color=(0,1,0)
        # )
        # gt = draw_up_field(
        #     img_rgb=d['image'].numpy().transpose(1,2,0).astype(np.uint8)[:,:,::-1], 
        #     vector_field=up_general,
        #     color=(0,1,0)
        # )
        # cv2.imwrite("debug_pred.png", pred)
        # cv2.imwrite("debug_gt.png", gt)
        # import pdb;pdb.set_trace()
        # up_err = torch.norm(up_general - d['gt_gravity'])
        # lat_err = torch.max(torch.abs(torch.rad2deg(torch.arcsin(d['gt_latitude'])) - torch.rad2deg(torch.arcsin(lat_general))))
        # if lat_err >  1 or up_err > 1:
        #     import pdb;pdb.set_trace()
    rel_pps = np.array(rel_pps)
    rel_focals = np.array(rel_focals)
    general_vfovs = np.array(general_vfovs)
    # original_vfovs = np.array(original_vfovs)
    d_lats = np.array(d_lats)

    sns.distplot(rel_focals, hist = True, kde = True,
                 kde_kws = {'linewidth': 3})
    plt.xlabel("Rel_focal")
    plt.title("Rel_focal from training data")
    plt.savefig('debug/density_Rel_focal.png')
    plt.close()

    sns.distplot(rel_pps[:, 0], hist = True, kde = True,
                 kde_kws = {'linewidth': 3})
    plt.xlabel("Relative cx w.r.t. image size")
    plt.title("Principal point location from training data")
    plt.savefig('debug/density_cx.png')
    plt.close()


    sns.distplot(rel_pps[:, 1], hist = True, kde = True,
                 kde_kws = {'linewidth': 3})
    plt.xlabel("Relative cy w.r.t. image size")
    plt.title("Principal point location from training data")
    plt.savefig('debug/density_cy.png')
    plt.close()

    sns.distplot(general_vfovs, hist = True, kde = True,
                 kde_kws = {'linewidth': 3})
    plt.xlabel("general_vfov")
    plt.title("general_vfov from training data")
    plt.savefig('debug/density_general_vfov.png')
    plt.close()

    # sns.distplot(original_vfovs, hist = True, kde = True,
    #              kde_kws = {'linewidth': 3})
    # plt.xlabel("original_vfovs")
    # plt.title("original_vfovs from training data")
    # plt.savefig('debug/density_original_vfovs.png')
    # plt.close()

    plt.plot(d_lats, general_vfovs, '.')
    plt.xlabel("$\delta$ lats")
    plt.ylabel("general_vfovs")
    plt.title("vfov_vs_dlats")
    plt.savefig('debug/vfov_vs_dlats.png')
    plt.close()
    import pdb;pdb.set_trace()
        # bin_map = d['gt_gfield']
        # vector_field = decode_bin(bin_map, cfg.MODEL.GRAVITY_HEAD.NUM_CLASSES)
        # cv2.imwrite(os.path.join(debug_folder, f'{i}_img.png'), 
        #         d['image'].numpy().transpose(1,2,0)*255)
        # zero = torch.zeros((1, vector_field.size(1), vector_field.size(2)))
        # normalmap = vis_surface_normal(torch.cat((vector_field,zero),0).numpy())
        # cv2.imwrite(os.path.join(debug_folder, f'{i}_gfield_bined.png'), 
        # np.array(normalmap).transpose(1,2,0))
        