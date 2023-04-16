# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
from torch.nn import functional as F
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
import detectron2.utils.comm as comm

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.logger import setup_logger, create_small_table


class LatitudeEvaluator:
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        cfg,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)

        if not self._logger.isEnabledFor(logging.INFO):
            setup_logger(name=__name__)
        self._cpu_device = torch.device("cpu")

        if cfg.MODEL.META_ARCHITECTURE == 'PerspectiveNet':
            self.loss_type = "classification"
            self.ignore_value = cfg.MODEL.FPN_LATITUDE_HEAD.IGNORE_VALUE
        elif cfg.MODEL.META_ARCHITECTURE == 'PersFormer':
            self.loss_type = cfg.MODEL.LATITUDE_DECODER.LOSS_TYPE
            self.ignore_value = cfg.MODEL.LATITUDE_DECODER.IGNORE_VALUE
        else:
            raise NotImplementedError

    def process(self, input, output):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        ret = {}
        if "pred_latitude_original" in output.keys():
            pred = output['pred_latitude']
            gt = input['gt_latitude'].to(pred.device)
            # pred = pred[:, :gt.shape[0], :gt.shape[1]]
            if self.loss_type == 'regression':
                loss = F.l1_loss(
                    pred, gt, reduction="mean",
                )
            elif self.loss_type == 'classification':
                loss = F.cross_entropy(
                    pred.unsqueeze(0), gt.unsqueeze(0), reduction="mean", ignore_index=self.ignore_value
                )
            else:
                raise NotImplementedError
            if input['gt_latitude_original_mode'] == 'rad':
                gt_lati_ori = torch.rad2deg(input['gt_latitude_original']).to(output['pred_latitude_original'].device)
            else:
                gt_lati_ori = input['gt_latitude_original'].to(output['pred_latitude_original'].device)
            if output['pred_latitude_original_mode'] == 'rad':
                pred_lati_ori = torch.rad2deg(output['pred_latitude_original'])
            else:
                pred_lati_ori = output['pred_latitude_original']
            if 'mask_on' in input.keys() and input['mask_on']:
                mask = mask_util.decode(input['mask']).astype(bool)
            else:
                mask = np.ones((input['height'], input['width']), dtype=bool)
            mask = torch.tensor(mask)
            ret['latitude_err_mean'] = torch.mean(torch.abs(pred_lati_ori - gt_lati_ori).to(self._cpu_device)[mask]).numpy()
            # ret['latitude_err_median'] = torch.median(torch.abs(pred_lati_ori - gt_lati_ori).to(self._cpu_device)[mask]).numpy()
            
            ret['latitude_loss'] = loss.to(self._cpu_device).numpy()
        return ret

    def evaluate(self, predictions):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        res = {}
        res["latitude_Loss"] = np.average([e['latitude_loss'] for e in predictions])
        res["latitude_err_mean"] = np.average([e['latitude_err_mean'] for e in predictions])
        res["latitude_err_median"] = np.median([e['latitude_err_mean'] for e in predictions])
        self._logger.info("latitude: \n"+create_small_table(res))  
        results = {"latitude": res}
        return results