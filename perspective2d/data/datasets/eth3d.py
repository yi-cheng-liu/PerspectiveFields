import csv
import json
import logging
import os

import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

logger = logging.getLogger(__name__)

__all__ = ["load_eth3d_json"]

def load_eth3d_json(json_file, img_root):
    with open(json_file) as f:
        summary = json.load(f)
    
    for idx, item in enumerate(summary["data"]):
        item["file_name"] = os.path.join(
            img_root, item["file_name"]
        )
        if "dataset" not in item.keys():
            item["dataset"] = "livingroom"
        item["mask_on"] = False

        if "latitude_file_name" in item.keys():
            item["latitude_file_name"] = os.path.join(
                img_root, item["latitude_file_name"]
            )
        if "gravity_file_name" in item.keys():
            item["gravity_file_name"] = os.path.join(
                img_root, item["gravity_file_name"]
            )
    breakpoint()
        
    logger.info(f"{os.path.basename(json_file)}: Loaded {len(summary['data'])} entries.")
    return summary["data"]