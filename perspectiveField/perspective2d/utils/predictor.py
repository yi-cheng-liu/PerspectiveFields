import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import torch.nn.functional as F
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import transforms as T

from perspective2d.utils import draw_vector_field, draw_latitude_field, draw_up_field, general_vfov_to_focal
from perspective2d.utils.visualizer import VisualizerPerspective
from perspective2d.utils.param_opt import predict_rpfpp
from .panocam import PanoCam


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        if x.is_cuda:
            return x.detach().cpu().numpy()
        else:
            return x.detach().numpy()
    else:
        return np.array(x)


class VisualizationDemo(object):
    def __init__(self, cfg=None, cfg_list=None, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        if cfg is not None:
            self.multi_cfg = False
            self.parallel = parallel
            self.predictor = DefaultPredictor(cfg)
            self.predictor.aug = T.Resize(cfg.DATALOADER.RESIZE)
            self.gravity_on = cfg.MODEL.GRAVITY_ON
            self.center_on = cfg.MODEL.CENTER_ON
            self.latitude_on = cfg.MODEL.LATITUDE_ON
            self.device = self.predictor.model.device
        elif cfg_list is not None:
            self.multi_cfg = True
            self.predictors = [DefaultPredictor(cfg_tmp) for cfg_tmp in cfg_list]
            print("# of models", len(self.predictors))
            self.gravity_on = any([cfg_tmp.MODEL.GRAVITY_ON for cfg_tmp in cfg_list])
            self.latitude_on = any([cfg_tmp.MODEL.LATITUDE_ON for cfg_tmp in cfg_list])
            for i in range(len(cfg_list)):
                self.predictors[i].aug = T.Resize(cfg_list[i].DATALOADER.RESIZE)    
            self.device = self.predictors[0].model.device

        else:
            raise NotImplementedError

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        
        if self.multi_cfg:
            predictions = {}
            for predictor in self.predictors:
                predictions.update(predictor(image.copy()))
        else:
            predictions = self.predictor(image.copy())
        return predictions

    def opt_rpfpp(self, predictions, device, net_init, pp_on):
        if not pp_on:
            predictions['pred_rel_cx'] = predictions['pred_rel_cy'] = 0

        if net_init:
            init_params = {
                'roll': to_numpy(predictions['pred_roll']),
                'pitch': to_numpy(predictions['pred_pitch']),
                'focal': general_vfov_to_focal(
                        to_numpy(predictions['pred_rel_cx']), 
                        to_numpy(predictions['pred_rel_cy']), 
                        1, 
                        to_numpy(predictions['pred_general_vfov']), 
                        degree=True,
                    ),
                'cx': to_numpy(predictions['pred_rel_cx']) + 0.5,
                'cy': to_numpy(predictions['pred_rel_cy']) + 0.5,
            }
        else:
            init_params = None

        rpfpp = predict_rpfpp(
            up=to_numpy(predictions['pred_gravity_original']), 
            latimap=to_numpy(torch.deg2rad(predictions['pred_latitude_original'])),
            tolerance=1e-7,
            device=device,
            init_params=init_params,
            pp_on=pp_on,
        )
        return rpfpp
        

    def draw(self, image, latimap, gravity, latimap_format='', info=None, up_color=(0,1,0)):
        vis_output = None
        visualizer = None

        # BGR 2 RGB
        img = image[:,:,::-1]
        if self.latitude_on:
            latimap = latimap.to(self.cpu_device).numpy()
            if latimap_format == 'sin':
                latimap = np.arcsin(latimap)
            elif latimap_format == 'deg':
                latimap = np.radians(latimap)
            elif latimap_format == 'rad':
                pass
            else:
                print(latimap_format)
                raise NotImplementedError
            img = draw_latitude_field(img, latimap)   
        if self.gravity_on:   
            img = draw_up_field(img, to_numpy(gravity).transpose(1,2,0), color=up_color)
             
        visualizer = VisualizerPerspective(img.copy())
            
        if info is not None:
            visualizer.draw_text(info, (5, 5), horizontal_alignment='left')
        vis_output = visualizer.output
        

        return vis_output

