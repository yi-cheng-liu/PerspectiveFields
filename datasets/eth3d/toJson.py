import json
import math
import os
from scipy.spatial.transform import Rotation as R
import numpy as np
import multiprocessing as mp
import glob
import tqdm
from detectron2.data.detection_utils import read_image
from pathlib import Path
from importlib import util
from perspective2d.utils import draw_from_r_p_f
import cv2
root_dir = Path(__file__).parents[2]
module_path = root_dir / 'demo' / 'demo.py'
spec = util.spec_from_file_location("demo.demo", module_path)
demo = util.module_from_spec(spec)
spec.loader.exec_module(demo)


class ImageProcessor:
    def __init__(self, demo_module):
        self.M_PI = math.pi
        self.demo = demo_module
        self.roll = 0
        self.pitch = 0
        self.vfov = 0

    @staticmethod
    def read_camera_params(camera_file_path):
        with open(camera_file_path, 'r') as file:
            # Skip the first 3 lines
            for _ in range(3):
                next(file)
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 6:
                    camera_params = {
                        "CAMERA_ID": int(parts[0]),
                        "MODEL": parts[1],
                        "WIDTH": int(parts[2]),
                        "HEIGHT": int(parts[3]),
                        # fx, fy, cx, cy, k1, k2, k3, k4, p1, p2, p3, p4
                        "PARAMS": [float(param) for param in parts[4:]]
                    }
                    return camera_params

    @staticmethod
    def calculate_vfov(fy, height):
        vfov_rad = 2 * math.atan((height / 2) / fy)
        return math.degrees(vfov_rad)
    
    def adjust_pose(self, data_list, axis_order='ZXY'):
        given_rot = R.from_quat(R.from_euler(axis_order, [self.roll, self.pitch, 0], degrees=True).as_quat())
        
        first_frame_pose = data_list[0]
        first_frame_rot_inv = R.from_quat([first_frame_pose['qx'], first_frame_pose['qy'], first_frame_pose['qz'], first_frame_pose['qw']]).inv()
        
        for i, item in enumerate(data_list):
            # Rotation
            current_rot = R.from_quat([item['qx'], item['qy'], item['qz'], item['qw']])
            adjusted_rot = current_rot * first_frame_rot_inv * given_rot
            qx, qy, qz, qw = adjusted_rot.as_quat()
            
            roll, pitch, yaw = adjusted_rot.as_euler(axis_order, degrees=True)
            
            if i != 0:
                roll, pitch, yaw = -roll, -pitch, -yaw
            
            # Translation
            current_trans = np.array([item['tx'], item['ty'], item['tz']])
            tx, ty, tz = current_trans

            item.update({
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "tx": tx, "ty": ty, "tz": tz,
                "roll": roll, "pitch": pitch, "yaw": yaw
            })

    def process_images_folder(self, input_file_path, output_json_path, output_txt_path, folder_name, camera_params, args, axis_order='ZXY'):
        data_list = []

        with open(input_file_path, 'r') as file:
            for _ in range(4):
                next(file)
            i = None

            while True:
                line = file.readline()
                if not line:
                    break
                parts = line.strip().split()
                if len(parts) == 10:
                    image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = parts
                    qw, qx, qy, qz, tx, ty, tz = map(float, (qw, qx, qy, qz, tx, ty, tz))

                    rot = R.from_quat([qx, qy, qz, qw])
                    # roll, pitch, yaw(z, x, y)
                    roll, pitch, yaw = rot.as_euler(axis_order, degrees=True)
                    
                    fy = camera_params["PARAMS"][1]
                    height = camera_params["HEIGHT"]
                    vfov = self.calculate_vfov(fy, height)
                    
                    line_dict = {
                        "file_name": name,
                        "image_id": image_id,
                        "qw": qw,
                        "qx": qx,
                        "qy": qy,
                        "qz": qz,
                        "tx": tx,
                        "ty": ty,
                        "tz": tz,
                        "roll": roll,
                        "pitch": pitch,
                        "yaw": yaw,
                        "WIDTH": camera_params["WIDTH"],
                        "HEIGHT": camera_params["HEIGHT"],
                        "vfov": vfov,
                        "dataset": f"{folder_name}_test",
                    }

                    data_list.append(line_dict)

        # Sort Data list
        data_list.sort(key=lambda x: x['file_name'])
        
        # Adjust pose
        self.adjust_pose(data_list, axis_order)
        
        # Draw visualization result
        for i, path in enumerate(tqdm.tqdm(args.input[0], disable=not args.output)):
            img = read_image(path, format="BGR")
            img = draw_from_r_p_f(img, data_list[i]['roll'], data_list[i]['pitch'], data_list[i]['vfov'], mode='deg', up_color=(0, 1, 0, 1))
            output_f = os.path.join(args.output, os.path.basename(folder_name), os.path.basename(path).split(".")[0])
            if args.output:
                os.makedirs(output_f, exist_ok=True)
                cv2.imwrite(os.path.join(output_f, "param_pred.png"), img[:, :, ::-1])
        
        # Output to JSON format
        output_dict = {"data": data_list}
        with open(output_json_path, 'w') as json_file:
            json.dump(output_dict, json_file, indent=4)
            
        # Output to txt in COLMAP format
        with open(output_txt_path, 'w') as txt_file:
            txt_file.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            for item in data_list:
                txt_line = f"{item['image_id']} {item['qw']} {item['qx']} {item['qy']} {item['qz']} {item['tx']} {item['ty']} {item['tz']} 0 {item['file_name']}\n"
                txt_file.write(txt_line)
                txt_file.write("placeholder\n")

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    args = demo.get_parser().parse_args()
    demo.setup_logger(name="fvcore")
    logger = demo.setup_logger()
    logger.info("Arguments: " + str(args))
    
    base = Path(os.getcwd()) / 'datasets' / 'eth3d'
    include_dirs = [
        'botanical_garden', 'courtyard', 'delivery_area', 'door', 'electro', 
        'exhibition_hall', 'facade', 'kicker', 'lecture_room', 'living_room', 
        'lounge', 'meadow', 'observatory', 'office', 'old_computer', 
        'pipes', 'playground', 'relief', 'relief_2', 'statue', 'terrace', 
        'terrace_2', 'terrains'
    ]
    subdirectories = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and d in include_dirs]

    cfg_list = demo.setup_cfg(args)
    demo_instance = demo.VisualizationDemo(cfg_list=cfg_list)
    image_processor = ImageProcessor(demo)
    
    for folder_name in subdirectories:
        print("===========", folder_name, "===========")
        # Inference the first image in the folder
        images_path = os.path.join(folder_name, 'images/dslr_images_resized')
        image_files = glob.glob(os.path.join(images_path, "*.JPG"))
        if image_files:
            args.input = [sorted(image_files)]
            first_image_path = args.input[0][0]
        else:
            logger.warning(f"No images found in {images_path}")
            continue
        
        # Inference the first image in the folder
        img = read_image(first_image_path, format="BGR")
        pred = demo_instance.run_on_image(img)
        print("roll: ", pred['pred_roll'].item())
        print("pitch: ", pred['pred_pitch'].item())
        print("vfov: ", pred['pred_vfov'].item())
        image_processor.roll = pred['pred_roll'].item()
        image_processor.pitch = pred['pred_pitch'].item()
        image_processor.vfov = pred['pred_vfov'].item()
        
        # Read camera parameters
        camera_file_path = os.path.join(folder_name, 'dslr_calibration_undistorted/cameras.txt')
        camera_params = image_processor.read_camera_params(camera_file_path)
        
        # Process Image
        # axis_list = ['ZXY']

        # for axis_order in axis_list:
        #     print("===========", axis_order, "===========")
        #     input_file_path = os.path.join(folder_name, 'dslr_calibration_undistorted/images.txt')
        #     output_json_path = os.path.join(folder_name, f'test_{axis_order}.json')
        #     output_txt_path = os.path.join(folder_name, f'images.txt')
        #     image_processor.process_images_folder(input_file_path, output_json_path, output_txt_path, os.path.basename(folder_name), camera_params, args, axis_order)
            
        input_file_path = os.path.join(folder_name, 'dslr_calibration_undistorted/images.txt')
        output_json_path = os.path.join(folder_name, f'test_ZXY.json')
        output_txt_path = os.path.join(folder_name, f'images.txt')
        image_processor.process_images_folder(input_file_path, output_json_path, output_txt_path, os.path.basename(folder_name), camera_params, args, axis_order='ZXY')