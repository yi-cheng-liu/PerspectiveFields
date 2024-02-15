import json
import math
import os

def read_camera_params(camera_file_path): 
    with open(camera_file_path, 'r') as file:
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
                    "PARAMS": [float(param) for param in parts[4:]]
                }
                
                return camera_params

def calculate_vfov(fy, height):
    vfov_rad = 2 * math.atan(height / (2 * fy))
    return math.degrees(vfov_rad)

def quaternion_to_rpy(qw, qx, qy, qz):
    roll = math.atan2(2 * (qw*qx + qy*qz), 1 - 2 * (qx**2 + qy**2))
    pitch = math.asin(max(min(2 * (qw*qy - qz*qx), 1), -1))
    yaw = math.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy**2 + qz**2))
    
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    
    return roll_deg, pitch_deg, yaw_deg

def process_images_folder(input_file_path, output_json_path, folder_name, camera_params): 
    data_list = []
    with open(input_file_path, 'r') as file:
        for _ in range(4):
            next(file)

        while True:
            line = file.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) == 10:
                image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = parts
                qw, qx, qy, qz = map(float, (qw, qx, qy, qz))
                roll, pitch, yaw = quaternion_to_rpy(qw, qx, qy, qz)
                
                fy = camera_params["PARAMS"][1]
                height = camera_params["HEIGHT"]
                vfov = calculate_vfov(fy, height)
                
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
                file.readline()

    output_dict = {
        "data": data_list
    }

    with open(output_json_path, 'w') as json_file:
        json.dump(output_dict, json_file, indent=4)

if __name__ == '__main__':
    base = os.getcwd()
    subdirectories = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]

    for folder_name in subdirectories:
        # Read camera parameters
        camera_file_path = os.path.join(folder_name, 'dslr_calibration_jpg/cameras.txt')
        camera_params = read_camera_params(camera_file_path)
        
        # Process Image
        input_file_path = os.path.join(folder_name, 'dslr_calibration_jpg/images.txt')
        output_json_path = os.path.join(folder_name, 'test.json')
        process_images_folder(input_file_path, output_json_path, os.path.basename(folder_name), camera_params)