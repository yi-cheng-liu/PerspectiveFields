import os

from detectron2.data import DatasetCatalog, MetadataCatalog


from perspective2d.data.datasets import (
    load_cities360_json,
    load_edina_json,
    load_gsv_json,
    load_tartanair_json,
    load_stanford2d3d_json,
    load_eth3d_json
)

SPLITS_STANFORD2D3D = {
    "stanford2d3d_test": (
        "./datasets/stanford2d3d-processed/test", 
        "./datasets/stanford2d3d-processed/test.json"
    ),
    "stanford2d3d_test_crop": (
        "./datasets/stanford2d3d-processed/stanford2d3d_crop", 
        "./datasets/stanford2d3d-processed/stanford2d3d_test_crop.json"
    ),
    "stanford2d3d_test_warp": (
        "./datasets/stanford2d3d-processed/stanford2d3d_warp", 
        "./datasets/stanford2d3d-processed/stanford2d3d_test_warp.json"
    )
}

SPLITS_TARTANAIR = {
    "tartanair_test": (
        "./datasets/tartanair-processed/test", 
        "./datasets/tartanair-processed/test.json"
    ),
    "tartanair_test_crop": (
        "./datasets/tartanair-processed/tartanair_crop", 
        "./datasets/tartanair-processed/tartanair_test_crop.json"
    ),
    "tartanair_test_warp": (
        "./datasets/tartanair-processed/tartanair_warp", 
        "./datasets/tartanair-processed/tartanair_test_warp.json"
    ),
}

SPLITS_GSV = {
    "gsv_train": (
        "./datasets/gsv/google_street_view_191210/manhattan",
        "./datasets/gsv/gsv_train_20210313.csv",
    ),
    "gsv_test": (
        "./datasets/gsv/google_street_view_191210/manhattan",
        "./datasets/gsv/gsv_test_20210313.csv",
    ),
    "gsv_val": (
        "./datasets/gsv/google_street_view_191210/manhattan",
        "./datasets/gsv/gsv_val_20210313.csv",
    ),
    "gsv_test_crop_uniform": (
        "./datasets/gsv/gsv_test_crop_uniform",
        "./datasets/gsv/gsv_test_crop_uniform.json",
    ),
}

SPLITS_EDINA = {
    "edina_train": (
        "./datasets/edina/edina_train",
        "./datasets/edina/edina_train.json",
    ),
    "edina_test": (
        "./datasets/edina/edina_test", 
        "./datasets/edina/edina_test.json"
    ),
    "edina_test_crop_uniform": (
        "./datasets/edina/edina_test_crop_uniform",
        "./datasets/edina/edina_test_crop_uniform.json",
    ),
    "edina_test_crop_vfov": (
        "./datasets/edina/edina_test_crop_vfov",
        "./datasets/edina/edina_test_crop_vfov.json",
    ),
}

SPLITS_CITIES360 = {
    "cities360_train": (
        "./datasets/cities360/cities360_json_v3/train",
        "./datasets/cities360/cities360_json_v3/train.json",
    ),
    "cities360_test": (
        "./datasets/cities360/cities360_json_v3/test",
        "./datasets/cities360/cities360_json_v3/test.json",
    ),
}

SPLITS_ETH3D = {
    "botanical_garden_test": (
        "./datasets/eth3d/botanical_garden/images",
        "./datasets/eth3d/botanical_garden/test.json"
    ),
    "boulders_test": (
        "./datasets/eth3d/boulders/images",
        "./datasets/eth3d/boulders/test.json"
    ),
    "bridge_test": (
        "./datasets/eth3d/bridge/images",
        "./datasets/eth3d/bridge/test.json"
    ),
    "door_test": (
        "./datasets/eth3d/door/images",
        "./datasets/eth3d/door/test.json"
    ),
    "exhibition_hall_test": (
        "./datasets/eth3d/exhibition_hall/images",
        "./datasets/eth3d/exhibition_hall/test.json"
    ),
    "lecture_room_test": (
        "./datasets/eth3d/lecture_room/images",
        "./datasets/eth3d/lecture_room/test.json"
    ),
    "living_room_test": (
        "./datasets/eth3d/living_room/images",
        "./datasets/eth3d/living_room/test.json"
    ),
    "lounge_test": (
        "./datasets/eth3d/lounge/images",
        "./datasets/eth3d/lounge/test.json"
    ),
    "observatory_test": (
        "./datasets/eth3d/observatory/images",
        "./datasets/eth3d/observatory/test.json"
    ),
    "old_computer_test": (
        "./datasets/eth3d/old_computer/images",
        "./datasets/eth3d/old_computer/test.json"
    ),
    "statue_test": (
        "./datasets/eth3d/statue/images",
        "./datasets/eth3d/statue/test.json"
    ),
    "terrace_2_test": (
        "./datasets/eth3d/terrace_2/images",
        "./datasets/eth3d/terrace_2/test.json"
    )
}


def register_gsv(dataset_name, json_file, img_root):
    if dataset_name not in DatasetCatalog.list():
        DatasetCatalog.register(dataset_name, lambda: load_gsv_json(json_file, img_root))
        MetadataCatalog.get(dataset_name).set(
            json_file=json_file,
            image_root=img_root,
            evaluator_type="perspective",
            ignore_label=-1,
        )
    else:
        print("GSV already registered: ", dataset_name)


def register_edina(dataset_name, json_file, img_root):
    if dataset_name not in DatasetCatalog.list():
        DatasetCatalog.register(dataset_name, lambda: load_edina_json(json_file, img_root))
        MetadataCatalog.get(dataset_name).set(
            json_file=json_file,
            image_root=img_root,
            evaluator_type="perspective",
            ignore_label=-1,
        )
    else:
        print("edina already registered: ", dataset_name)


def register_cities360(dataset_name, json_file, img_root="datasets"):
    if dataset_name not in DatasetCatalog.list():
        DatasetCatalog.register(
            dataset_name, lambda: load_cities360_json(json_file, img_root)
        )
        MetadataCatalog.get(dataset_name).set(
            json_file=json_file,
            image_root=img_root,
            evaluator_type="perspective",
            ignore_label=-1,
        )
    else:
        print("cities360 already registered: ", dataset_name)


def register_tartanair(dataset_name, json_file, img_root="datasets"):
    if dataset_name not in DatasetCatalog.list():
        DatasetCatalog.register(
            dataset_name, lambda: load_tartanair_json(json_file, img_root)
        )
        MetadataCatalog.get(dataset_name).set(
            json_file=json_file, image_root=img_root, evaluator_type="perspective",
            ignore_label=-1,
        )
    else:
        print("tantanair already registered: ", dataset_name)

def register_stanford2d3d(dataset_name, json_file, img_root="datasets"):
    if dataset_name not in DatasetCatalog.list():
        DatasetCatalog.register(
            dataset_name, lambda: load_stanford2d3d_json(json_file, img_root)
        )
        MetadataCatalog.get(dataset_name).set(
            json_file=json_file, image_root=img_root, evaluator_type="perspective",
            ignore_label=-1,
        )
    else:
        print("stanford2d3d already registered: ", dataset_name)
    
def register_eth3d(dataset_name, json_file, img_root="datasets"):
    if dataset_name not in DatasetCatalog.list():
        DatasetCatalog.register(
            dataset_name, lambda: load_eth3d_json(json_file, img_root)
        )
        MetadataCatalog.get(dataset_name).set(
            json_file=json_file, image_root=img_root, evaluator_type="perspective",
            ignore_label=-1,
        )
    else:
        print("eth3d already registered: ", dataset_name)

for key, (img_root, anno_file) in SPLITS_GSV.items():
    register_gsv(key, anno_file, img_root)

for key, (img_root, anno_file) in SPLITS_EDINA.items():
    register_edina(key, anno_file, img_root)

for key, (img_root, anno_file) in SPLITS_CITIES360.items():
    register_cities360(key, anno_file, img_root)

for key, (img_root, anno_file) in SPLITS_TARTANAIR.items():
    register_tartanair(key, anno_file, img_root)

for key, (img_root, anno_file) in SPLITS_STANFORD2D3D.items():
    register_stanford2d3d(key, anno_file, img_root)
    
for key, (img_root, anno_file) in SPLITS_ETH3D.items():
    register_eth3d(key, anno_file, img_root)