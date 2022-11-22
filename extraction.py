import fiftyone as fo
import fiftyone.zoo as foz
import os

the_number = 5

dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    lable_types=["detections"],
    classes=["Fish"],
    max_samples=the_number,
    )
dataset.export(export_dir= os.getcwd(),dataset_type=fo.types.YOLOv5Dataset,)

print(os.getcwd())