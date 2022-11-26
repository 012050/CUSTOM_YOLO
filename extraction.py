import fiftyone as fo
import fiftyone.zoo as foz
import os
from glob import glob
from sklearn.model_selection import train_test_split
import yaml

the_number = 5
text_rate = 0.2

dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    lable_types=["detections"],
    classes=["Fish"],
    max_samples=the_number,
    )
dataset.export(export_dir= os.getcwd(),dataset_type=fo.types.YOLOv5Dataset,)

print(os.getcwd())

# -----------------------------------------------------------
path1 = os.getcwd() + '/images/val/*.jpg'
img_list = glob(path1)
print(len(img_list))

# -----------------------------------------------------------
train_img_list, val_img_list = train_test_split(img_list, test_size=text_rate, random_state=len(img_list))
print(len(train_img_list), len(val_img_list))

# -----------------------------------------------------------
path2 = os.getcwd() + "/train.txt"
with open(path2, 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')
path3 = os.getcwd() + "/val.txt"
with open(path3, 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')

# -----------------------------------------------------------
path4 = os.getcwd() + "/dataset.yaml"
with open(path4, 'r') as f:
    data = yaml.safe_load(f)
print(data)
data['train'] = os.getcwd() + "/train.txt"
data['val'] = os.getcwd() + "/val.txt"
path5 = os.getcwd() + "/dataset.yaml"
with open(path5, 'w') as f:
    yaml.dump(data, f)
