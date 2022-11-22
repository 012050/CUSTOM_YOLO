from glob import glob
import os

from sklearn.model_selection import train_test_split

import yaml

# -----------------------------------------------------------
path1 = os.getcwd() + '/images/val/*.jpg'
img_list = glob(path1)
print(len(img_list))

# -----------------------------------------------------------
train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=len(img_list))
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

print(data)