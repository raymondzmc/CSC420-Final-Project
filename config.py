import os

############################## Paths ##############################

ROOT_DIR = os.getcwd()

# Path for storing images and label
data_path = os.path.join(ROOT_DIR, 'images')
label_path = os.path.join(ROOT_DIR, 'labels')

# Path for Mask R-CNN results
maskrcnn_path = os.path.join(ROOT_DIR, 'results-seg', 'mask_rcnn')
maskrcnn_visual_path = os.path.join(ROOT_DIR, 'results-seg', 'mask_rcnn_results')

# For storing final results
person_path = os.path.join(ROOT_DIR, 'results-seg', 'person')
clothes_path = os.path.join(ROOT_DIR, 'results-seg', 'clothes')

# For storing checkpoints
ckpt_path = os.path.join(ROOT_DIR, 'checkpoints')

# Create directories if doesn't exists
if not os.path.isdir(person_path):
    os.makedirs(person_path, exist_ok=True)
if not os.path.isdir(clothes_path):
    os.makedirs(clothes_path, exist_ok=True)
if not os.path.isdir(ckpt_path):
    os.makedirs(ckpt_path, exist_ok=True)
if not os.path.isdir(maskrcnn_visual_path):
    os.makedirs(maskrcnn_visual_path, exist_ok=True)

############################## Training ##############################

cls_names = {
    0: 'background',
    1: 'skin',
    2: 'hair',
    3: 'tshirt',
    4: 'shoes',
    5: 'pants',
    6: 'dress'
}

# Training parameters
epochs = 300
lr = 1e-4
decay = 1e-5

# ImageNet mean and std
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Network input size
input_size = (224, 224)

# Original input size
image_size = (600, 400)

color_mapping = {
    0: [0, 0, 0],
    1: [128, 0, 0],
    2: [0, 128, 0],
    3: [128, 128, 0],
    4: [0, 0, 128],
    5: [128, 0, 128],
    6: [0, 128, 128]
}