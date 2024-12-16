# Copyright (c) 2024 Qihang Zhou
# Licensed under the MIT License (refer to the LICENSE file for details)

# Modifications made by Jitao Ma, 2024

import nibabel as nib
import os, json
import numpy as np
from PIL import Image


paths = [{"image":"./imagesTr/lung_053.nii.gz","label":"./labelsTr/lung_053.nii.gz"},{"image":"./imagesTr/lung_022.nii.gz","label":"./labelsTr/lung_022.nii.gz"},{"image":"./imagesTr/lung_041.nii.gz","label":"./labelsTr/lung_041.nii.gz"},{"image":"./imagesTr/lung_069.nii.gz","label":"./labelsTr/lung_069.nii.gz"},{"image":"./imagesTr/lung_014.nii.gz","label":"./labelsTr/lung_014.nii.gz"},{"image":"./imagesTr/lung_006.nii.gz","label":"./labelsTr/lung_006.nii.gz"},{"image":"./imagesTr/lung_065.nii.gz","label":"./labelsTr/lung_065.nii.gz"},{"image":"./imagesTr/lung_018.nii.gz","label":"./labelsTr/lung_018.nii.gz"},{"image":"./imagesTr/lung_096.nii.gz","label":"./labelsTr/lung_096.nii.gz"},{"image":"./imagesTr/lung_084.nii.gz","label":"./labelsTr/lung_084.nii.gz"},{"image":"./imagesTr/lung_086.nii.gz","label":"./labelsTr/lung_086.nii.gz"},{"image":"./imagesTr/lung_043.nii.gz","label":"./labelsTr/lung_043.nii.gz"},{"image":"./imagesTr/lung_020.nii.gz","label":"./labelsTr/lung_020.nii.gz"},{"image":"./imagesTr/lung_051.nii.gz","label":"./labelsTr/lung_051.nii.gz"},{"image":"./imagesTr/lung_079.nii.gz","label":"./labelsTr/lung_079.nii.gz"},{"image":"./imagesTr/lung_004.nii.gz","label":"./labelsTr/lung_004.nii.gz"},{"image":"./imagesTr/lung_075.nii.gz","label":"./labelsTr/lung_075.nii.gz"},{"image":"./imagesTr/lung_016.nii.gz","label":"./labelsTr/lung_016.nii.gz"},{"image":"./imagesTr/lung_071.nii.gz","label":"./labelsTr/lung_071.nii.gz"},{"image":"./imagesTr/lung_028.nii.gz","label":"./labelsTr/lung_028.nii.gz"},{"image":"./imagesTr/lung_055.nii.gz","label":"./labelsTr/lung_055.nii.gz"},{"image":"./imagesTr/lung_036.nii.gz","label":"./labelsTr/lung_036.nii.gz"},{"image":"./imagesTr/lung_047.nii.gz","label":"./labelsTr/lung_047.nii.gz"},{"image":"./imagesTr/lung_059.nii.gz","label":"./labelsTr/lung_059.nii.gz"},{"image":"./imagesTr/lung_061.nii.gz","label":"./labelsTr/lung_061.nii.gz"},{"image":"./imagesTr/lung_010.nii.gz","label":"./labelsTr/lung_010.nii.gz"},{"image":"./imagesTr/lung_073.nii.gz","label":"./labelsTr/lung_073.nii.gz"},{"image":"./imagesTr/lung_026.nii.gz","label":"./labelsTr/lung_026.nii.gz"},{"image":"./imagesTr/lung_038.nii.gz","label":"./labelsTr/lung_038.nii.gz"},{"image":"./imagesTr/lung_045.nii.gz","label":"./labelsTr/lung_045.nii.gz"},{"image":"./imagesTr/lung_034.nii.gz","label":"./labelsTr/lung_034.nii.gz"},{"image":"./imagesTr/lung_049.nii.gz","label":"./labelsTr/lung_049.nii.gz"},{"image":"./imagesTr/lung_057.nii.gz","label":"./labelsTr/lung_057.nii.gz"},{"image":"./imagesTr/lung_080.nii.gz","label":"./labelsTr/lung_080.nii.gz"},{"image":"./imagesTr/lung_092.nii.gz","label":"./labelsTr/lung_092.nii.gz"},{"image":"./imagesTr/lung_015.nii.gz","label":"./labelsTr/lung_015.nii.gz"},{"image":"./imagesTr/lung_064.nii.gz","label":"./labelsTr/lung_064.nii.gz"},{"image":"./imagesTr/lung_031.nii.gz","label":"./labelsTr/lung_031.nii.gz"},{"image":"./imagesTr/lung_023.nii.gz","label":"./labelsTr/lung_023.nii.gz"},{"image":"./imagesTr/lung_005.nii.gz","label":"./labelsTr/lung_005.nii.gz"},{"image":"./imagesTr/lung_078.nii.gz","label":"./labelsTr/lung_078.nii.gz"},{"image":"./imagesTr/lung_066.nii.gz","label":"./labelsTr/lung_066.nii.gz"},{"image":"./imagesTr/lung_009.nii.gz","label":"./labelsTr/lung_009.nii.gz"},{"image":"./imagesTr/lung_074.nii.gz","label":"./labelsTr/lung_074.nii.gz"},{"image":"./imagesTr/lung_042.nii.gz","label":"./labelsTr/lung_042.nii.gz"},{"image":"./imagesTr/lung_033.nii.gz","label":"./labelsTr/lung_033.nii.gz"},{"image":"./imagesTr/lung_095.nii.gz","label":"./labelsTr/lung_095.nii.gz"},{"image":"./imagesTr/lung_037.nii.gz","label":"./labelsTr/lung_037.nii.gz"},{"image":"./imagesTr/lung_054.nii.gz","label":"./labelsTr/lung_054.nii.gz"},{"image":"./imagesTr/lung_029.nii.gz","label":"./labelsTr/lung_029.nii.gz"},{"image":"./imagesTr/lung_058.nii.gz","label":"./labelsTr/lung_058.nii.gz"},{"image":"./imagesTr/lung_025.nii.gz","label":"./labelsTr/lung_025.nii.gz"},{"image":"./imagesTr/lung_046.nii.gz","label":"./labelsTr/lung_046.nii.gz"},{"image":"./imagesTr/lung_070.nii.gz","label":"./labelsTr/lung_070.nii.gz"},{"image":"./imagesTr/lung_001.nii.gz","label":"./labelsTr/lung_001.nii.gz"},{"image":"./imagesTr/lung_062.nii.gz","label":"./labelsTr/lung_062.nii.gz"},{"image":"./imagesTr/lung_083.nii.gz","label":"./labelsTr/lung_083.nii.gz"},{"image":"./imagesTr/lung_081.nii.gz","label":"./labelsTr/lung_081.nii.gz"},{"image":"./imagesTr/lung_093.nii.gz","label":"./labelsTr/lung_093.nii.gz"},{"image":"./imagesTr/lung_044.nii.gz","label":"./labelsTr/lung_044.nii.gz"},{"image":"./imagesTr/lung_027.nii.gz","label":"./labelsTr/lung_027.nii.gz"},{"image":"./imagesTr/lung_048.nii.gz","label":"./labelsTr/lung_048.nii.gz"},{"image":"./imagesTr/lung_003.nii.gz","label":"./labelsTr/lung_003.nii.gz"}]


info = dict(test={})
# classes = []
for idx, path in enumerate(paths):
    if idx % 3 == 0:
        info = dict(test={})
    img = nib.load('/home/worker1/AD-datasets/MSD_Task06_Lung/Task06_Lung' + path['image'][1:])
    img_data = img.get_fdata()
    label = nib.load('/home/worker1/AD-datasets/MSD_Task06_Lung/Task06_Lung' + path['label'][1:])
    label_data = label.get_fdata()
    name = os.path.splitext(os.path.splitext(os.path.split(path['image'][1:])[1])[0])[0]
    info['test'][name] = []
    # classes.append(name)
    if label_data.max() != 1:
        continue
    for i in range(img_data.shape[-1]):
        # d = img_data[:, :, [i]]
        # d = (d - d.min()) / (d.max() - d.min()) * 255
        # d = d.astype(np.uint8)
        # d = Image.fromarray(np.repeat(d, 3, axis=-1))
        # l = (label_data[:, :, i] * 255).astype(np.uint8)
        # l = Image.fromarray(l)
        is_bad = label_data[:, :, i].max()
        if is_bad == 1:
            # d.save('/home/worker1/AD-datasets/MSD_Task06_Lung/Task06_Lung/liver/train/bad/' + name + f'_{i}.png')
            # l.save('/home/worker1/AD-datasets/MSD_Task06_Lung/Task06_Lung/liver/ground_truth/bad/' + name + f'_{i}.png')
            save_info = dict(
                img_path='liver/train/bad/' + name + f'_{i}.png',
                mask_path='liver/ground_truth/bad/' + name + f'_{i}.png',
                cls_name=name,
                specie_name='bad',
                anomaly=1,)
        else:
            # d.save('/home/worker1/AD-datasets/MSD_Task06_Lung/Task06_Lung/liver/train/good/' + name + f'_{i}.png')
            save_info = dict(
                img_path='liver/train/good/' + name + f'_{i}.png',
                mask_path='',
                cls_name=name,
                specie_name='good',
                anomaly=0,)
        info['test'][name].append(save_info)
    if idx % 3 == 0:
        with open('/home/worker1/AD-datasets/MSD_Task06_Lung/Task06_Lung/meta_{}.json'.format(idx), 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
# print(classes)
