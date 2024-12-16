import pydicom
import os, json
import numpy as np
from PIL import Image


info = dict(test={})
classes = []
for i in [1, 2, 5, 6, 8, 10, 14, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
    img_paths = os.listdir('/home/worker1/AD-datasets/CHAOS_Train/Train_Sets/CT/' + f'{i}/DICOM_anon/')
    lab_paths = os.listdir('/home/worker1/AD-datasets/CHAOS_Train/Train_Sets/CT/' + f'{i}/Ground/')
    info['test'][str(i)] = []
    for img_path, lab_path in zip(img_paths, lab_paths):
        # img = np.array(pydicom.read_file('/home/worker1/AD-datasets/CHAOS_Train/Train_Sets/CT/' + f'{i}/DICOM_anon/' + img_path).pixel_array)
        lab = Image.open('/home/worker1/AD-datasets/CHAOS_Train/Train_Sets/CT/' + f'{i}/Ground/' + lab_path)
        name = os.path.splitext(img_path)[0]
        is_bad = np.asarray(lab).max()

        # img = (img - img.min()) / (img.max() - img.min()) * 255
        # img = img.astype(np.uint8)
        # img = Image.fromarray(np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1))
        if is_bad == 1:
            # img.save('/home/worker1/AD-datasets/CHAOS_Train/Train_Sets/liver/train/bad/' + name + '.png')
            # lab.save('/home/worker1/AD-datasets/CHAOS_Train/Train_Sets/liver/ground_truth/bad/' + name + '.png')
            save_info = dict(
                img_path='liver/train/bad/' + name + '.png',
                mask_path='liver/ground_truth/bad/' + name + f'.png',
                cls_name=str(i),
                specie_name='bad',
                anomaly=1,)
        else:
            # img.save('/home/worker1/AD-datasets/CHAOS_Train/Train_Sets/liver/train/good/' + name + '.png')
            save_info = dict(
                img_path='liver/train/good/' + name + '.png',
                mask_path='',
                cls_name=str(i),
                specie_name='good',
                anomaly=0,)
        info['test'][str(i)].append(save_info)
    classes.append(str(i))
with open('/home/worker1/AD-datasets/CHAOS_Train/Train_Sets/meta.json', 'w') as f:
    f.write(json.dumps(info, indent=4) + "\n")
print(classes)
