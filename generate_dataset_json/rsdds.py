# Copyright (c) 2024 Qihang Zhou
# Licensed under the MIT License (refer to the LICENSE file for details)

# Modifications made by Jitao Ma, 2024

import os
import json


class MVTecSolver(object):

    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(test={})
        anomaly_samples = 0
        normal_samples = 0
        for phase in ['test']:
            cls_info = []
            images = os.listdir(f'{self.root}/2D_Image/INPUT_IMG')
            masks = os.listdir(f'{self.root}/Ground Truth')
            images.sort()
            masks.sort()
            for image, mask in zip(images, masks):
                is_abnormal = True
                info_img = dict(
                    img_path=f'{self.root}/2D_Image/INPUT_IMG/' + image,
                    mask_path=f'{self.root}/Ground Truth/' + mask if is_abnormal else '',
                    cls_name='rail',
                    specie_name='bad',
                    anomaly=1 if is_abnormal else 0,
                )
                cls_info.append(info_img)
                if phase == 'test':
                    if is_abnormal:
                        anomaly_samples = anomaly_samples + 1
                    else:
                        normal_samples = normal_samples + 1
            info[phase]['rail'] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)
if __name__ == '__main__':
    runner = MVTecSolver(root='/home/worker1/AD-datasets/rsdds')
    runner.run()
