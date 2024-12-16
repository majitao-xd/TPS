# Copyright (c) 2024 Qihang Zhou
# Licensed under the MIT License (refer to the LICENSE file for details)

# Modifications made by Jitao Ma, 2024

import os
import json
import pandas as pd


class ClinicDBSolver(object):
    CLSNAMES = [
        'eye',
    ]

    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}'
            for phase in ['train']:
                cls_info = []
                # is_abnormal = True if specie not in ['good'] else False
                img_names = os.listdir(f'{cls_dir}/images')
                mask_names = os.listdir(f'{cls_dir}/1st_manual')
                img_names.sort()
                mask_names.sort() if mask_names is not None else None
                for idx, img_name in enumerate(img_names):
                    info_img = dict(
                        img_path=f'{cls_dir}/images/{img_name}',
                        mask_path=f'{cls_dir}/1st_manual/{mask_names[idx]}',
                        cls_name=cls_name,
                        specie_name='',
                        anomaly=1
                    )
                    cls_info.append(info_img)
                    if phase == 'test':
                        if True:
                            anomaly_samples = anomaly_samples + 1
                        else:
                            normal_samples = normal_samples + 1
                info['test'][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)



if __name__ == '__main__':
    runner = ClinicDBSolver(root='/home/worker1/AD-datasets/DRIVE/train')
    runner.run()