# Copyright (c) 2024 Qihang Zhou
# Licensed under the MIT License (refer to the LICENSE file for details)

import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'SDD':
        obj_list = ['electrical commutators']
    elif dataset_name == 'DTD':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'colon':
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        obj_list = ['skin']
    elif dataset_name == 'Chest':
        obj_list = ['chest']
    elif dataset_name == 'thyroid':
        obj_list = ['thyroid']
    elif dataset_name == 'headct':
        obj_list = ['brain']
    elif dataset_name == 'brainmri':
        obj_list = ['brain']
    elif dataset_name == 'br35':
        obj_list = ['brain']
    elif dataset_name == 'sds':
        obj_list = ['iron']
    elif dataset_name == 'msd':
        obj_list = ['lung_053', 'lung_022', 'lung_041', 'lung_069', 'lung_014', 'lung_006', 'lung_065', 'lung_018', 'lung_096', 'lung_084', 'lung_086', 'lung_043', 'lung_020', 'lung_051', 'lung_079', 'lung_004', 'lung_075', 'lung_016', 'lung_071', 'lung_028', 'lung_055', 'lung_036', 'lung_047', 'lung_059', 'lung_061', 'lung_010', 'lung_073', 'lung_026', 'lung_038', 'lung_045', 'lung_034', 'lung_049', 'lung_057', 'lung_080', 'lung_092', 'lung_015', 'lung_064', 'lung_031', 'lung_023', 'lung_005', 'lung_078', 'lung_066', 'lung_009', 'lung_074', 'lung_042', 'lung_033', 'lung_095', 'lung_037', 'lung_054', 'lung_029', 'lung_058', 'lung_025', 'lung_046', 'lung_070', 'lung_001', 'lung_062', 'lung_083', 'lung_081', 'lung_093', 'lung_044', 'lung_027', 'lung_048', 'lung_003']
    elif dataset_name == 'chaos':
        obj_list = ['1', '2', '5', '6', '8', '10', '14', '16', '18', '19', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
    elif dataset_name == 'rsdds':
        obj_list = ['rail']
    elif dataset_name == 'drive':
        obj_list = ['eye']
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(   
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}    