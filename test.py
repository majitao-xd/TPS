# Copyright (c) 2024 Qihang Zhou
# Licensed under the MIT License (refer to the LICENSE file for details)

# Modifications made by Jitao Ma, 2024

import math

import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
                              "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform,
                        dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    prompt_learner.eval()
    class_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters, classnames=['part'],
                                              state_list=['the flawless {}', 'the damaged {}'])
    class_learner.load_state_dict(checkpoint["class_learner"])
    class_learner.to(device)
    class_learner.eval()
    former = AnomalyCLIP_lib.PathWay(model.text_projection, model.visual.proj)
    former.load_state_dict(checkpoint["former"])
    former.to(device)
    former.eval()
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=None)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features, text_features_all = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text)
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    prompts, tokenized_prompts, compound_prompts_text = class_learner(cls_id=None)
    patch_classes, patch_classes_all = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text)
    patch_classes = torch.stack(torch.chunk(patch_classes, dim=0, chunks=2), dim=1)
    # patch_classes = patch_classes / patch_classes.norm(dim=-1, keepdim=True)

    model.to(device)
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            image_features, patch_features, all_image_features = model.encode_image(image, features_list,
                                                                                    DPAM_layer=None)
            # image_features = former.img_forward(image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features_norm.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            # similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
            # anomaly_map_list = [(similarity_map[...,1] + 1 - similarity_map[...,0])/2.0]
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    class_feature, _ = former(text_features.repeat(patch_feature.shape[0], 1, 1), patch_classes.repeat(patch_feature.shape[0], 1, 1), patch_feature[:, 1:, :])
                    # class_feature = class_feature / class_feature.norm(dim=-1, keepdim=True)
                    class_feature = patch_classes / patch_classes.norm(dim=-1, keepdim=True)
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    # similarity = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity = AnomalyCLIP_lib.compute_similarity(patch_feature, class_feature)
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list)

            anomaly_map = anomaly_map.sum(dim=0)
            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            anomaly_map = torch.stack(
                [torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()], dim=0)
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    for obj in obj_list:
        table = []
        table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
        elif args.metrics == 'pixel-level':
            if results[obj]['imgs_masks'].shape[0] >= 2000:
                n = math.floor(results[obj]['imgs_masks'].shape[0] / 100)
                pixel_auroc = 0
                pixel_aupro = 0
                for i in range(n):
                    gt = results[obj]['imgs_masks'][i * 100: (i + 1) * 100, ...] if (i + 1) != n else results[obj][
                                                                                                          'imgs_masks'][
                                                                                                      i * 100:, ...]
                    pr = results[obj]['anomaly_maps'][i * 100: (i + 1) * 100, ...] if (i + 1) != n else results[obj][
                                                                                                            'imgs_masks'][
                                                                                                        i * 100:, ...]
                    pixel_auroc += pixel_level_metrics([gt, pr], obj, "pixel-auroc")
                    pixel_aupro += pixel_level_metrics([gt, pr], obj, "pixel-aupro")
                pixel_auroc /= (i + 1)
                pixel_aupro /= (i + 1)
            else:
                gt = results[obj]['imgs_masks']
                pr = results[obj]['anomaly_maps']
                pixel_auroc = pixel_level_metrics([gt, pr], obj, "pixel-auroc")
                pixel_aupro = pixel_level_metrics([gt, pr], obj, "pixel-aupro")
            # pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            # pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        elif args.metrics == 'image-pixel-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")

            if results[obj]['imgs_masks'].shape[0] >= 2000:
                n = math.floor(results[obj]['imgs_masks'].shape[0] / 100)
                pixel_auroc = 0
                pixel_aupro = 0
                for i in range(n):
                    gt = results[obj]['imgs_masks'][i * 100: (i + 1) * 100, ...] if (i + 1) != n else results[obj][
                                                                                                          'imgs_masks'][
                                                                                                      i * 100:, ...]
                    pr = results[obj]['anomaly_maps'][i * 100: (i + 1) * 100, ...] if (i + 1) != n else results[obj][
                                                                                                            'imgs_masks'][
                                                                                                        i * 100:, ...]
                    pixel_auroc += pixel_level_metrics([gt, pr], obj, "pixel-auroc")
                    pixel_aupro += pixel_level_metrics([gt, pr], obj, "pixel-aupro")
                pixel_auroc /= (i + 1)
                pixel_aupro /= (i + 1)
            else:
                gt = results[obj]['imgs_masks']
                pr = results[obj]['anomaly_maps']
                pixel_auroc = pixel_level_metrics([gt, pr], obj, "pixel-auroc")
                pixel_aupro = pixel_level_metrics([gt, pr], obj, "pixel-aupro")
            # pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            # pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")

            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        table_ls.append(table)

    if args.metrics == 'image-level':
        # logger
        table_ls.append(['mean',
                         str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                         str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    elif args.metrics == 'pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                         str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))
                         ])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
    elif args.metrics == 'image-pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                         str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)),
                         str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                         str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'],
                           tablefmt="pipe")
    logger.info("\n%s", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    # paths image-pixel level
    # parser.add_argument("--data_path", type=str, default="/home/worker1/AD-datasets/VisA_20220922-dataset", help="path to test dataset")
    # parser.add_argument("--data_path", type=str, default="/home/worker1/AD-datasets/mvtec-dataset", help="path to test dataset")
    # parser.add_argument("--data_path", type=str, default="/home/worker1/AD-datasets/btad/BTech_Dataset_transformed")
    # parser.add_argument("--data_path", type=str, default="/home/worker1/AD-datasets/MPDD_OneDrive_1_2024-6-5")
    # paths pixel level
    # parser.add_argument("--data_path", type=str, default="/home/worker1/AD-datasets/SD_saliency_900")
    # parser.add_argument("--data_path", type=str, default="/home/worker1/AD-datasets/MSD_Task06_Lung/Task06_Lung")
    # parser.add_argument("--data_path", type=str, default="/home/worker1/AD-datasets/CHAOS_Train/Train_Sets")
    # parser.add_argument("--data_path", type=str, default="/home/worker1/AD-datasets/ISBI2016")
    # parser.add_argument("--data_path", type=str, default="/home/worker1/AD-datasets/kvasir-seg/Kvasir-SEG") # colon
    # parser.add_argument("--data_path", type=str, default="/home/worker1/AD-datasets/Thyroid Dataset/tn3k") # thyroid
    parser.add_argument("--data_path", type=str, default="/home/worker1/AD-datasets/rsdds")  # rsdds
    # parser.add_argument("--data_path", type=str, default="/home/worker1/AD-datasets/DRIVE/train")  # drive
    # paths image level
    # parser.add_argument("--data_path", type=str, default="/home/worker1/AD-datasets/HeadCT-dataset")
    # parser.add_argument("--data_path", type=str, default="/home/worker1/AD-datasets/BrainMRI-dataset")
    # parser.add_argument("--data_path", type=str, default="/home/worker1/AD-datasets/br35h")
    # other paths
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    # parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/9_12_4_multiscale_visa/epoch_15.pth', help='path to checkpoint')
    # parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/fusion_epoch_15_mvtec.pth', help='path to checkpoint')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/epoch_15_6.pth', help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='rsdds')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")

    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
