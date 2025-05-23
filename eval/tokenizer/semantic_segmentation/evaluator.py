import argparse

import numpy as np
import torch
from tqdm import tqdm
from prettytable import PrettyTable

from metric import FMeasureGPU, IoUDifferentSizeGPUWithBoundary, IoUGPU
from tools import get_loader, get_param

def evaluator(loader, num_classes, device):
    log = {}

    T = torch.zeros(size=(num_classes + 1,)).to(device)
    P = torch.zeros(size=(num_classes + 1,)).to(device)
    TP = torch.zeros(size=(num_classes + 1,)).to(device)
    IoU = torch.zeros(size=(num_classes + 1,)).to(device)
    BT = torch.zeros(size=(num_classes + 1,)).to(device)
    BP = torch.zeros(size=(num_classes + 1,)).to(device)
    BTP = torch.zeros(size=(num_classes + 1,)).to(device)
    BIoU = torch.zeros(size=(num_classes + 1,)).to(device)
    FMeasure = 0.
    ACC = 0.

    # mIoU with different object sizes
    Ts = [torch.zeros(size=(num_classes + 1,)).to(device) for _ in range(4)]
    Ps = [torch.zeros(size=(num_classes + 1,)).to(device) for _ in range(4)]
    TPs = [torch.zeros(size=(num_classes + 1,)).to(device) for _ in range(4)]
    mIoUs = [torch.zeros(size=(num_classes + 1,)).to(device) for _ in range(4)]

    BTs = [torch.zeros(size=(num_classes + 1,)).to(device) for _ in range(4)]
    BPs = [torch.zeros(size=(num_classes + 1,)).to(device) for _ in range(4)]
    BTPs = [torch.zeros(size=(num_classes + 1,)).to(device) for _ in range(4)]
    mBIoUs = [torch.zeros(size=(num_classes + 1,)).to(device) for _ in range(4)]

    for idx, (gt, predict, boundary_gt, boundary_predict) in enumerate(tqdm(loader)):
        gt = gt.to(device)
        predict = predict.to(device)
        boundary_gt = boundary_gt.to(device)
        boundary_predict = boundary_predict.to(device)

        # IoU calculations
        area_intersection, area_output, area_target = IoUGPU(predict.view(-1), gt.view(-1), num_classes + 1)
        area_intersection_boundary, area_output_boundary, area_target_boundary = IoUGPU(boundary_predict.view(-1), boundary_gt.view(-1), num_classes + 2)
        
        IoUDifferentSizeGPUWithBoundary(predict.view(-1), gt.view(-1), boundary_predict.view(-1), boundary_gt.view(-1), num_classes + 1, Ts, Ps, TPs, BTs, BPs, BTPs)
        f_score = FMeasureGPU(predict, gt)

        T += area_output
        P += area_target
        TP += area_intersection
        BT += area_output_boundary[1:]
        BP += area_target_boundary[1:]
        BTP += area_intersection_boundary[1:]
        FMeasure += f_score

        img_label = torch.argmax(area_output[1:]) + 1
        ACC += (area_target[img_label] > 0) * (area_output[img_label] > 0)

    # Calculate IoU and other metrics
    IoU = TP / (T + P - TP + 1e-10) * 100
    BIoU = BTP / (BT + BP - BTP + 1e-10) * 100
    FMeasure = FMeasure / len(loader.dataset)
    ACC = ACC / len(loader.dataset)

    mIoU = torch.mean(IoU).item()
    mBIoU = torch.mean(BIoU).item()
    FMeasure = FMeasure.item() * 100
    ACC = ACC.item() * 100

    for i in range(4):
        if torch.sum(Ps[i]) > 0:
            mIoUs[i] = torch.mean((TPs[i] / (Ts[i] + Ps[i] - TPs[i] + 1e-10))[Ps[i] > 0]).item() * 100
            mBIoUs[i] = torch.mean((BTPs[i] / (BTs[i] + BPs[i] - BTPs[i] + 1e-10))[BPs[i] > 0]).item() * 100
        else:
            mIoUs[i] = 0
            mBIoUs[i] = 0

    log['Acc'] = ACC
    log['mIoU'] = mIoU
    log['mBIoU'] = mBIoU
    log['S'] = mIoUs[0]
    log['MS'] = mIoUs[1]
    log['ML'] = mIoUs[2]
    log['L'] = mIoUs[3]
    log['BS'] = mBIoUs[0]
    log['BMS'] = mBIoUs[1]
    log['BML'] = mBIoUs[2]
    log['BL'] = mBIoUs[3]
    log['IoUs'] = IoU.tolist()
    log['FMeasure'] = FMeasure

    return log

def display(log):
    class_table_data = PrettyTable()

    IoUs = [np.round(value, 2) for value in log['IoUs']]
    class_table_data.add_column('classes', log['classes'])
    class_table_data.add_column('IoUs', IoUs)

    summary_table_data = PrettyTable()
    for key, val in log.items():
        if key == 'IoUs' or key == 'classes':
            continue
        summary_table_data.add_column(key, [np.round(val, 2)])

    print('per class IoUs:\n')
    print(class_table_data.get_string())
    print('Summary:\n')
    print(summary_table_data.get_string())

def evaludation(args):
    params = get_param(args.mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_loader = get_loader(args.predict_dir, args.gt_dir,
                            name_list=args.name_list,
                            workers=args.workers,
                            match=args.match)

    log = evaluator(val_loader, num_classes=params['num_classes'], device=device)

    classes = getattr(val_loader.dataset, params['classes'])
    log['classes'] = classes
    display(log)

    return log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-dir", type=str)
    parser.add_argument("--gt-dir", type=str)
    parser.add_argument('--workers', default=32, type=int)
    parser.add_argument('--match', default=None, type=str)
    parser.add_argument('--mode', type=str, default='919', choices=['50', '300', '919'])
    parser.add_argument('--name-list', type=str, required=True, help='Path to the name list file')
    args = parser.parse_args()

    evaludation(args)
