import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.freqnet import freqnet
from options.test_options import TestOptions
import numpy as np
import random

DetectionTests = {}

opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

if "DS_v1_1" in opt.test_sets:
    DetectionTests["DS_v1_1"] = {
        'dataroot': '/home/ubuntu/deepspeak-experiments/deepspeak-neurips-2025/FreqNet_test_format/deepspeak_v1_1__test',
        'no_resize': False,
        'no_crop': True,
    }

if "DS_v2" in opt.test_sets:
    DetectionTests["DS_v2"] = {
        'dataroot': '/home/ubuntu/deepspeak-experiments/deepspeak-neurips-2025/FreqNet_test_format/deepspeak_v2__test',
        'no_resize': False,
        'no_crop': True,
    }

if "DS_full" in opt.test_sets:
    DetectionTests["DS_full"] = {
        'dataroot': '/home/ubuntu/deepspeak-experiments/deepspeak-neurips-2025/FreqNet_test_format/deepspeak_full__test',
        'no_resize': False,
        'no_crop': True,
    }

if "Celeb_DF_2" in opt.test_sets:
    DetectionTests["Celeb_DF_2"] = {
        'dataroot': '/home/ubuntu/deepspeak-experiments/deepspeak-neurips-2025/Celeb-DF-v2-preprocessed',
        'no_resize': False,
        'no_crop': True,
    }

if "GAN-Detection" in opt.test_sets:
    DetectionTests["GAN-Detection"] = {
        'dataroot': '/home/ubuntu/deepspeak-experiments/deepspeak-neurips-2025/video_experiments/FreqNet-DeepfakeDetection/GAN-Detection',
        'no_resize': False,
        'no_crop': True,
    }

# get model
model = freqnet(num_classes=1)
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.cuda()
model.eval()

for testSet in DetectionTests.keys():
    dataroot = DetectionTests[testSet]['dataroot']
    printSet(testSet)

    accs = []
    aps = []
    real_accs = []
    fake_accs = []
    f1s = []
    balanced_accs = []

    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    for v_id, val in enumerate(os.listdir(dataroot)):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes = ''
        opt.no_resize = DetectionTests[testSet]['no_resize']
        opt.no_crop = DetectionTests[testSet]['no_crop']

        acc, ap, real_acc, fake_acc, f1, balanced_acc, _ = validate(model, opt)
        accs.append(acc)
        aps.append(ap)
        real_accs.append(real_acc)
        fake_accs.append(fake_acc)
        f1s.append(f1)
        balanced_accs.append(balanced_acc)

        print(f"({v_id:2d} {val:12}) acc: {acc*100:.1f}; ap: {ap*100:.1f}; real_acc: {real_acc*100:.1f}; fake_acc: {fake_acc*100:.1f}; f1: {f1*100:.1f}; balanced_acc: {balanced_acc*100:.1f}")

    print(f"({v_id+1:2d} {'Mean':10}) acc: {np.mean(accs)*100:.1f}; ap: {np.mean(aps)*100:.1f}; real_acc: {np.mean(real_accs)*100:.1f}; fake_acc: {np.mean(fake_accs)*100:.1f}; f1: {np.mean(f1s)*100:.1f}; balanced_acc: {np.mean(balanced_accs)*100:.1f}")
    print('*'*50)