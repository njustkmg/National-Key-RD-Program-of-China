# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------


from .ViTAE_Window_NoShift.models import ViTAE_Window_NoShift_12_basic_stages4_14
import torch
import torch.nn as nn

def VitAE_v2(pretrained=False, input_shape=[224, 224], num_classes=1000):


    model = ViTAE_Window_NoShift_12_basic_stages4_14(img_size=input_shape[0],
                                                     num_classes=num_classes, window_size=7)

    if pretrained:
        # temp=torch.load("model_data/segformer_b2_weights_voc.pth")
        temp=torch.load("model_data/rsp-vitaev2-s-ckpt.pth")['model']
        print("加载了预训练权重！")
        for k in list(temp.keys()):
            if 'head' in k:
                del temp[k]
        print(model.load_state_dict(temp,strict=False))

    if num_classes!=1000:
        model.head = nn.Linear(model.tokens_dims[-1], num_classes)
    return model

