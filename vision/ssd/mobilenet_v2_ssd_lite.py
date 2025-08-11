import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from ..nn.mobilenet_v2 import MobileNetV2, InvertedResidual

from .ssd import SSD, GraphPath
from .predictor import Predictor
from .config import mobilenetv1_ssd_config as config

import torch
import torch.nn as nn

class SimpleSSD(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSSD, self).__init__()
        # Simple feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # downsample
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Detection heads
        self.loc_head = nn.Conv2d(128, 4 * 4, kernel_size=3, padding=1)   # 4 boxes per location
        self.cls_head = nn.Conv2d(128, 4 * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.features(x)
        locs = self.loc_head(x)
        confs = self.cls_head(x)

        # reshape outputs for SSD format
        locs = locs.permute(0, 2, 3, 1).contiguous()
        confs = confs.permute(0, 2, 3, 1).contiguous()

        return locs, confs

def create_mobilenetv2_ssd_lite(num_classes):
    return SimpleSSD(num_classes)
# -----------------------------------------------------------------------------------------------------------
# def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
#     """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
#     """
#     ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
#     return Sequential(
#         Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
#                groups=in_channels, stride=stride, padding="same"),
#         BatchNorm2d(in_channels),
#         ReLU(),
#         Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
#     )


# def create_mobilenetv2_ssd_lite(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False):
#     base_net = MobileNetV2(width_mult=width_mult, use_batch_norm=use_batch_norm,
#                            onnx_compatible=onnx_compatible).features

#     source_layer_indexes = [
#         GraphPath(14, 'conv', 3),
#         19,
#     ]
#     extras = ModuleList([
#         InvertedResidual(1500, 800, stride=2, expand_ratio=0.2),
#         InvertedResidual(800, 600, stride=2, expand_ratio=0.25),
#         InvertedResidual(600, 300, stride=2, expand_ratio=0.5),
#         InvertedResidual(300, 64, stride=2, expand_ratio=0.25)
#     ])

#     regression_headers = ModuleList([
#         SeperableConv2d(in_channels=round(576 * width_mult), out_channels=6 * 4,
#                         kernel_size=5, padding=1, onnx_compatible=False),
#         SeperableConv2d(in_channels=1500, out_channels=6 * 4, kernel_size=5, padding=1, onnx_compatible=False),
#         SeperableConv2d(in_channels=800, out_channels=6 * 4, kernel_size=5, padding=1, onnx_compatible=False),
#         SeperableConv2d(in_channels=600, out_channels=6 * 4, kernel_size=5, padding=1, onnx_compatible=False),
#         SeperableConv2d(in_channels=300, out_channels=6 * 4, kernel_size=5, padding=1, onnx_compatible=False),
#         # SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=5, padding=1, onnx_compatible=False),
#         Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
#     ])

#     classification_headers = ModuleList([
#         SeperableConv2d(in_channels=round(576 * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1),
#         SeperableConv2d(in_channels=1500, out_channels=6 * num_classes, kernel_size=5, padding=1),
#         SeperableConv2d(in_channels=800, out_channels=6 * num_classes, kernel_size=5, padding=1),
#         SeperableConv2d(in_channels=600, out_channels=6 * num_classes, kernel_size=5, padding=1),
#         SeperableConv2d(in_channels=300, out_channels=6 * num_classes, kernel_size=5, padding=1),
#         # SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=5, padding=1),
#         Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
#     ])

#     return SSD(num_classes, base_net, source_layer_indexes,
#                extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
