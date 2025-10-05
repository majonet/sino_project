import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

# specs = [
#     SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
#     SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
#     SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
#     SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
#     SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
#     SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
# ]
# specs = [
#     SSDSpec(38, 8,  SSDBoxSizes(15, 30), [1, 2, 1/2]),
#     SSDSpec(19, 16, SSDBoxSizes(30, 60), [1, 2, 3, 1/2, 1/3]),
#     SSDSpec(10, 32, SSDBoxSizes(60, 111), [1, 2, 3, 1/2, 1/3]),
#     SSDSpec(5, 64,  SSDBoxSizes(111, 162), [1, 2, 1/2]),
#     SSDSpec(3, 100, SSDBoxSizes(162, 213), [1, 2, 1/2]),
#     SSDSpec(1, 300, SSDBoxSizes(213, 264), [1, 2, 1/2])
# ]
specs = [
    SSDSpec(19, 16,  SSDBoxSizes(15, 30), [1, 2]),
    SSDSpec(14, 16, SSDBoxSizes(30, 60), [1]),
    SSDSpec(1, 1, SSDBoxSizes(60, 111), [1, 2, 3, 1/2, 1/3]),
    # SSDSpec(1, 6,  SSDBoxSizes(111, 162), [1, 2, 1/2]),
    SSDSpec(2, 10, SSDBoxSizes(162, 213), [1, 2, 1/2]),
    SSDSpec(1, 300, SSDBoxSizes(213, 264), [1, 2, 1/2])
]


priors = generate_ssd_priors(specs, image_size)
