# Data attribution and licenses

The original software is licensed under MIT. Dataset-derived files retain the following source terms.

## SECOM

`data/scored/secom.csv` contains identifiers, labels, timestamps, fixed splits, fitted risk scores, and risk groups derived from the SECOM dataset by Michael McCann and Adrian Johnston, UCI Machine Learning Repository, 2008. The source is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

- Dataset: https://archive.ics.uci.edu/dataset/179/secom
- DOI: https://doi.org/10.24432/C54305
- License: https://creativecommons.org/licenses/by/4.0/

The transformations are chronological sorting, train/calibration/test partitioning, model scoring, and score grouping. Original measurement files can be acquired with `download_data.py secom`.

## KolektorSDD2

`data/scored/ksdd2.csv` and `data/ksdd2_features.npz` contain image identifiers, binary defect labels, fixed partitions, scores, groups, and 512-dimensional frozen image features derived from KolektorSDD2. These derived data are distributed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0), matching the source dataset license.

- Dataset and license statement: https://www.vicos.si/resources/kolektorsdd2/
- License: https://creativecommons.org/licenses/by-nc-sa/4.0/
- Citation: Jakob Božič, Domen Tabernik, and Danijel Skočaj. “Mixed supervision for surface-defect detection: From weakly to fully supervised learning.” Computers in Industry 129 (2021), 103459. https://doi.org/10.1016/j.compind.2021.103459

The transformations are binary labeling from annotation masks, image resizing and normalization, frozen ResNet-18 feature extraction, stratified partitioning of the official training split, classifier scoring, and score grouping. The official test split is retained. Original images and masks can be acquired with `download_data.py ksdd2`.

## Feature extractor

Image features were computed using TorchVision ResNet-18 with `IMAGENET1K_V1` weights. The implementation is distributed by the TorchVision project under its BSD-style license. The feature preparation program retrieves weights through the TorchVision API.

- Architecture: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. “Deep Residual Learning for Image Recognition.” CVPR (2016), 770–778. https://doi.org/10.1109/CVPR.2016.90
- Implementation and weights: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
