﻿# COVID-19-Lung-CT-Lesion-Segmentation-Challenge

Challenge hosted on [grand-challenge.org](https://covid-segmentation.grand-challenge.org/)

## Group Members
* [Ilan Valencius](https://github.com/ivalencius)
* [Cole Gvozdas](https://github.com/colegvozdas1)
* William Gibbons

## Architecture
* [[challenge_baseline]](https://github.com/ivalencius/COVID-19-Lung-CT-Lesion-Segmentation-Challenge/tree/main/challenge_baseline): Base segmentation model provided from [MONAI](https://github.com/Project-MONAI/tutorials/tree/master/3d_segmentation/challenge_baseline)
* [[model_info.csv]](https://github.com/ivalencius/COVID-19-Lung-CT-Lesion-Segmentation-Challenge/blob/main/model_info.csv): Stores info for model architecture.
* [[Training_info.csv]](https://github.com/ivalencius/COVID-19-Lung-CT-Lesion-Segmentation-Challenge/blob/main/training_info.csv): Info on parameters used for model training. 
* [[/nets/]](https://github.com/ivalencius/COVID-19-Lung-CT-Lesion-Segmentation-Challenge/blob/main/nets): Stores files defining models used for training.
* [[/sample_data/]](https://github.com/ivalencius/COVID-19-Lung-CT-Lesion-Segmentation-Challenge/tree/main/sample_data): Small subset of training data used to validate code is working before full training.
* [[/for_submit/]](https://github.com/ivalencius/COVID-19-Lung-CT-Lesion-Segmentation-Challenge/blob/main/for_submit): Stores files uploaded for [Biomedical Image Analysis class](https://github.com/ivalencius/Biomedical_Image_Analysis_CSCI3397). 

## Training (and validation after every epoch)
```python run_net.py train --data_folder "COVID-19-20_v2/Train" --model_folder "runs" --net_type "BasicUnet"```

During training, the top three models will be selected based on the per-epoch validation and stored at ```--model_folder```.

Currently supported models:
* ```"BasicUnet"```
* ```"DynUnet"```

## Inference
```python run_net.py infer --data_folder "COVID-19-20_v2/Train" --model_folder "runs" --net_type "BasicUnet"```

This command will load the best validation model, run inference, and store the predictions at ```./output```
