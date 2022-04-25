# COVID-19-Lung-CT-Lesion-Segmentation-Challenge

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

Currently supported models (click for source):
* [`BasicUnet`](https://www.nature.com/articles/s41592-018-0261-2)
* [`DynUnet`](https://arxiv.org/abs/1904.08128)
* [`HighResNet`](https://arxiv.org/abs/1707.01992)
* [`VNet`](https://arxiv.org/pdf/1606.04797.pdf)

## Evaluation
```python run_net.py evaluate --data_folder "COVID-19-20_v2/Validation" --model_folder "runs" --net_type "BasicUnet"``

This command will evaluate the loss for every image in a folder and will store the image loss, mean loss, and standard deviation in `validation_loss.csv` inside the model folder.

## Inference
```python run_net.py infer --data_folder "COVID-19-20_v2/Validation" --model_folder "runs" --net_type "BasicUnet"```

This command will load the best validation model, run inference, and store the predictions at ```./output```. Files will be stored according to the challenge specifications and can be uploaded for submission.
