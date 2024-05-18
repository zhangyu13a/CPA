
# Corruption-assisted Physical Attacks


## Requirements

 - To install requirements:
    - pytorch3d: 0.6.0
    - torch: 1.8.0
    - torchvision: 0.9.0

>📋  you need to download dataset and model weight before running the code:

- dataset 
    - both trainset and testset can be accessed in https://pan.baidu.com/s/17Ct17jdDPOripL79peGIcw (tran)

    - the trainset is made up of two .rar files, which could be unpaced into dataset\trainset with WinRAR software.

- model weight

    - The models are trained on Visdrone2019 and finetuned on dataset of CARLA. The checkpoint of ViTDet can be downloaded here: 链接：https://pan.baidu.com/s/170HGfgbDU1eYCZjXhtvZDQ?pwd=b231 (b231) 

## Optimizing the generator 

To train the generaotr in the paper, run this command:

```train
python attack_ViTDet.py --train_dir <path_to_data> --weightfile <path_to_weight>   
```

## Generating adversarial examples
After training, the weights of the generator are saved at 'auxNetWeight', which can be used to create adversarial examples by running this command:

```test
python generate_adv_examples.py 
```

<!-- ## Contributing

>📋  Pick a licence and describe how to contribute to your code repository.  -->
