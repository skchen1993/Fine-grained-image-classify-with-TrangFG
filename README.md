NYCU VRDL hw1 : try to used TransFG to do the Fine-grained Recognition

# TransFG: A Transformer Architecture for Fine-grained Recognition
Official PyTorch code for the paper:  [*TransFG: A Transformer Architecture for Fine-grained Recognition*](https://arxiv.org/abs/2103.07976)  
Implementation based on [DeiT](https://arxiv.org/abs/2012.12877) pretrained on ImageNet-1K with distillation fine-tuning will be released soon.

## Framework

![](./TransFG.png)

## Dependencies:
+ Python 3.7.3
+ PyTorch 1.5.1
+ torchvision 0.6.1
+ ml_collections

## Reproduce Submission
To reproduce the submission, do the following step:
### 1. Install required packages

Install dependencies with the following command:

```bash
pip3 install -r requirements.txt
```

### 2. Download pretrained TransFG model

+ [Pretrained TransFG model](https://drive.google.com/file/d/1B03DSv1eGXNyAySEdcqcpoboakF-V7y9/view?usp=sharing)

Please download the model and put it into the `output` folder

### 3. used the following command to inference

```bash
python inference.py --test_img_path {testing image path} --pretrained_model_path {TransFG pretrained model path, default is in "output"}
```

## Report
+ [Report](https://drive.google.com/file/d/1sQqPq8-r65oI9hdXD6oXyIqFXtdQNsgF/view?usp=sharing)




## Citation
If you find our work helpful in your research, please cite it as:

```
@article{he2021transfg,
  title={TransFG: A Transformer Architecture for Fine-grained Recognition},
  author={He, Ju and Chen, Jieneng and Liu, Shuai and Kortylewski, Adam and Yang, Cheng and Bai, Yutong and Wang, Changhu and Yuille, Alan},
  journal={arXiv preprint arXiv:2103.07976},
  year={2021}
}
```

## Acknowledgement

Many thanks to [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) for the PyTorch reimplementation of [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

