from __future__ import absolute_import, division, print_function
import models.configs as configs
import logging
import argparse
import os
import random
import numpy as np
import time
from PIL import Image
from datetime import timedelta
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from models.modeling import VisionTransformer, CONFIGS

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}


class FineGrainDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, test=False):
        self.root_dir = root_dir  # 圖片本人路徑
        self.annotations = pd.read_csv(annotation_file)  # 上一步做的CSV
        self.transform = transform  # 定義要做的transform, 含有resize把圖片先resize成依樣
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]  # 取出image id => ex: 0003.jpg
        if self.test == False:
            img = Image.open(os.path.join(self.root_dir, img_id)).convert(
                "RGB")  # 取出image id 對應的圖片本人, 並且轉RGB(等等用transform來轉tensor)
            # y_label = torch.LongTensor(self.annotations.iloc[index, 1])           #取出對應的 image label 並且轉成float tensor
            temp = self.annotations.iloc[index, 1]
            y_label = torch.tensor(self.annotations.iloc[index, 1]).long()
            # print(y_label)
            # print("fetch label :", y_label)
            img = self.transform(img)
            return (img, y_label - 1)
        else:
            # print("fetch testing image id: ", img_id)
            img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
            img = self.transform(img)
            return (img, img_id)

def make_csv(args):
    test_df = pd.DataFrame(columns=["img_name"])
    f = open(args.orderfile_path)
    img_name = []
    for line in f:
        img_name.append(line.replace("\n", ""))
    test_df["img_name"] = img_name
    test_df.to_csv (r'test_csv.csv', index = False, header=True)


def main():
    #training model parameter setting
    config = CONFIGS["ViT-B_16"]
    config.slide_step = 12
    config.split = 'overlap'


    #inference parameter setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default = 4, help="inference batch size")
    parser.add_argument("--orderfile_path", default = "testing_img_order.txt", help="testing_img_order.txt's path")
    parser.add_argument("--classfile_path", default = "classes.txt", help="classes.txt's path")
    parser.add_argument("--test_img_path", default = "./testing_data", help="testing image folder path")
    parser.add_argument("--pretrained_model_path", default = "./output/DL_CV_HW1_checkpoint.bin", help="TransFG training model path")
    parser.add_argument("--device", default = "cuda", help="if no GPU, set it to cpu")
    args = parser.parse_args()

    #make csv for helping predict image in order
    make_csv(args)


    # create lookup table(image id : class label)
    f = open(args.classfile_path)
    classes = []
    mapping = {}
    for line in f:
        classes.append(line.replace("\n", ""))

    for i in range(200):
        mapping[i] = classes[i]

    #inference data transform
    test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                         transforms.CenterCrop((448, 448)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #inference dataset and dataloader prepared
    test_img_path = args.test_img_path
    test_csv_name = "test_csv.csv"
    test_dataset = FineGrainDataset(test_img_path, test_csv_name, test_transform, test=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    #Pretrained TransFG mdoel prepared
    pretrained_model_path = args.pretrained_model_path
    model = VisionTransformer(config, 448, zero_head=True, num_classes=200,
                              smoothing_value=0.0)
    if pretrained_model_path is not None:
        pretrained_model = torch.load(pretrained_model_path)['model']
        model.load_state_dict(pretrained_model)
    model.to(args.device)
    print("------model prepared-------")

    model.eval()

    count = 0
    prediction = []
    imgid = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            print("count:", count)
            count += 1
            test_pred = model(data[0].cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                prediction.append(y)
            for x in data[1]:
                imgid.append(x)

    print("length: ", len(imgid), " , ", len(prediction))
    result = "answer.txt"
    f = open(result, 'w')
    for i in range(len(imgid)):
        content = imgid[i] + " " + mapping[prediction[i]] + "\n"
        f.write(content)
    f.close()



if __name__ == "__main__":
    main()