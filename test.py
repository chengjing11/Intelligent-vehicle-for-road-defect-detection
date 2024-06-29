import os
from models.unet import U_Net
import cv2
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from utils.data_loader_yy import RescaleT
from utils.data_loader_yy import ToTensor
from utils.data_loader_yy import ToTensorLab
from utils.data_loader_yy import SalObjDataset

# from Evaluation.Evaluation import *

from models.myu2net import U2NET  # full size version 173.6 MB
from models.myu2net import U2NETP  # small version u2net 4.7 MB
# from model.u2net import U2NETP  # small version u2net 4.7 MB


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict = (predict > 0.5).float()
    predict_np = predict.cpu().data.numpy() * 255
    im = Image.fromarray(predict_np).convert('L')

    img_name = image_name.split(os.sep)[-1]
    # image = io.imread(image_name)
    # imo = np.resize(predict_np, (image.shape[1], image.shape[0]))
    imo = im.resize((400, 400), resample=Image.NEAREST)

    # aaa = img_name.split(".")
    # bbb = aaa[0:-1]
    # imidx = bbb[0]
    # for i in range(1, len(bbb)):imo.save(d_dir + imidx + '.jpg')
    
    #     imidx = imidx + "." + bbb[i]

    imidx = img_name.split("/")[-1].split('.')[0]

    imo.save(d_dir + imidx + '_mask.png')


def make_dir(txt_path):
    f = open(txt_path, "r", encoding='utf-8')
    txt_tables = f.readlines()
    crack_list = [None] * len(txt_tables)
    mask_list = [None] * len(txt_tables)
    for i in range(len(txt_tables)):
        if i != len(txt_tables) - 1:
            txt_tables[i] = txt_tables[i][:-1]
        else:
            txt_tables[i] = txt_tables[i]
        crack_list[i], mask_list[i] = txt_tables[i].split()
        crack_list[i], mask_list[i] = 'CRACK500/' + crack_list[i], 'CRACK500/' + mask_list[i]
    return crack_list, mask_list


def main(pth):
    # --------- 1. get image path and name ---------
    model_name = 'u2netp'  # u2netp

    image_dir = 'DataSet/Test/Images'
    # image_dir = 'D:\\Desktop\\Graduation Design\\my_u2_net\\DataSet\\Test\\Images'
    # prediction_dir = os.path.join(os.getcwd(), 'test_results', model_name + ' without kd ' + pth + os.sep)
    prediction_dir = os.path.join(os.getcwd(), 'test_results', model_name + '_results_best' + os.sep)
    model_dir = 'saved_models/u2net with kd/best.pth'

    # img_name_list = glob.glob(image_dir + os.sep + '*')
    # mask_name_list = glob.glob('DataSet/Test/Masks' + os.sep + '*')
    img_name_list, mask_name_list = make_dir('CRACK500/test.txt')
    print(img_name_list)

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=mask_name_list,
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=4)

    # --------- 3. model define ---------
    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)

    elif (model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    # net = U_Net(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    acc = 0.  # Accuracy
    SE = 0.  # Sensitivity (Recall)
    SP = 0.  # Specificity
    PC = 0.  # Precision
    F1 = 0.  # F1 Score
    length = 0
    i = 0
    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        GT = data_test['label'][:, 0, :, :].cuda()
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        _, _, _, _, SR, d4, d5, d6 = net(inputs_test)
        # SR = net(inputs_test)
        # SR = net(inputs_test)
        # SR = torch.sigmoid(SR)

        # acc += get_accuracy(SR, GT)
        # SE += get_sensitivity(SR, GT)
        # SP += get_specificity(SR, GT)
        # PC += get_precision(SR, GT)
        # F1 += get_F1(SR, GT)
        # length += 1

        # normalization
        pred = SR[:, 0, :, :]
        # eval_image(pred, inputs_lable)
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)
        i += 1
        # del SR, d2, d3, d4, d5, d6, d7
        del SR
    print(i)
    # acc = acc / length
    # SE = SE / length
    # SP = SP / length
    # PC = PC / length
    # F1 = F1 / length
    # print('[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f' % (
    # acc, SE, SP, PC, F1))


if __name__ == "__main__":
    main('best.pth')
