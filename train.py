import dataset
import model
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import sys


def cnn1_loss(Y, ground_img):
    loss = nn.L1Loss(reduction='mean')
    l = loss(Y, ground_img)
    return l


def cnn2_loss(Y, numerator_img, denominator_img):
    loss1 = nn.L1Loss(reduction='mean')
    loss2 = nn.L1Loss(reduction='mean')
    Y_m, Y_d = torch.split(Y, 1, dim=1)
    l1 = loss1(Y_m, numerator_img)
    l2 = loss2(Y_d, denominator_img)
    l = l1 + l2
    return l


def train(net, img_dir, split_nums, epochs, lr, device):
    net.to(device)
    start_time = time.time()
    optimizer1 = torch.optim.SGD(lr=lr, params=net.parameters(), weight_decay=0.0005, momentum=0.9)
    total_loss = 0
    train_loss = []
    test_loss = []
    max_loss = sys.maxsize
    # train cnn1
    for epoch in range(epochs):
        read_time = 0
        cal_time = 0
        if epoch != 0:
            end_time = time.time()
            print(f'单个epoch运行时间: {end_time - start_time} 秒')
            start_time = time.time()

        j = 1
        for k in range(split_nums):
            read_start = time.time()
            train_data = dataset.Dataset(True, img_dir, split_nums, k)
            train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
            read_end = time.time()
            read_time += (read_end - read_start)
            net.train()
            cal_start = time.time()
            for _, (input_img, ground_img, numerator_img, denominator_img) in enumerate(train_iter):
                input_img, ground_img, numerator_img, denominator_img = input_img.to(device), ground_img.to(
                    device), numerator_img.to(device), denominator_img.to(device)
                optimizer1.zero_grad()
                cnn1_out, cnn2_out = net(input_img)
                l1 = cnn1_loss(cnn1_out, ground_img)
                l2 = cnn2_loss(cnn2_out, numerator_img, denominator_img)
                l = l1 + l2
                if l == 'nan':
                    print(j)
                l.mean().backward()
                total_loss += l.item()
                optimizer1.step()
                j += 1
                torch.cuda.empty_cache()
            cal_end = time.time()
            cal_time += (cal_end - cal_start)
        train_loss.append((total_loss / j))
        total_loss = 0
        j = 1
        for k in range(split_nums):
            read_start = time.time()
            test_data = dataset.Dataset(False, img_dir, split_nums, k)
            test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
            read_end = time.time()
            read_time += (read_end - read_start)

            cal_start = time.time()
            net.eval()
            with torch.no_grad():
                for _, (input_img, ground_img, numerator_img, denominator_img) in enumerate(test_iter):
                    input_img, ground_img, numerator_img, denominator_img = input_img.to(device), ground_img.to(
                        device), numerator_img.to(device), denominator_img.to(device)

                    cnn1_out, cnn2_out = net(input_img)
                    l1 = cnn1_loss(cnn1_out, ground_img)
                    l2 = cnn2_loss(cnn2_out, numerator_img, denominator_img)
                    l = l1 + l2
                    total_loss += l.item()
                    j += 1

            cal_end = time.time()
            cal_time += (cal_end - cal_start)
        test_loss.append((total_loss / j))

        print(f'epoch:{epoch}   net_loos:{total_loss / j}')
        print(f'read_time: {read_time}')
        print(f'cal_time: {cal_time}')
        np.savetxt('train_loss', np.array(train_loss))
        np.savetxt('test_loss', np.array(test_loss))

        if (max_loss > total_loss / j):
            torch.save(net.state_dict(), 'net_params')
            print(f'last params: {max_loss}')
            print("save params")
            max_loss = total_loss / j

        total_loss = 0


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.02)
        nn.init.zeros_(m.bias)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float16)
    batch_size = 2
    split_nums = 40
    lr = 0.0001
    epochs = 300
    img_dir = "../../datasets/cal_wrapped_phase"
    device = "cuda:0"

    net = model.ResNetFPN()
    net.apply(init_normal)

    # net.load_state_dict(torch.load('net_params'))
    train(net, img_dir, split_nums, epochs, lr, device)
