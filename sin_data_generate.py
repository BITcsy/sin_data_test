from torch.utils.data import Dataset, DataLoader
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch

class SinDataset(Dataset):
    def __init__(self, x, y, class_label):
        self.x = x
        self.y = y
        self.class_label = class_label
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.class_label[index]
    def __len__(self):
        return len(self.x)

def sin_generate_random(OOD, point_num, class_num):
    A = 1.0
    B = 2.0
    epsilon = 0.01
    omega = 1.0
    x = []
    y = []
    class_label = []
    class_reso = (2.0 * (A + B) + 0.01) / class_num
    err_sum = 0.0
    OOD_ratio = 1.5 if OOD else 1.0
    for i in range(point_num):
        x_temp = random.uniform(-math.pi * OOD_ratio, math.pi * OOD_ratio)
        x.append(x_temp)
        random_err = random.uniform(-epsilon, epsilon)
        y_temp = A * math.sin(x_temp * omega) + B * math.sin(x_temp * 3.0 * omega) + random_err
        y.append(y_temp)
        class_temp = int((y_temp + A + B) / class_reso)
        class_label.append(class_temp)
        err_sum += math.fabs(random_err)
    return x,y,class_label,err_sum / point_num

class SinDataset2D(Dataset):
    def __init__(self, xy_matrix, z):
        self.xy = torch.tensor(xy_matrix)
        self.z = torch.tensor(z)
    def __getitem__(self, index):
        return self.xy[index], self.z[index]
    def __len__(self):
        return len(self.z)

def sin_generate_random_2d(OOD, point_num, area_size, reso):
    # generate x,y to z by using A * sin(x) + B * cos(y) function
    # data_sin2D is a list of [x,y], [x1, y1, x2, y2 ....]

    A = 1.0
    B = 2.0
    epsilon = 0.01
    omegaX = 1.0
    omegaY = 2.0
    OOD_ratio = 1.5 if OOD else 1.0
    dist2bound = (area_size - 1) / 2 * reso
    ub = math.pi * OOD_ratio # upper bound
    lb = -math.pi * OOD_ratio # low bound
    data_sin2D = []
    data_z = []
    err_sum = 0.0
    for i in range(point_num):
        random_err = 0.0
        nums = []
        z_temp = 0.0
        if area_size == 1:
            x_temp = random.uniform(lb + dist2bound, ub - dist2bound)
            y_temp = random.uniform(lb + dist2bound, ub - dist2bound)
            nums.append(x_temp)
            nums.append(y_temp)
            random_err = random.uniform(-epsilon, epsilon)
            z_temp = A * math.sin(omegaX * x_temp) + B * math.cos(omegaY * y_temp) + random_err
        else:
            x_lb = random.uniform(lb + dist2bound, ub - dist2bound) - dist2bound
            y_lb = random.uniform(lb + dist2bound, ub - dist2bound) - dist2bound
            for k in range(area_size):
                for t in range(area_size):
                    x_temp = x_lb + k * reso
                    y_temp = y_lb + t * reso
                    nums.append(x_temp)
                    nums.append(y_temp)
                    random_err = random.uniform(-epsilon, epsilon)
                    if k == t and k == (area_size - 1) / 2: # middle num
                        z_temp = A * math.sin(omegaX * x_temp) + B * math.cos(omegaY * y_temp) + random_err
        data_sin2D.append(nums)
        data_z.append(z_temp)
        err_sum += math.fabs(random_err)
    return data_sin2D, data_z, err_sum / point_num

def get_middle_ptxy(data_sin2D, area_size):
    x_mid_pts = []
    y_mid_pts = []
    for i in range(len(data_sin2D)):
        xy_matrix = data_sin2D[i]
        x_mid_temp = xy_matrix[int(area_size * area_size - 1)] # (n^2 - 1) / 2 * 2
        y_mid_temp = xy_matrix[int(area_size * area_size)]
        x_mid_pts.append(x_mid_temp)
        y_mid_pts.append(y_mid_temp)
    return x_mid_pts, y_mid_pts

if __name__ == "__main__":
    OOD = False
    area_size = 5

    data_sin2D, data_z, avr_err = sin_generate_random_2d(OOD, 1000, area_size, 0.2)

    # print("data_sin2D")
    # print(data_sin2D)
    # print("data_z")
    # print(data_z)
    # print("avr_err")
    # print(avr_err)

    # test data 2d matrix
    # xy = data_sin2D[0]
    # x_plot = []
    # y_plot = []
    # for i in range(len(xy)):
    #     x_plot.append(xy[i][0])
    #     y_plot.append(xy[i][1])
    # plt.plot(x_plot, y_plot, 'o')
    # plt.show()

    # test xyz
    x_mid_pts, y_mid_pts = get_middle_ptxy(data_sin2D, area_size)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_mid_pts, y_mid_pts, data_z, c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


