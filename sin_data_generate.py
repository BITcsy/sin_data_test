from torch.utils.data import Dataset, DataLoader
import random
import math

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