# sin_data_test
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sin_data_generate import SinDataset, sin_generate_random
from sin_net import SinClassifyNet, SinRegressionNet

class Sin1DClassify():
    def __init__(self):
        self.class_num = 10

    def train_classify(self, num_epoch, train_loader):
        model = SinClassifyNet(self.class_num, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss_fn = nn.CrossEntropyLoss()
        train_acc = 0.0
        match_num = 0.0
        total_num = 0
        for epoch in range(num_epoch):
            model.train()
            train_acc = 0.0
            for i, (x, y, label) in enumerate(train_loader):
                optimizer.zero_grad()
                x_in = x.unsqueeze(x.dim()).float()  # suitable dimension & double->float
                output = model(x_in)
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()
                total_num += label.shape[0]
                match_num += torch.sum(output.max(1)[1] == label)  # normal mode
            train_acc = match_num.item() / total_num
            match_num = 0
            total_num = 0
        torch.save(model, 'sin_classify_model.pth')
        print('save sin_classify_model.pth')
        return train_acc

    def test_classify(self, test_loader):
        model = torch.load('sin_classify_model.pth')
        model.eval()
        match_num = 0
        inD_match_num = 0
        OOD_match_num = 0
        total_num = 0
        inD_num = 0
        OOD_num = 0
        test_acc = 0.0
        inD_acc = 0.0
        OOD_acc = 0.0
        for i, (x, y, label) in enumerate(test_loader):
            x_in = x.unsqueeze(x.dim()).float()
            output = model(x_in)
            total_num += label.shape[0]
            label_pred = output.max(1)[1]
            match_num += torch.sum(label_pred == label)
            for j, out in enumerate(output):
                if (x[j].item() > -math.pi and x[j].item() < math.pi):
                    inD_num += 1
                    if label_pred[j] == label[j]:
                        inD_match_num += 1
                else:
                    OOD_num += 1
                    if label_pred[j] == label[j]:
                        OOD_match_num += 1
            test_acc = match_num.item() / total_num
            inD_acc = inD_match_num / inD_num
            if OOD_num > 0:
                OOD_acc = OOD_match_num / OOD_num
            else:
                pass
        return test_acc, inD_acc, OOD_acc

    def do_multi_train_and_test(self):
        do_train = True
        do_test = True
        point_num_tests = 1
        epochs_num_tests = 1
        train_pt_num_arr = np.zeros((point_num_tests, 1), dtype='int32')
        train_epochs_arr = np.zeros((epochs_num_tests, 1), dtype='int32')
        train_acc_arr = np.zeros((point_num_tests, epochs_num_tests), dtype='float32')
        test_acc_arr = np.zeros((point_num_tests, epochs_num_tests), dtype='float32')
        inD_acc_arr = np.zeros((point_num_tests, epochs_num_tests), dtype='float32')
        OOD_acc_arr = np.zeros((point_num_tests, epochs_num_tests), dtype='float32')
        for i in range(point_num_tests):
            point_num = 1000 + i * 100
            train_pt_num_arr[i] = point_num
            for j in range(epochs_num_tests):
                if do_train:
                    epochs = 10
                    train_epochs_arr[j] = epochs
                    x, y, class_label, low_err_bound = sin_generate_random(False, point_num, self.class_num)
                    dataset = SinDataset(x, y, class_label)
                    train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
                    train_acc = self.train_classify(epochs, train_loader)
                    print('point_num = %d, epochs = %d, train acc = %f' % (point_num, epochs, train_acc))
                    train_acc_arr[i][j] = train_acc
                    print("Train finish")

                if do_test:
                    test_pt_num = 1000
                    x_test, y_test, class_label_test, low_err_bound = sin_generate_random(True, test_pt_num, self.class_num)
                    dataset = SinDataset(x_test, y_test, class_label)
                    test_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
                    test_acc, inD_acc, OOD_acc = self.test_classify(test_loader)
                    print('i = %d, j = %d, test_acc = %f, inD_acc = %f, OOD_acc = %f' % (
                    i, j, test_acc, inD_acc, OOD_acc))
                    test_acc_arr[i][j] = test_acc
                    inD_acc_arr[i][j] = inD_acc
                    OOD_acc_arr[i][j] = OOD_acc
                    print('Tesh finish')

class Sin1DRegression():
    def __init__(self):
        self.class_num = 10

    def train_regession(self, num_epoch, train_loader):
        model = SinRegressionNet(10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss_fn = torch.nn.MSELoss()
        loss_save = []
        for epoch in range(num_epoch):
            model.train()
            loss = []
            for i,(x,y,label) in enumerate(train_loader):
                optimizer.zero_grad()
                x_in = x.unsqueeze(x.dim()).float()  # suitable dimension & double->float
                output = model(x_in)
                loss = loss_fn(output, y.unsqueeze(y.dim()).float())
                loss.backward()
                optimizer.step()
            loss_save.append(loss.detach().numpy())
        torch.save(model, 'sin_regression_model.pth')
        print('save sin_regression_model.pth')
        return loss_save

    def test_regression(self, test_loader):
        model = torch.load('sin_regression_model.pth')
        model.eval()
        total_num = 0
        inD_num = 0
        OOD_num = 0
        test_acc = 0.0
        inD_err_sum = 0.0
        inD_err_avr = 0.0
        OOD_err_sum = 0.0
        OOD_err_avr = 0.0
        x_data = []
        y_pred = []

        for i, (x, y, label) in enumerate(test_loader):
            x_in = x.unsqueeze(x.dim()).float()
            output = model(x_in)
            for j, out in enumerate(output):
                x_data.append(x[j])
                y_pred.append(out)
                err = math.fabs(out - y[j])
                if (x[j].item() > -math.pi and x[j].item() < math.pi):
                    inD_num += 1
                    inD_err_sum += err
                else:
                    OOD_num += 1
                    OOD_err_sum += err
            inD_err_avr = inD_err_sum / inD_num
            if (OOD_num > 0):
                OOD_err_avr = OOD_err_sum / OOD_num
            else:
                pass
        return inD_err_avr, OOD_err_avr, x_data, y_pred

    def sin_regression_train_test(self):
        do_train = True
        do_test = True
        train_pt_num = 5000
        # train_data
        x, y, class_label, low_err_bound = sin_generate_random(False, train_pt_num, self.class_num)
        dataset_test = SinDataset(x, y, class_label)
        train_loader = DataLoader(dataset=dataset_test, batch_size=64, shuffle=False)

        # test data
        test_pt_num = 1000
        x_test, y_test, class_label, low_err_bound = sin_generate_random(False, train_pt_num, self.class_num)
        dataset_train = SinDataset(x_test, y_test, class_label)
        test_loader = DataLoader(dataset=dataset_train, batch_size=64, shuffle=False)

        for i in range(5):
            if do_train:
                epochs = 100
                loss_save = self.train_regession(epochs, train_loader)
                print("Train finish")

            if do_test:
                inD_err, OOD_err, x_data, y_pred = self.test_regression(test_loader)
                print('i = %d, low_err_bound = %f, inD_err = %f, OOD_err = %f' % (i, low_err_bound, inD_err, OOD_err))
                print('Test finish')
        plt.plot(x_data, y_test, '.b')
        plt.plot(x_data, y_pred, '.r')
        plt.show()

if __name__ == "__main__":
    sin_classify = Sin1DClassify()
    sin_classify.do_multi_train_and_test()
    # sin_regression = Sin1DRegression()
    # sin_regression.sin_regression_train_test()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # train_pt_num_arr,train_epochs_arr = np.meshgrid(train_pt_num_arr,train_epochs_arr)
    # ax.plot_surface(train_pt_num_arr,train_epochs_arr,train_acc_arr)
    # plt.show()

    print('All finish')

