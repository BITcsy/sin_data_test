# sin_data_test
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sin_data_generate import SinDataset, sin_generate_random, SinDataset2D, sin_generate_random_2d
from sin_net import SinClassifyNet, SinRegressionNet, Sin2DMLP
import pickle

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

class Sin2DRegression():
    def __init__(self):
        self.area_size = 5   # length and width of the data matrix
        self.area_size2 = self.area_size**2
        self.reso = 0.2
        self.OOD = True
        self.train_pt_num = 1000
        self.test_pt_num = 500
        self.model_hidden_size = 20
        self.do_train = True
        self.do_test = True
        self.epochs = 100
        self.no_valid_data = False   # generate new data or using pickle

    def get_configs(self):
        configs = dict()
        configs["area_size"] = self.area_size
        configs["reso"] = self.reso
        configs["OOD"] = self.OOD
        configs["train_pt_num"] = self.train_pt_num
        configs["test_pt_num"] = self.test_pt_num
        configs["model_hidden_size"] = self.model_hidden_size
        configs["do_train"] = self.do_train
        configs["do_test"] = self.do_test
        configs["epochs"] = self.epochs
        configs["no_valid_data"] = self.no_valid_data
        return configs

    def set_configs(self, configs):
        self.area_size = configs["area_size"]  # length and width of the data matrix
        self.area_size2 = self.area_size ** 2
        self.reso = configs["reso"]
        self.OOD = configs["OOD"]
        self.train_pt_num = configs["train_pt_num"]
        self.test_pt_num = configs["test_pt_num"]
        self.model_hidden_size = configs["model_hidden_size"]
        self.do_train = configs["do_train"]
        self.do_test = configs["do_test"]
        self.epochs = configs["epochs"]
        self.no_valid_data = configs["no_valid_data"]  # generate new data or using pickle

    def train(self, epochs, train_loader):
        model = Sin2DMLP(self.area_size, self.model_hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss_fn = torch.nn.MSELoss()
        loss_save = []
        for i in range(epochs):
            model.train()
            loss = []
            for idx,(xy,z) in enumerate(train_loader):
                optimizer.zero_grad()
                # xy_in = xy.unsqueeze(xy.dim()).float()  # suitable dimension & double->float
                output = model(xy)
                loss = loss_fn(output, z.unsqueeze(z.dim()).float())
                loss.backward()
                optimizer.step()
            loss_save.append(loss.detach().numpy())
        torch.save(model, 'sin_2d_model.pth')
        print('save sin_2d_model.pth')
        return loss_save

    def test(self, test_loader):
        model = torch.load('sin_2d_model.pth')
        model.eval()
        inD_num = 0
        OOD_num = 0
        inD_err_sum = 0.0
        inD_err_avr = 0.0
        OOD_err_sum = 0.0
        OOD_err_avr = 0.0
        x_data = []
        y_data = []
        z_data = []
        z_pred = []

        for i,(xy,z) in enumerate(test_loader):
            xy_in = xy.float()
            output = model(xy_in)
            for j, out in enumerate(output):
                err = math.fabs(out - z[j])
                x_data.append(xy[j][self.area_size2 - 1].item())
                y_data.append(xy[j][self.area_size2].item())
                z_data.append(z[j].item())
                z_pred.append(out.item())
                if (xy[j][self.area_size2 - 1].item() > -math.pi and xy[j][self.area_size2 - 1].item() < math.pi and
                        xy[j][self.area_size2] > -math.pi and xy[j][self.area_size2].item() < math.pi):   # batch size, area_size^2 * 2
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
        print("inD num = %d, inD avr err = %lf, OOD num = %d, OOD avr err = %lf" % (inD_num, inD_err_avr, OOD_num, OOD_err_avr))
        return inD_err_avr, OOD_err_avr, x_data, y_data, z_data, z_pred

    def train_test(self):
        data_sin2D_train, data_z_train = [], []
        data_sin2D_test, data_z_test = [], []
        avr_train_data_err, avr_test_data_err = 0.0, 0.0
        configs = self.get_configs()
        if self.no_valid_data:
            data_sin2D_train, data_z_train, avr_train_data_err = sin_generate_random_2d(False, self.train_pt_num, self.area_size, self.reso)
            with open("dataset/train_data.pkl", "wb") as f:
                pickle.dump(data_sin2D_train, f)
                pickle.dump(data_z_train, f)
                pickle.dump(avr_train_data_err, f)
                pickle.dump(configs, f)

            data_sin2D_test, data_z_test, avr_test_data_err = sin_generate_random_2d(self.OOD, self.test_pt_num, self.area_size, self.reso)
            with open("dataset/test_data.pkl", "wb") as f:
                pickle.dump(data_sin2D_test, f)
                pickle.dump(data_z_test, f)
                pickle.dump(avr_test_data_err, f)
                pickle.dump(configs, f)
        else:
            with open("dataset/train_data.pkl", "rb") as f:
                data_sin2D_train = pickle.load(f)
                data_z_train = pickle.load(f)
                avr_train_data_err = pickle.load(f)
                configs = pickle.load(f)
            with open("dataset/test_data.pkl", "rb") as f:
                data_sin2D_test = pickle.load(f)
                data_z_test = pickle.load(f)
                avr_test_data_err = pickle.load(f)
            self.set_configs(configs)

        dataset_train = SinDataset2D(data_sin2D_train, data_z_train)
        dataset_test = SinDataset2D(data_sin2D_test, data_z_test)
        train_loader = DataLoader(dataset=dataset_train, batch_size=64, shuffle=False)
        test_loader = DataLoader(dataset=dataset_test, batch_size=64, shuffle=False)

        if self.do_train:
            loss_save = self.train(self.epochs, train_loader)
            print("sin 2d train finished, avr train err = %lf, last loss = %lf" % (avr_train_data_err, loss_save[-1]))
        if self.do_test:
            inD_err_avr, OOD_err_avr, x_data, y_data, z_data, z_pred = self.test(test_loader)
            print("sin 2d test finished, avr data err = %lf, inD err = %f, OOD err = %f" % (avr_test_data_err, inD_err_avr, OOD_err_avr))
            # plot figure
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(x_data, y_data, z_data, c='b', marker='o')
            # ax.scatter(x_data, y_data, z_pred, c='r', marker='o')
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # plt.show()

if __name__ == "__main__":
    # sin_classify = Sin1DClassify()
    # sin_classify.do_multi_train_and_test()
    # sin_regression = Sin1DRegression()
    # sin_regression.sin_regression_train_test()
    sin_2d = Sin2DRegression()
    sin_2d.train_test()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # train_pt_num_arr,train_epochs_arr = np.meshgrid(train_pt_num_arr,train_epochs_arr)
    # ax.plot_surface(train_pt_num_arr,train_epochs_arr,train_acc_arr)
    # plt.show()

    print('All finish')

