import datetime
import time

import numpy as np
import scipy.linalg as LA
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def soft_thr(input, theta):
    return F.relu(input - theta) - F.relu(-input - theta)


def unbiased_X(X_out, Y_test, A, epsilon=1e-4):
    x_inv = np.zeros_like(X_out)
    for i in range(Y_test.shape[1]):
        A_s = A[:, np.abs(X_out[:, i]) > epsilon]
        tmp = LA.pinv(A_s) @ Y_test[:, i]
        x_inv[np.abs(X_out[:, i]) > epsilon, i] = tmp
    return x_inv


def mean_std_RSNR(X_test, X_out):
    n = X_test.shape[1]
    lista_snr_list = [compute_RSNR(X_test[:, i], X_out[:, i]) for i in range(n)]
    return np.mean(lista_snr_list), np.std(lista_snr_list)


def compute_RSNR(X_test, X_out):
    err = LA.norm(X_test - X_out)
    return -20 * np.log10(err / LA.norm(X_test))


def pes(x, x_est):
    d = []
    for i in range(x.shape[1]):
        M = max(np.sum(x[:, i] != 0), np.sum(x_est[:, i] != 0))
        pes_ = (M - np.sum((x[:, i] != 0) * (x_est[:, i] != 0))) / M
        if not np.isnan(pes_):
            d.append(pes_)
        else:
            print(M)
            print("nan is found here")
    return np.mean(d), np.std(d)


def data_gen(D, SNR, k, p, rng):

    m, n = D.shape
    x = rng.normal(0, 1, (n, p)) * rng.binomial(1, k, (n, p))

    for i in range(p):
        if np.linalg.norm(x[:, i]) == 0:
            print("ERROR, zero norm test data sample")

    y = D @ x

    y1 = D @ x
    noise = rng.normal(0, 1, y1.shape)

    print("Input noise SNR is:", SNR)

    noise_level = LA.norm(y1, 2) / ((10 ** (SNR / 20)) * LA.norm(noise, 2))
    y = y1 + noise_level * noise

    print(noise_level)

    snr = 20 * np.log10(LA.norm(y1, 2) / LA.norm(noise_level * noise, 2))
    print("Added noise SNR is:", snr)

    return x, y


class dataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :]


def train_model(X, Y, D, numEpochs, numLayers, device, learning_rate, Model, thr_val=None):

    m, n = D.shape

    train_size = Y.shape[1]
    batch_size = 250
    print("Total dataset size is ", train_size)
    if train_size % batch_size != 0:
        print("Bad Training dataset size")

    # convert the data into tensors
    y_t = torch.from_numpy(Y.T)
    y_t = y_t.float().to(device)

    d_t = torch.from_numpy(D.T)
    d_t = d_t.float().to(device)

    # we need to use ISTA to get X
    x_t = torch.from_numpy(X.T)
    x_t = x_t.float().to(device)

    valid_size = int(0.2 * train_size)
    dataset_train = dataset(x_t[:-valid_size, :], y_t[:-valid_size, :])
    dataset_valid = dataset(x_t[-valid_size:, :], y_t[-valid_size:, :])
    print("DataSet size is: ", dataset_train.__len__())
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    # compute the max eigen value of the D'*D
    alpha = (np.linalg.norm(D, 2) ** 2) * 1.001

    # Numpy Random State if rng (passed through arguments)
    net = Model(m, n, D, numLayers, alpha=alpha, device=device, thr_val=thr_val)
    net = net.float().to(device)
    # -------------------*****************-_-----------------
    net.weights_init()

    # build the optimizer and criterion
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # list of losses at every epoch
    train_loss_list = []
    valid_loss_list = []
    time_list = []
    step_list = []
    thr_list = []

    start = time.time()

    t = datetime.datetime.now().timetuple()

    net.to(device)
    tb = SummaryWriter(f"runs/{Model}_{t[1]}-{t[2]}-{t[3]}-{t[4]}")
    best_loss = 1e6
    lr = learning_rate
    # ------- Training phase --------------
    for epoch in tqdm(range(numEpochs)):

        if epoch == round(numEpochs * 0.5):
            lr = learning_rate * 0.2
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
        elif epoch == round(numEpochs * 0.75):
            lr = learning_rate * 0.02
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
        else:
            pass
        thr_list.append(net.thr.data.clone().cpu().numpy())
        step_list.append(net.step.data.clone().cpu().numpy())
        time_list.append(time.time() - start)

        tot_loss = 0
        net.train()

        for data in data_loader_train:
            X_GT_batch, Y_batch = data
            X_batch_hat = net(Y_batch.float())
            loss = sum(
                (criterion(X_batch_hat[i].float(), X_GT_batch.float()) for i in range(numLayers))
            )
            # compute the losss
            loss /= numLayers
            tot_loss += loss.detach().cpu().data
            optimizer.zero_grad()  # clear the gradients
            loss.backward()  # compute the gradiettns

            optimizer.step()  # Update the weights
            net.zero_grad()

        if epoch % 1 == 0:
            print("Training - Epoch: {}, Loss: {}".format(epoch, tot_loss))
            print("step sizes", net.step.T)
            print("lambdas", net.thr.T)

        tb.add_scalar("training Loss", tot_loss / 4, epoch)

        # Validation stage
        with torch.no_grad():
            train_loss_list.append(tot_loss.detach().data / 4)
            tot_loss = 0
            for data in data_loader_valid:

                X_GT_batch, Y_batch = data
                X_batch_hat = net(Y_batch.float())  # get the outputs
                loss = sum((
                    criterion(X_batch_hat[i].float(), X_GT_batch.float())
                    for i in range(numLayers)
                ))
                # compute the losss
                loss /= numLayers
                tot_loss += loss.detach().cpu().data
            valid_loss_list.append(tot_loss)

            if best_loss > tot_loss:
                best_loss = tot_loss

        if epoch % 1 == 0:
            print("Validation - Epoch: {}, Loss: {}".format(epoch, tot_loss))

        tb.add_scalar("validation Loss", tot_loss, epoch)
        tb.add_histogram("thresholds", net.thr, epoch)
        tb.add_histogram("steps", net.step, epoch)
        tb.add_scalar("Learning rate", lr, epoch)

    weight_list = [thr_list, step_list]
    loss_list = [train_loss_list, valid_loss_list]
    print("*" * 40)
    print(f"Training time: {time.time() - start:.3f}")
    return net, loss_list, weight_list, time_list


def test_model(net, Y, D, device):

    # convert the data into tensors
    y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        y_t.view(1, -1)
    y_t = y_t.float().to(device)
    d_t = torch.from_numpy(D.T)
    d_t = d_t.float().to(device)

    with torch.no_grad():
        # Compute the output
        net.eval()
        X_iter = net(y_t.float())
        if len(Y.shape) <= 1:
            X_iter = X_iter.view(-1)
        X_final = X_iter[-1].cpu().numpy()
        X_final = X_final.T

    return X_final, X_iter