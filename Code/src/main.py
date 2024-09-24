import datetime
import time
from argparse import ArgumentParser

import numpy as np
import scipy.linalg as LA
import torch
from helper import data_gen, mean_std_RSNR, unbiased_X, train_model, test_model
from models import TF_LISTA, LISTA, ALISTA

parser = ArgumentParser(description="TF-LISTA")
parser.add_argument(
    "--sparsity", type=int, default=10, help="% of non-zeros in the sparse vector"
)
parser.add_argument("--input_SNR", type=float, default=30, help="Noise Level")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of Epochs")
parser.add_argument(
    "--bits", type=str, default="fp", help="Number of bits used in quantization"
)
parser.add_argument(
    "--quant_epoch",
    type=int,
    default=50,
    help="Epoch after which quantization should begin",
)
parser.add_argument("--quant_scheme", type=str, default="kmeans", help="Quantizer type")
parser.add_argument(
    "--device", type=str, default="cpu", help="The GPU ID if GPU is present"
)
parser.add_argument(
    "--num_layers", type=int, default=15, help="Number of layers in the network"
)
parser.add_argument("--start_epoch", type=int, default=0, help="load previous model")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="learning rate")
parser.add_argument("--thr_val", type=float, default=None, help="Threshold value for the Soft Threshold function.")
parser.add_argument("--num_iter", type=int, default=None, help="Number of iterations for the optimization algorithm.")


args = parser.parse_args()

input_SNR = args.input_SNR
sparsity = args.sparsity
bits = args.bits

if bits != "fp":
    bits = int(bits)


seed = 80
print("Seed:", seed)
rng = np.random.RandomState(seed)

m = 70
n = 100
# create the random matrix
D = rng.normal(0, 1 / np.sqrt(m), [m, n])
D /= np.linalg.norm(D, 2, axis=0)


num_train = 100000
num_test = 100
X_train, Y_train = data_gen(D, input_SNR, sparsity / 100, num_train, rng)
X_test, Y_test = data_gen(D, input_SNR, sparsity / 100, num_test, rng)


num_epochs = args.num_epochs
num_layers = args.num_layers

thr_val = args.thr_val
num_iter = args.num_iter

if thr_val or num_iter:
    if thr_val and num_iter:
        num_epochs = 0 
        num_layers = args.num_iter
    else:
        raise Exception("Please provide thr_val and num_iter for the iterative algorithm.")


if torch.cuda.is_available():
    device = args.device
else:
    device = "cpu"

learning_rate = args.learning_rate

net, loss_list, weight_list, time_list = train_model(
    X_train,
    Y_train,
    D,
    num_epochs,
    num_layers,
    device,
    learning_rate,
    Model=LISTA,
    # Model=TF_LISTA,
    # Model=ALISTA,
    thr_val=thr_val,
)

X_out, X_iter = test_model(net, Y_test, D, device)


mean_RSNR, std_RSNR = mean_std_RSNR(X_test, X_out)

X_u = unbiased_X(X_out, Y_test, D)
u_mean_RSNR, u_std_RSNR = mean_std_RSNR(X_test, X_u)

print(f"RSNR is {mean_RSNR:.3f}±{std_RSNR:.3f}")
print(f"Unbiased RSNR is {u_mean_RSNR:.3f}±{u_std_RSNR:.3f}")
