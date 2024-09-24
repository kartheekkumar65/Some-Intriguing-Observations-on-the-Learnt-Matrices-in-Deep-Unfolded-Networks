import numpy as np
import scipy.linalg as LA
import torch
from helper import soft_thr
from torch import nn


class LISTA(nn.Module):
    def __init__(self, m, n, Dict, numIter, alpha, device, thr_val=None):
        super(LISTA, self).__init__()
        self._W = nn.Linear(in_features=m, out_features=n, bias=False)
        self._S = nn.Linear(in_features=n, out_features=n, bias=False)
        self.thr = nn.Parameter(torch.rand(numIter, 1), requires_grad=True)
        self.step = nn.Parameter(1 * torch.ones(numIter, 1), requires_grad=True)
        self.numIter = numIter
        self.A = Dict
        self.alpha = alpha
        self.device = device
        self.thr_val = thr_val

    # custom weights initialization called on network
    def weights_init(self):
        a = self.A
        alpha = self.alpha
        s = torch.from_numpy(np.eye(a.shape[1]) - (1 / alpha) * np.matmul(a.T, a))
        s = s.float().to(self.device)
        b = torch.from_numpy((1 / alpha) * a.T)
        b = b.float().to(self.device)

        # random gaussian initialization of S
        # S = torch.normal(mean = torch.zeros_like(S), std = 0.007)
        # B = torch.normal(mean = torch.zeros_like(B), std = 0.0564)
        
        # self._S.weight = nn.Parameter(S)
        # self._W.weight = nn.Parameter(B)
        
        if self.thr_val:
            thr = torch.ones(self.numIter, 1) * self.thr_val/alpha
        else:
            thr = torch.ones(self.numIter, 1) * 0.1 / alpha

        self._S.weight = nn.Parameter(s)
        self._W.weight = nn.Parameter(b)
        self.thr.data = nn.Parameter(thr.to(self.device))

    def forward(self, y):
        # LISTA implementation
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device=self.device)

        for iter in range(self.numIter):
            d = soft_thr(self._W(y) + self._S(d), self.thr[iter])
            x.append(d)
        return x


class TF_LISTA(nn.Module):
    def __init__(self, m, n, D, numIter, alpha, device, thr_val=None):
        super(TF_LISTA, self).__init__()
        self.thr = nn.Parameter(torch.rand(numIter, 1), requires_grad=True)
        self.step = nn.Parameter(1 * torch.ones(numIter, 1), requires_grad=True)
        self.numIter = numIter
        self.A = torch.tensor(D, dtype=torch.float32, device=device)
        self.W = torch.tensor(LA.inv(D @ D.T) @ D, dtype=torch.float32, device=device)
        self.alpha = alpha
        self.device = device
        self.numIter = numIter
        self.thr_val = thr_val

    # custom weights initialization called on network
    def weights_init(self):
        alpha = self.alpha

        if self.thr_val:
            thr = torch.ones(self.numIter, 1) * self.thr_val
        else:
            thr = torch.ones(self.numIter, 1) * 0.1 / alpha

        self.thr.data = nn.Parameter(thr.to(self.device))

    def forward(self, y):
        # TF-LISTA implementations
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device=self.device)

        for iter_ in range(self.numIter):
            d = soft_thr(
                d - self.step[iter_] * torch.mm(self.W.T, (torch.mm(self.A, d.T) - y.T)).T,
                self.thr[iter_],
            )
            x.append(d)
        return x
    

class ALISTA(nn.Module):
    def __init__(self, m, n, D, numIter, alpha, device,  thr_val=None):
        super(ALISTA, self).__init__()
        self.thr = nn.Parameter(torch.rand(numIter,1), requires_grad=True)
        self.step = nn.Parameter(1 *torch.ones(numIter,1), requires_grad=True)
        self.numIter = numIter
        self.A = torch.tensor(D, dtype=torch.float32, device = device)
        W =  LA.inv(D @ D.T) @ D
        for i in range(n):
            w = W[:, i]
            w /= w.dot(D[:, i])
            W[:, i] = w
        self.W = torch.tensor(W, dtype=torch.float32, device = device)
        self.alpha = alpha
        self.device = device
        self.numIter = numIter
        self.thr_val = thr_val
        
    # custom weights initialization called on network
    def weights_init(self):
        alpha = self.alpha
        if self.thr_val:
            thr = torch.ones(self.numIter, 1) * self.thr_val
        else:
            thr = torch.ones(self.numIter, 1) * 0.1 / alpha
        self.thr.data = nn.Parameter(thr.to(self.device))


    def forward(self, y):
        # ALISTA implementations
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        for iter_ in range(self.numIter):
            d = soft_thr( d - self.step[iter_] * torch.mm(self.W.T, (torch.mm(self.A, d.T) - y.T)).T, self.thr[iter_])
            x.append(d)
        return x

