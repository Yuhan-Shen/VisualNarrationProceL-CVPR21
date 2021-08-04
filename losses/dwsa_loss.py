import numpy as np
import torch
import torch.nn.functional as F
from numba import jit
from torch.autograd import Function

@jit(nopython = True)
def compute_dwsa_loss(C, gamma):
    '''
       compute differentiable weak sequence alignment loss
    '''
    M, N = C.shape
    D = np.ones((M, N)) * 100 
    D[0, :] = C[0, :]

    for i in range(1, M):
        for j in range(0, N):
            if j % 2 == 0:
                last_row = D[i-1, :j+1]
            else:
                last_row = D[i-1, :j]

            rmin = np.min(last_row)
            rsum = np.sum(np.exp( - (last_row - rmin) / gamma))
            softmin = -gamma * np.log(rsum) + rmin 
            D[i, j] = softmin + C[i, j]

    last_row = D[-1, :]
    rmin = np.min(last_row)
    rsum = np.sum(np.exp( - (last_row - rmin) / gamma))
    softmin = -gamma * np.log(rsum) + rmin

    return D, softmin

@jit(nopython = True)
def compute_dwsa_loss_backward(C, D, gamma):
    M, N = C.shape
    #E = np.exp( -D / gamma)
    G = np.zeros((M, N), dtype=C.dtype)
    final_row = D[-1, :] - D[-1, :].min()
    exp_final_row = np.exp( -final_row / gamma)
    G[-1, :] = exp_final_row / (exp_final_row.sum()+1e-4)

    for i in range(M-2, -1, -1):
        for j in range(N-1, -1, -1):
            if j % 2 == 0:
                delta = D[i+1, j:] - C[i+1, j:] - D[i, j]
                delta = G[i+1, j:] * np.exp(delta/gamma)
                G[i, j] = delta.sum()
            else:
                delta = D[i+1, j+1:] - C[i+1, j+1:] - D[i, j] 
                delta = G[i+1, j+1:] * np.exp(delta/gamma)
                G[i, j] = delta.sum()

    return G

class _DWSALoss(Function):
    @staticmethod
    def forward(ctx, C, gamma):
        dev = C.device
        dtype = C.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)
        C_ = C.detach().cpu().numpy()
        g_ = gamma.item()
        D, softmin = compute_dwsa_loss(C_, g_)
        D = torch.Tensor(D).to(dev).type(dtype)
        softmin = torch.Tensor([softmin]).to(dev).type(dtype)
        ctx.save_for_backward(C, D, softmin, gamma)
        return softmin

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        C, D, softmin, gamma = ctx.saved_tensors
        C_ = C.detach().cpu().numpy()
        D_ = D.detach().cpu().numpy()
        softmin_ = softmin.item()
        g_ = gamma.item()
        G = torch.Tensor(compute_dwsa_loss_backward(C_, D_, g_)).to(dev).type(dtype)
        return grad_output.view(-1, 1).expand_as(G) * G, None

class DWSA_Loss(torch.nn.Module):
    def __init__(self, beta=0.001, threshold=2, softmax='row', keep_times=False):
        super(DWSA_Loss, self).__init__()

        self.beta = beta
        self.func_apply = _DWSALoss.apply
        self.threshold = threshold
        self.softmax = softmax
        self.keep_times = keep_times


    def forward(self, centers_a, centers_b):
        M, N = centers_a.shape[0], centers_b.shape[0]

        sorted_centers_a = centers_a[centers_a[:, -1].argsort()]
        sorted_centers_b = centers_b[centers_b[:, -1].argsort()]

        if not self.keep_times:
            sorted_centers_a = sorted_centers_a[:, :-1]
            sorted_centers_b = sorted_centers_b[:, :-1]

        sorted_centers_a = sorted_centers_a / torch.sqrt(torch.sum(sorted_centers_a**2, axis=-1, keepdims=True) + 1e-10) 
        sorted_centers_b = sorted_centers_b / torch.sqrt(torch.sum(sorted_centers_b**2, axis=-1, keepdims=True) + 1e-10)
        
        cost = 1 - torch.matmul(sorted_centers_a, sorted_centers_b.t())

        threshold = self.threshold
        cost = torch.cat((cost, torch.ones_like(cost)*threshold))
        cost = cost.t().reshape(2*N, M).t()
        cost = torch.cat((threshold*torch.ones(M, 1, dtype=cost.dtype).to(cost.device), cost), dim=1)

        if self.softmax == 'row':
            cost = F.softmax(cost, -1)
        elif self.softmax == 'col':
            cost = F.softmax(cost, 0)
        elif self.softmax == 'all':
            cost = F.softmax(cost.view(-1), 0).view(M, -1)

        loss = self.func_apply(cost, self.beta)

        return loss 
