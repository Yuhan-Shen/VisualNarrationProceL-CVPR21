import numpy as np
#import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F


#np.random.seed(123)
#torch.manual_seed(0)

class SOPL_Layer(nn.Module):
    def __init__(self, K, D, n_iter=3, beta=0.001, init='kmeans++'):
        super(SOPL_Layer, self).__init__()
        """
          K: the number of clusters
          D: feature dimension
          n_iter: the number of iterations in SOPL
          beta: smooth parameter
          init: how to initialize the prototypes (centers) SOPL (kmeans++ or random) 
        """
        self.K = K
        self.D = D
        self.n_iter = n_iter
        self.beta = beta
        self.init = init

    def forward(self, x):
        if self.init == 'kmeans++':
            ### Use the _k_init in sklearn to initialize centers (prototypes)
            from sklearn.cluster.k_means_ import _k_init # kmeans++ initialization
            data = x.detach().cpu().numpy()
            squared_norm = (data ** 2).sum(axis=1)
            init_centers = _k_init(data, self.K, squared_norm, np.random.RandomState(1))
            centers = torch.from_numpy(init_centers).to(x.device).type(x.dtype)
        else:
            centers = 0.1 * torch.randn([self.K, x.shape[-1]]).to(x.device).type(x.dtype)


        x_ = torch.unsqueeze(x, 1).repeat(1, self.K, 1)
        for i in range(self.n_iter):
            ### compute pairwise distance between data and centers
            dist = torch.sum((x_ - centers)**2, dim=-1)
            ### apply softmax to get distribution weights
            dist = F.softmax(-dist / self.beta, dim=-1)
            ### update centers
            centers = torch.mm(dist.t(), x) / (torch.sum(dist.t(), dim=-1, keepdim=True) + 1e-10)

        return centers


class MultiModal_SOPL(nn.Module):
    def __init__(self, K_a, K_b, D, n_iter=3, beta=0.001, center_init='kmeans++'):
        super(MultiModal_SOPL, self).__init__()
        self.K_a = K_a
        self.K_b = K_b
        self.D = D
        self.n_iter = n_iter
        self.fc_1 = nn.Linear(D, D).double()
        self.fc_2 = nn.Linear(D, D).double()

        ### use identity matrix and zero vector to initialize weights and bias, 
        ### to keep the data the same as original at the beginning of training
        self.fc_1.weight.data.copy_(torch.eye(D))
        self.fc_2.weight.data.copy_(torch.eye(D))
        self.fc_1.bias.data.copy_(torch.zeros([D]))
        self.fc_2.bias.data.copy_(torch.zeros([D]))


        self.sopl_1 = SOPL_Layer(K_a, D+1, n_iter, beta, center_init)
        self.sopl_2 = SOPL_Layer(K_b, D+1, n_iter, beta, center_init)

    def forward(self, x, y, x_times=None, y_times=None):
        """
          Args:
              x: input features for modality a
              y: input features for modality b
              x_times: input timestamps for modality a 
              y_times: input timestamps for modality b
          Returns:
              x_centers: output prototypes (feature prototype concatenated with time prototype, for modality a
              y_centers: output prototypes (feature prototype concatenated with time prototype, for modality b
              x: learned features for modality a
              y: learned features for modality b
        """

        x = self.fc_1(x)
        x = F.normalize(x, dim=-1)
        x = torch.cat((x, x_times), -1)
        x_centers = self.sopl_1(x)

        y = self.fc_2(y)
        y = F.normalize(y, dim=-1)
        y = torch.cat((y, y_times), -1)
        y_centers = self.sopl_2(y)

        return x_centers, y_centers, x, y

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)


class MultiModal_Features(nn.Module):
    def __init__(self, D):
        super(MultiModal_Features, self).__init__()
        self.D = D
        self.fc_1 = nn.Linear(D, D).double()
        self.fc_2 = nn.Linear(D, D).double()
        self.fc_1.weight.data.copy_(torch.eye(D))
        self.fc_2.weight.data.copy_(torch.eye(D))
        self.fc_1.bias.data.copy_(torch.zeros([D]))
        self.fc_2.bias.data.copy_(torch.zeros([D]))

    def forward(self, x, y, x_times=None, y_times=None):
        """
          This function outputs the learned features for two modalities 
          (does not include SOPL modules to obtain centers)
          Args:
              x: input features for modality a
              y: input features for modality b
              x_times: input timestamps for modality a 
              y_times: input timestamps for modality b
          Returns:
              x: learned features for modality a
              y: learned features for modality b
        """

        x = self.fc_1(x)
        x = F.normalize(x, dim=-1)
        x = torch.cat((x, x_times), -1)

        y = self.fc_2(y)
        y = F.normalize(y, dim=-1)
        y = torch.cat((y, y_times), -1)

        return x, y

    def extract_visual_features(self, x, x_times):
        x = self.fc_1(x)
        x = F.normalize(x, dim=-1)
        x = torch.cat((x, x_times), -1)

        return x
    
    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))
