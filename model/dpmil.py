import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import numpy as np
from torch import nn
from torch.distributions import Beta, Categorical, Uniform, MultivariateNormal, Dirichlet
from scipy.stats import beta
import torch.nn.functional as F
import matplotlib.pyplot as plt


from abc import ABCMeta, abstractmethod


class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass


class ReparametrizedGaussian(Distribution):
    """
    Diagonal ReparametrizedGaussian distribution with parameters mu (mean) and rho. The standard
    deviation is parametrized as sigma = log(1 + exp(rho))
    A sample from the distribution can be obtained by sampling from a unit Gaussian,
    shifting the samples by the mean and scaling by the standard deviation:
    w = mu + log(1 + exp(rho)) * epsilon
    """

    def __init__(self, mu, rho):
        self.mean = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)
        self.point_estimate = self.mean

    @property
    def std_dev(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self, n_samples=1):
        epsilon = torch.distributions.Normal(0, 1).sample(sample_shape=(n_samples, *self.mean.size()))
        epsilon = epsilon.to(self.mean.device)
        return self.mean + self.std_dev * epsilon

    def log_prob(self, target):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.std_dev)
                - ((target - self.mean) ** 2) / (2 * self.std_dev ** 2)).sum(dim=-1)

    def entropy(self):
        """
        Computes the entropy of the Diagonal Gaussian distribution.
        Details on the computation can be found in the 'diagonal_gaussian_entropy' notes in the repo
        """
        if self.mean.dim() > 1:
            # n_inputs, n_outputs = self.mean.shape
            dim = 1
            for d in self.mean.shape:
                dim *= d
        elif self.mean.dim() == 0:
            dim = 1
        else:
            dim = len(self.mean)
            # n_outputs = 1

        part1 = dim / 2 * (math.log(2 * math.pi) + 1)
        part2 = torch.sum(torch.log(self.std_dev))

        return (part1 + part2).unsqueeze(0)

# class DP_Cluster(nn.Module):
#
#     def __init__(self, concentration, trunc, eta, batch_size, epoch, dim=1024, n_sample=100):
#         super().__init__()
#         self.alpha = concentration
#         self.T = trunc
#         self.dim = dim
#         self.batch_size = batch_size
#         self.epoch = epoch
#
#         self.mu = nn.ParameterList([nn.Parameter(torch.zeros([self.dim])) for t in range(self.T)])
#         # self.sig = nn.Parameter(torch.stack([torch.eye(self.dim) for _ in range(self.T)]))
#         self.rho = nn.ParameterList([nn.Parameter(torch.zeros([self.dim])) for t in range(self.T)])
#         self.gaussians = [ReparametrizedGaussian(self.mu[t], self.rho[t]) for t in range(self.T)]
#         self.phi = torch.ones([self.T, self.dim]) / self.T
#
#         self.eta = eta
#         self.gamma_1 = torch.ones(self.T)
#         self.gamma_2 = torch.ones(self.T) * eta
#         self.dp_process = DirichletProcess(self.alpha,self.T,self.eta,self.batch_size,self.dim)
#
#     def forward(self,x):
#         optimizer = torch.optim.Adam(self.dp_process.parameters(), lr=1e-1 )
#         # train_losses = []
#         for i in range(self.epoch):
#             self.dp_process.train()
#             train_loss = -self.dp_process.likelihood(x)
#             # train_losses.append(train_loss.cpu().detach().numpy())
#             optimizer.zero_grad()
#             train_loss.backward()
#             optimizer.step()
#         # train_losses = np.array(train_losses)
#         # plt.plot(np.arange(epoch), train_losses)
#         # plt.savefig('train_loss.jpg')
#
#         return self.dp_process.infer(x)

class HDP_binary_classifier(nn.Module): #only eta
    def __init__(self, concentration, trunc, eta, batch_size, MC_num, dim=1024, n_sample=100):
        super().__init__()
        self.D_P_cluster = DirichletProcess(concentration,trunc,eta,batch_size,dim,n_sample)
        self.trunc = trunc
        self.MC_num = MC_num
        self.D_P_clssfy = DirichletProcess(concentration,2,eta,batch_size,dim,n_sample)

    def forward(self,x):
        neg_likelyhood = -self.D_P_cluster.likelihood(x)
        betas = self.D_P_cluster.sample_beta(self.MC_num)
        weights = self.D_P_cluster.mix_weights(betas)[:, :-1]
        normalize_factor = torch.transpose(torch.vstack((torch.sum(weights, axis=1), torch.sum(weights, axis=1))), 0, 1)
        weights = weights / normalize_factor
        MC_samples = []
        for j in range(self.MC_num):
            prob = weights[j,:].cpu().numpy()
            draw = np.random.choice(self.trunc,size=1,p=prob)[0]
            s = self.D_P_cluster.gaussians[draw].sample(1)[0,:]
            MC_samples.append(s)
        MC_samples = torch.stack(MC_samples)
        logits = self.D_P_clssfy.infer(MC_samples)

        return logits, neg_likelyhood

class DirichletProcess(nn.Module):
    def __init__(self, concentration, trunc, eta, batch_size, dim=1024, n_sample=100):
        super().__init__()
        self.alpha = concentration
        self.T = trunc
        self.dim = dim
        self.batch_size = batch_size

        self.mu = nn.ParameterList([nn.Parameter(torch.zeros([self.dim])) for t in range(self.T)])
        # self.sig = nn.Parameter(torch.stack([torch.eye(self.dim) for _ in range(self.T)]))
        self.rho = nn.ParameterList([nn.Parameter(torch.zeros([self.dim])) for t in range(self.T)])
        self.gaussians = [ReparametrizedGaussian(self.mu[t], self.rho[t]) for t in range(self.T)]
        self.phi = torch.ones([self.T, self.dim]) / self.T

        self.eta = eta
        self.gamma_1 = torch.ones(self.T)
        self.gamma_2 = torch.ones(self.T) * eta

    def mix_weights(self, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

    def entropy(self):
        entropy = [self.gaussians[t].entropy() for t in range(self.T)]
        entropy = torch.stack(entropy, dim=0).mean()

        return entropy

    def get_log_prob(self, x):
        pdfs = [self.gaussians[t].log_prob(x) for t in range(self.T)]
        pdfs = torch.stack(pdfs, dim=-1)
        return pdfs

    def sample_beta(self, size):
        a = self.gamma_1.detach().cpu().numpy()
        b = self.gamma_2.detach().cpu().numpy()

        samples = beta.rvs(a, b, size=(size, self.T))
        samples = torch.from_numpy(samples).cuda()

        return samples

    def likelihood(self, x): # calculate ELBO
        # x shape is [num_sample, dim]
        beta = self.sample_beta(x.shape[0])
        pi = self.mix_weights(beta)[:, :-1]
        entropy = self.entropy()
        log_pdfs = self.get_log_prob(x)

        self.phi = torch.softmax(torch.log(pi) + entropy + log_pdfs, dim=-1)

        likelihood = self.phi * (entropy + log_pdfs)
        likelihood = likelihood.sum(1).mean(0)

        return likelihood

    def infer(self, x):
        # x shape is [num_sample, dim]
        """
        Get logit

        return: Logits with length T
        """
        beta = self.sample_beta(x.shape[0])
        pi = self.mix_weights(beta)[:, :-1]
        log_pdfs = self.get_log_prob(x)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logits = torch.tensor(pi).to(device) * torch.softmax(log_pdfs.to(device), dim=1)

        return logits

    def update_gamma(self):

        phi = self.phi

        phi_flipped = torch.flip(phi, dims=[1])
        cum_sum = torch.cumsum(phi_flipped, dim=1) - phi_flipped
        cum_sum = torch.flip(cum_sum, dims=[1])

        self.gamma_1 = 1 + phi.sum(0)
        self.gamma_2 = self.eta + cum_sum.sum(0)

    def update_mean(self, phi):
        N = phi.sum(0)
        pass

    def update_variance(self):
        pass


class BClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BClassifier, self).__init__()
        self.L = input_size
        self.D = input_size
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.D, num_classes)
        )
        
    def forward(self, x):
        H = x
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxL
        Y_prob = self.classifier(M)
        return Y_prob


class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
