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


class ReparametrizedGaussian_VI(Distribution):
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

    def set_parameters(self, mu, rho):
        self.mean = mu
        self.rho = rho


class DirichletProcess_VI(nn.Module):
    def __init__(self, trunc, eta, batch_size, dim=1024, n_sample=100):
        super().__init__()
        self.T = trunc
        self.dim = dim
        self.batch_size = batch_size
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.mu = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.dim).uniform_(-0.5, 0.5)) for t in range(self.T)])
        # self.sig = nn.Parameter(torch.stack([torch.eye(self.dim) for _ in range(self.T)]))
        self.rho = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.dim).uniform_(-0.5, 0.5)) for t in range(self.T)])
        self.gaussians = [ReparametrizedGaussian_VI(self.mu[t], self.rho[t]) for t in range(self.T)]
        self.phi = torch.ones([self.dim, self.T]) / self.T

        self.eta = eta
        self.gamma_1 = torch.ones(self.T)
        self.gamma_2 = torch.ones(self.T) * eta

    def mix_weights(self, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        pi = F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
        return pi

    def entropy(self):
        entropy = [self.gaussians[t].entropy() for t in range(self.T)]
        entropy = torch.stack(entropy, dim=-1)

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

    def forward(self, x):
        batch_size = x.shape[0]

        beta = self.sample_beta(batch_size)
        pi = self.mix_weights(beta)[:, :-1]
        log_pdfs = self.get_log_prob(x)
        entropy = self.entropy()
        entropy = entropy.expand(batch_size, -1)

        phi_new, kl_gaussian = self.get_phi(torch.log(pi), entropy, log_pdfs)

        self.update_gamma()

        likelihood = phi_new * kl_gaussian
        likelihood = likelihood.sum(1).mean(0)

        self.phi = phi_new.data

        return likelihood

    def infer(self, x):
        """
        Get logit

        return: Logits with length T
        """

        beta = self.sample_beta(x.shape[0])
        pi = self.mix_weights(beta)[:, :-1]
        log_pdfs = self.get_log_prob(x)
        logits = torch.log(pi) + log_pdfs
        assert not torch.isnan(logits).any()
        # logits = F.normalize(logits, dim=2)

        return logits

    def get_phi(self, log_pi, entropy, log_pdf):
        # maybe mention this in the paper we do this to improve numerical stability
        kl_gaussian = log_pdf + entropy
        kl_pi = log_pi

        N_t_gaussian = kl_gaussian.sum(0, keepdim=True)
        N_t_pi = kl_pi.sum(0, keepdim=True)
        mix = (N_t_pi / (N_t_gaussian + N_t_pi))

        kl = mix * kl_gaussian + (1 - mix) * kl_pi

        return kl.softmax(dim=1), mix * kl_gaussian

    def update_gamma(self):
        phi = self.phi

        phi_flipped = torch.flip(phi, dims=[1])
        cum_sum = torch.cumsum(phi_flipped, dim=1) - phi_flipped
        cum_sum = torch.flip(cum_sum, dims=[1])

        self.gamma_1 = 1 + phi.mean(0)
        self.gamma_2 = self.eta + cum_sum.mean(0)

    def standardize(self, x):
        x = (x - x.mean(1)) / x.std(1)
        return x

    def update_mean(self, phi):
        N = phi.sum(0)
        pass

    def update_variance(self):
        pass


class DP_Classifier(nn.Module):

    def __init__(self,  trunc, eta, batch_size, dim=1024, n_sample=100):
        super().__init__()
        self.dp_process = DirichletProcess_VI( trunc, eta, batch_size, dim)

    def forward(self, x):
        return self.dp_process.infer(x)


class ReparametrizedGaussian_EM(Distribution):
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
        return torch.exp(self.rho)

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

    def set_parameters(self, mu, rho):
        self.mean = mu
        self.rho = rho

class DirichletProcess_EM(nn.Module):
    def __init__(self, trunc, eta, batch_size, dim=1024, n_sample=100):
        super().__init__()
        self.T = trunc
        self.dim = dim
        self.batch_size = batch_size
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.mu = torch.FloatTensor(self.T, self.dim).uniform_(-0.5, 0.5).to(self.device)
        self.rho = torch.FloatTensor(self.T, self.dim).uniform_(-4, -3).to(self.device)
        # self.mu_encode = nn.Sequential(
        #     nn.Linear(dim, dim * 2),
        #     # nn.Tanh(),
        #     # nn.Linear(dim * 2, dim * 2),
        #     nn.Tanh(),
        #     nn.Linear(dim * 2, dim * trunc)
        # )
        # self.rho_encode = nn.Sequential(
        #     nn.Linear(dim, dim * 2),
        #     # nn.Tanh(),
        #     # nn.Linear(dim * 2, dim * 2),
        #     nn.Tanh(),
        #     nn.Linear(dim * 2, dim * trunc)
        # )
        self.gaussians = [ReparametrizedGaussian_EM(self.mu[t], self.rho[t]) for t in range(self.T)]
        self.phi = (torch.ones([self.batch_size, self.T]) / self.T).to(self.device)

        self.eta = eta
        self.gamma_1 = torch.ones(self.T).cuda()
        self.gamma_2 = (torch.ones(self.T) * eta).cuda()

    def sample_beta(self, size):
        a = self.gamma_1.detach().cpu().numpy()
        b = self.gamma_2.detach().cpu().numpy()

        samples = beta.rvs(a, b, size=(size, self.T))
        samples = torch.from_numpy(samples).cuda()

        return samples

    def forward(self, x):
        batch_size = x.shape[-2]

        beta = self.sample_beta(batch_size)
        pi = self.mix_weights(beta)[:, :-1]
        log_pdfs = self.get_log_prob(x)
        entropy = self.entropy()
        entropy = entropy.expand(batch_size, -1)

        phi_new, kl_gaussian = self.get_phi(torch.log(pi), entropy, log_pdfs)
        # log_phi = torch.log(pi) + entropy + log_pdfs
        # phi_new = torch.softmax(log_phi, dim=1)

        self.update_gaussians(x.data, phi_new.data)
        self.update_gamma()

        likelihood = phi_new * kl_gaussian
        likelihood = likelihood.sum(1).mean(0)

        self.phi = phi_new.data

        return - likelihood

    def inference(self, x, sm=True):
        """
        Get logit

        return: Logits with length T
        """

        beta = self.sample_beta(x.shape[-2])
        pi = self.mix_weights(beta)[:, :-1]

        log_pdfs = self.get_log_prob(x)
        logits = torch.log(pi) + log_pdfs

        # N_t_gaussian = log_pdfs.min()
        # N_t_pi = torch.log(pi).min()
        # mix = (N_t_pi / (N_t_gaussian + N_t_pi))
        # logits = mix * log_pdfs + (1-mix) * torch.log(pi)

        # logits = F.normalize(logits, dim=2)
        if sm:
            logits = F.softmax(logits, dim=-1)

        return logits

    def get_phi(self, log_pi, entropy, log_pdf):
        # TODO: maybe mention this in the paper we do this to improve numerical stability
        kl_gaussian = log_pdf + entropy
        kl_pi = log_pi

        N_t_gaussian = kl_gaussian.min()
        N_t_pi = kl_pi.min()
        mix = (N_t_pi / (N_t_gaussian + N_t_pi))

        kl = mix * kl_gaussian + (1-mix) * kl_pi

        return kl.softmax(dim=1), mix * kl_gaussian
        # return kl.softmax(dim=1),  kl_gaussian

    def update_gamma(self):

        phi = self.phi

        phi_flipped = torch.flip(phi, dims=[1])
        cum_sum = torch.cumsum(phi_flipped, dim=1) - phi_flipped
        cum_sum = torch.flip(cum_sum, dims=[1])

        self.gamma_1 = 1 + phi.reshape(-1, self.T).mean(0)
        self.gamma_2 = self.eta + cum_sum.reshape(-1, self.T).mean(0)

        # self.gamma_1 = self.gamma_1 + phi.reshape(-1, self.T).mean(0)
        # self.gamma_2 = self.gamma_2 + cum_sum.reshape(-1, self.T).mean(0)

    def mix_weights(self, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        pi = F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
        return pi

    def entropy(self):
        entropy = [self.gaussians[t].entropy() for t in range(self.T)]
        entropy = torch.stack(entropy, dim=-1)

        return entropy

    def get_log_prob(self, x):
        pdfs = [self.gaussians[t].log_prob(x) for t in range(self.T)]
        pdfs = torch.stack(pdfs, dim=-1)
        return pdfs

    def update_mean(self, x, phi_new):
        phi = self.phi
        mu = self.mu

        N_t = phi.sum(0, keepdim=True).clamp(1e-6).T
        N_t_new = phi_new.sum(0, keepdim=True).clamp(1e-6).T

        mu_new = torch.einsum("ij,ik->jk", phi_new, x) / N_t_new
        mix = (N_t / (N_t_new + N_t)).expand(-1, mu.shape[1])
        self.mu = mix * mu + (1 - mix) * mu_new

    def update_variance(self, x, phi_new):
        phi = self.phi
        sig = torch.exp(self.rho)

        # sig_new = torch.einsum("ij,ik->jk", (x - x.mean(0)), (x - x.mean(0))) / x.shape[0]
        # sig_new = torch.einsum("ji,ki->ik", phi_new, sig_new.double())
        # sig_new = torch.diagonal(sig_new).unsqueeze(0).expand(x.shape[0], -1)

        N_t = phi.sum(0, keepdim=True).clamp(1e-6).T
        N_t_new = phi_new.sum(0, keepdim=True).clamp(1e-6).T

        # sig_new = torch.einsum("ij,ik->jk", phi_new.float(), (x - x.mean(0)) ** 2) / N_t_new
        sig_new = (phi_new.unsqueeze(-1) * (x.unsqueeze(1) - self.mu) ** 2).sum(0) / N_t_new
        sig_new = torch.sqrt(sig_new)

        mix = N_t / (N_t_new + N_t)

        updated_sig = mix * sig + (1 - mix) * sig_new
        self.rho = torch.log(updated_sig)

    def update_gaussians(self, x, phi_new):
        self.update_mean(x, phi_new)
        self.update_variance(x, phi_new)
        [self.gaussians[t].set_parameters(self.mu[t], self.rho[t]) for t in range(self.T)]


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

class DP_Cluster_EM(nn.Module):
    def __init__(self,  trunc, eta, batch_size, epoch, dim=256):
        super().__init__()
        self.epoch = epoch
        self.dp = DirichletProcess_EM(trunc=trunc,eta=eta,batch_size=batch_size,dim=dim)

    def forward(self,x):
        for i in range(self.epoch):
            self.dp(x)
        return self.dp.inference(x)

class DP_Cluster_VI(nn.Module):

    def __init__(self, trunc, eta, batch_size, epoch, dim=1024, n_sample=100):
        super().__init__()

        self.epoch = epoch
        self.dp_process = DirichletProcess_VI(trunc=trunc,eta=eta,batch_size=batch_size,dim=dim)

    def forward(self,x):
        optimizer = torch.optim.Adam(self.dp_process.parameters(), lr=1e-2 )
        # train_losses = []
        for i in range(self.epoch):
            self.dp_process.train()
            train_loss = -self.dp_process(x)
            # train_losses.append(train_loss.cpu().detach().numpy())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        # train_losses = np.array(train_losses)
        # plt.plot(np.arange(epoch), train_losses)
        # plt.savefig('train_loss.jpg')

        return self.dp_process.infer(x)


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

class HierarchicalDirichletProcess_EM(nn.Module):
    def __init__(self, n_dps, trunc, eta, batch_size, dim=256):
        super().__init__()
        self.T = n_dps
        self.eta = eta
        self.dps = [
            DirichletProcess_EM(
                eta=eta,
                trunc=trunc,
                batch_size=batch_size,
                dim=dim
            )
            for _ in range(n_dps)
        ]
        self.dim = dim

        self.gamma_1 = torch.ones(n_dps).cuda()
        self.gamma_2 = (torch.ones(n_dps) * eta).cuda()

    def forward(self, x):

        batch_size = x.shape[-2]
        beta = self.sample_beta(batch_size)
        pi = self.mix_weights(beta)[:, :-1]

        likelihoods = [dp(x) for dp in self.dps]
        likelihoods = torch.stack(likelihoods)
        loss = - (pi * likelihoods).mean(0).sum()

        self.phi = self.get_phi(pi)
        self.update_gamma()

        return loss

    def get_phi(self, pi):

        log_phi = [torch.log(dp.phi).sum(1) for dp in self.dps]
        log_phi = torch.stack(log_phi, dim=-1)
        phi = (torch.log(pi) + log_phi).softmax(-1)

        return phi

    def inference(self, x):

        batch_size = x.shape[-2]
        beta = self.sample_beta(batch_size)
        pi = self.mix_weights(beta)[:, :-1]

        preds = [dp.inference(x, sm=False).sum(-1) for dp in self.dps]
        preds = torch.stack(preds, dim=-1)
        preds = (torch.log(pi) + preds).softmax(-1)

        return preds

    def update_gamma(self):

        phi = self.phi

        phi_flipped = torch.flip(phi, dims=[1])
        cum_sum = torch.cumsum(phi_flipped, dim=1) - phi_flipped
        cum_sum = torch.flip(cum_sum, dims=[1])

        self.gamma_1 = 1 + phi.reshape(-1, self.T).mean(0)
        self.gamma_2 = self.eta + cum_sum.reshape(-1, self.T).mean(0)

    def sample_beta(self, size):
        a = self.gamma_1.detach().cpu().numpy()
        b = self.gamma_2.detach().cpu().numpy()

        samples = beta.rvs(a, b, size=(size, self.T))
        samples = torch.from_numpy(samples).cuda()

        return samples

    def mix_weights(self, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        pi = F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
        return pi

class HDP_Cluster_EM(nn.Module):
    def __init__(self, n_dps, trunc, eta, batch_size, epoch, dim=256):
        super().__init__()
        self.epoch = epoch
        self.hdp = HierarchicalDirichletProcess_EM(n_dps,trunc,eta,batch_size,dim)

    def forward(self,x):
        for i in range(self.epoch):
            self.hdp(x)
        return self.hdp.inference(x)