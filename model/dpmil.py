import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x):
        beta = self.sample_beta(x.shape[1])
        pi = self.mix_weights(beta)[:, :-1]
        entropy = self.entropy()
        log_pdfs = self.get_log_prob(x)

        self.phi = torch.softmax(torch.log(pi) + entropy + log_pdfs, dim=-1)

        likelihood = self.phi * (entropy + log_pdfs)
        likelihood = likelihood.sum(1).mean(0)

        return likelihood

    def inference(self, x):
        """
        Get logit

        return: Logits with length T
        """
        beta = self.sample_beta(x.shape[1])
        pi = self.mix_weights(beta)[:, :-1]
        log_pdfs = self.get_log_prob(x)
        logits = pi * torch.softmax(log_pdfs, dim=2)

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
