import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy import stats
from torch import nn
from torch.distributions import Normal

from util import ParameterDistribution


class MixtureDistribution(ParameterDistribution):
    """
    Mixture distribution of multiple ParameterDistribution's.
    """

    def __init__(self, mixtures: nn.ModuleList, mixture_weights: torch.Tensor, sample_shape):
        super(MixtureDistribution, self).__init__()  # always make sure to include the super-class init call!
        assert len(mixtures) == mixture_weights.shape[0]
        self.mixtures = mixtures
        self.mixture_weights = mixture_weights
        self.sample_shape = sample_shape

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(self.sample_shape)
        for i, dist in enumerate(self.mixtures):
            x += dist.log_likelihood(values) * self.mixture_weights[i]

        return x

    def sample(self) -> torch.Tensor:
        x = torch.zeros(self.sample_shape)
        for i, dist in enumerate(self.mixtures):
            x += dist.sample() * self.mixture_weights[i]

        return x


class UnivariateGaussian(ParameterDistribution):
    """
    Univariate Gaussian distribution.
    For multivariate data, this assumes all elements to be i.i.d.
    """

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        super(UnivariateGaussian, self).__init__()  # always make sure to include the super-class init call!
        assert mu.size() == () and sigma.size() == ()
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        """
        log likelihood: log L(mu, sigma; x_1, ..., x_n) = log prod_n p(x | mu, sigma)
        """
        # p = (1 / (self.sigma * math.sqrt(2 * np.pi))) * torch.exp(-(values - self.mu) ** 2 / (2 * self.sigma**2))
        m = Normal(self.mu, self.sigma)
        log_p = m.log_prob(values)

        return log_p.sum()

    def sample(self) -> torch.Tensor:
        z = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(1.0))
        x = self.mu + self.sigma * z

        return x


class MultivariateDiagonalGaussian(ParameterDistribution):
    """
    Multivariate diagonal Gaussian distribution,
    i.e., assumes all elements to be independent Gaussians
    but with different means and standard deviations.
    This parameterizes the standard deviation via a parameter rho as
    sigma = softplus(rho).
    """

    def __init__(self, mu: torch.Tensor, rho: torch.Tensor):
        super(MultivariateDiagonalGaussian, self).__init__()  # always make sure to include the super-class init call!
        assert mu.size() == rho.size()
        self.mu = mu
        self.rho = rho

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        variances = torch.pow(F.softplus(self.rho), 2)
        m = Normal(self.mu.view(-1), variances.view(-1))

        log_p = m.log_prob(values.view(-1))

        return log_p.sum()

    def sample(self) -> torch.Tensor:
        # since we have a diagonal covariance matrix we can draw from n unvariate gaussians
        Z = torch.empty(self.mu.size()).normal_(0, 1)

        # re-parameterization
        x = self.mu + F.softplus(self.rho) * Z

        return x


if __name__ == '__main__':
    mu = 10.0
    sigma = 3.0

    normal_dist = UnivariateGaussian(torch.tensor(mu), torch.tensor(sigma))

    data_scipy = np.random.randn(100000) * sigma + mu

    # compute log likelihood
    log_l = normal_dist.log_likelihood(torch.from_numpy(data_scipy))
    log_l_scipy = np.log(stats.norm.pdf(data_scipy, mu, sigma)).sum()

    print("log likelihood ours", log_l)
    print("log likelihood scipy", log_l_scipy)

    # generating samples
    x = np.asarray([normal_dist.sample().numpy() for i in range(100000)])

    # plotting samples
    plt.hist(data_scipy, bins=100)
    plt.title("scipy samples")
    plt.show()

    plt.hist(x, bins=100)
    plt.title("our samples")
    plt.show()

    # multivariate tests
    mu = [10.0, 5.0]
    sigma = [1.0, 3.0]

    multivariate_diag_normal = MultivariateDiagonalGaussian(torch.tensor(mu), torch.tensor(sigma))

    samples = np.asarray([multivariate_diag_normal.sample().numpy() for i in range(100000)])

    # log likelihood
    log_l_mult = multivariate_diag_normal.log_likelihood(torch.from_numpy(samples))
    log_l_mult_scipy = np.log(stats.norm.pdf(samples, mu, sigma)).sum()

    print("log likelihood multi variate", log_l_mult)
    print("log likelihood scipy", log_l_mult_scipy)

    # plotting samples
    plt.hist(samples[:, 0], bins=100)
    plt.title("multivariate first dim")
    plt.show()

    plt.hist(samples[:, 1], bins=100)
    plt.title("multivariate 2nd dim")
    plt.show()
