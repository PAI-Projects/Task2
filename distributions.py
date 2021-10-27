import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import stats
from torch.distributions import Normal

from util import ParameterDistribution


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
        p = torch.exp(m.log_prob(values))

        return torch.log(p).sum()

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
        # TODO: Implement this
        return 0.0

    def sample(self) -> torch.Tensor:
        # TODO: Implement this
        raise NotImplementedError()


if __name__ == '__main__':
    mu = 10.0
    sigma = 3.0

    normal_dist = UnivariateGaussian(torch.tensor(mu), torch.tensor(sigma))

    data_scipy = np.random.randn(100000) * sigma + mu

    # compute log likelihood
    log_l = normal_dist.log_likelihood(torch.from_numpy(data_scipy))
    log_l_scipy = np.log(stats.norm.pdf(data_scipy, mu, sigma)).sum()

    print("likelihood ours", log_l)
    print("likelihood scipy", log_l_scipy)

    # generating samples
    x = np.asarray([normal_dist.sample().numpy() for i in range(100000)])

    print("plotting samples")
    plt.hist(data_scipy, bins=100)
    plt.title("scipy samples")
    plt.show()

    plt.hist(x, bins=100)
    plt.title("our samples")
    plt.show()
