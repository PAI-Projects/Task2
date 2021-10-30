import torch
import torch.optim
from torch import nn
from torch.nn import functional as F

from distributions import MultivariateDiagonalGaussian, UnivariateGaussian
from util import ParameterDistribution


class BayesianLayer(nn.Module):
    """
    Module implementing a single Bayesian feedforward layer.
    It maintains a prior and variational posterior for the weights (and biases)
    and uses sampling to approximate the gradients via Bayes by backprop.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Create a BayesianLayer.

        :param in_features: Number of input features
        :param out_features: Number of output features
        :param bias: If true, use a bias term (i.e., affine instead of linear transformation)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # TODO: Create a suitable prior for weights and biases as an instance of ParameterDistribution.
        #  You can use the same prior for both weights and biases, but are free to experiment with different priors.
        #  You can create constants using torch.tensor(...).
        #  Do NOT use torch.Parameter(...) here since the prior should not be optimized!
        #  Example: self.prior = MyPrior(torch.tensor(0.0), torch.tensor(1.0))
        # TODO add mixture used in paper
        nr_mixture_components = 2
        # refers to pi in paper; pi in {0.25, 0.5, 0.75}
        self.mixture_weight = 0.5
        # according to paper: sigma1 in e^-{0, 1, 2}, sigma2 in e^-{6,7,8}
        prior_sigma = [torch.exp(torch.tensor(-1)), torch.exp(torch.tensor(-7))]
        self.prior = UnivariateGaussian(torch.tensor(0.0), prior_sigma[0])
        assert isinstance(self.prior, ParameterDistribution)
        assert not any(True for _ in self.prior.parameters()), 'Prior cannot have parameters'

        # TODO: Create a suitable variational posterior for weights as an instance of ParameterDistribution.
        #  You need to create separate ParameterDistribution instances for weights and biases,
        #  but can use the same family of distributions if you want.
        #  IMPORTANT: You need to create a nn.Parameter(...) for each parameter
        #  and add those parameters as an attribute in the ParameterDistribution instances.
        #  If you forget to do so, PyTorch will not be able to optimize your variational posterior.
        #  Example: self.weights_var_posterior = MyPosterior(
        #      torch.nn.Parameter(torch.zeros((out_features, in_features))),
        #      torch.nn.Parameter(torch.ones((out_features, in_features)))
        #  )
        self.weights_var_posterior = MultivariateDiagonalGaussian(  # TODO in the paper they use two gaussians per weight
            torch.nn.Parameter(torch.zeros(out_features, in_features)),
            torch.nn.Parameter(torch.ones(out_features, in_features) * torch.tensor(-3))
        )

        assert isinstance(self.weights_var_posterior, ParameterDistribution)
        assert any(True for _ in self.weights_var_posterior.parameters()), 'Weight posterior must have parameters'

        if self.use_bias:
            # TODO: As for the weights, create the bias variational posterior instance here.
            #  Make sure to follow the same rules as for the weight variational posterior.
            self.bias_var_posterior = MultivariateDiagonalGaussian(
                torch.nn.Parameter(torch.zeros(out_features)),  # TODO in the paper they use two gaussians per weight
                torch.nn.Parameter(torch.ones(out_features) * torch.tensor(-3))
            )
            assert isinstance(self.bias_var_posterior, ParameterDistribution)
            assert any(True for _ in self.bias_var_posterior.parameters()), 'Bias posterior must have parameters'
        else:
            self.bias_var_posterior = None

    def forward(self, inputs: torch.Tensor):
        """
        Perform one forward pass through this layer.
        If you need to sample weights from the variational posterior, you can do it here during the forward pass.
        Just make sure that you use the same weights to approximate all quantities
        present in a single Bayes by backprop sampling step.

        :param inputs: Flattened input images as a (batch_size, in_features) float tensor
        :return: 3-tuple containing
            i) transformed features using stochastic weights from the variational posterior,
            ii) sample of the log-prior probability, and
            iii) sample of the log-variational-posterior probability
        """
        # TODO: Perform a forward pass as described in this method's docstring.
        #  Make sure to check whether `self.use_bias` is True,
        #  and if yes, include the bias as well.
        weights = self.weights_var_posterior.sample()

        bias = self.bias_var_posterior.sample() if self.use_bias else None

        # TODO: is this correct?
        log_prior = self.prior.log_likelihood(weights)
        log_variational_posterior = self.weights_var_posterior.log_likelihood(weights)
        if self.use_bias:
            log_prior += self.prior.log_likelihood(bias)  # TODO: use self.mixture_weight
            log_variational_posterior += self.bias_var_posterior.log_likelihood(bias)

        return F.linear(inputs, weights, bias), log_prior, log_variational_posterior
