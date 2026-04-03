import torch

from scvi_neural_ode.distributions import NB


def test_NB():
    r, p = torch.tensor([6.0, 17.0]), torch.tensor([0.2])
    # Consider a Negative Binomial with parameters r, p where p is probability of success.
    # with mean r*(1-p)/p and variance r*(1-p)/p**2.
    # All the parameterizations below should give the same output.
    true_mean, true_var = r * (1 - p) / p, r * (1 - p) / p**2
    samples = torch.tensor([4.0, 7.0, 9.0]).unsqueeze(1)
    true_logpdf = torch.tensor([[-5.7129, -19.7673], [-4.5441, -16.5128], [-4.0630, -14.8388]])

    theta, mu = r, true_mean
    dist1 = NB(total_count=r, probs=p)
    dist2 = NB(mu=mu, theta=theta)
    dist3 = NB(mu=mu, probs=p)
    torch.testing.assert_allclose(dist1.log_prob(samples), true_logpdf)
    torch.testing.assert_allclose(dist1.mean, true_mean)
    torch.testing.assert_allclose(dist1.variance, true_var)
    x = dist1.sample()
    assert x.shape == torch.Size([2])  # len(r)
    x = dist1.sample(sample_shape=torch.Size([100]))
    assert x.shape == torch.Size([100, 2])

    torch.testing.assert_allclose(dist2.log_prob(samples), true_logpdf)
    torch.testing.assert_allclose(dist2.mean, true_mean)
    torch.testing.assert_allclose(dist2.variance, true_var)
    torch.testing.assert_allclose(dist3.log_prob(samples), true_logpdf)
    torch.testing.assert_allclose(dist3.mean, true_mean)
    torch.testing.assert_allclose(dist3.variance, true_var)
