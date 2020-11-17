"""Extension to SciPy's multivariate normal-distribution.

We're extending the existing class by adding a method for computing the
derivative/gradient wrt the variable x.

Author: Paul Sarbu 2020. Architecture based on SciPy's
`_multivariate.py` module by Joris Vankerschaver 2013
"""

import numpy as np
from scipy.stats._multivariate import multivariate_normal_gen, multivariate_normal_frozen
from scipy.stats._multivariate import _PSD, _squeeze_output


# -----------------------------------------------------------------------------

class multivariate_normal_extended_gen(multivariate_normal_gen):

    def __init__(self, seed=None):
        super().__init__(seed)

    def __call__(self, mean=None, cov=1, allow_singular=False, seed=None):
        super().__call__(mean, cov, allow_singular, seed)

    def dpdf(self, x, mean=None, cov=1, allow_singular=False):
        """Derivative of the multivariate normal distribution probability
        density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        mean : array_like, optional
            Mean of the distribution (default zero)
        cov : array_like, optional
            Covariance matrix of the distribution (default one)

        Returns
        -------
        dpdf : Derivative probability density function evaluated at `x`.

        Examples
        --------
        FIXME.

        """
        dim, mean, cov = super()._process_parameters(None, mean, cov)
        x = super()._process_quantiles(x, dim)
        psd = _PSD(cov, allow_singular=allow_singular)

        # logpdf
        dev = x - mean
        maha = np.sum(np.square(np.dot(dev, psd.U)), axis=-1)
        logp = -0.5 * (psd.rank * np.log(2 * np.pi) + psd.log_pdet + maha)

        # derivative part of the power: -(x-mean) * cov^{-1}
        sm = np.dot(-dev, psd.pinv)

        res = np.transpose(np.transpose(sm) * np.exp(logp))
        return _squeeze_output(res)


class multivariate_normal_extended_frozen(multivariate_normal_frozen):

    def __init__(self, mean=None, cov=1, allow_singular=False, seed=None,
                 maxpts=None, abseps=1e-5, releps=1e-5):
        super().__init__(
            mean, cov, allow_singular, seed, maxpts, abseps, releps)


# -----------------------------------------------------------------------------
multivariate_normal_extended = multivariate_normal_extended_gen()
