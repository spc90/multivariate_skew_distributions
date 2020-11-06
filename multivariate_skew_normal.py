"""Multivariate skew-normal-distribution.

Author: Paul Sarbu 2020. Architecture based on SciPy's
`_multivariate.py` module by Joris Vankerschaver 2013
"""

import numpy as np
from scipy._lib._util import check_random_state
from scipy.stats._multivariate import _PSD, multi_rv_generic, multi_rv_frozen
from scipy.stats._multivariate import _squeeze_output
from scipy.stats import multivariate_normal, norm


# -----------------------------------------------------------------------------

class multivariate_skew_normal_gen(multi_rv_generic):

    def __init__(self, seed=None):
        """Initialize a multivariate skew-normal-distributed random variable.

        Parameters
        ----------
        seed : {None, int, np.random.RandomState, np.random.Generator}, optional
            Used for drawing random variates (default is None).
        """
        self._random_state = check_random_state(seed)

    def __call__(self, loc=None, scale=1, shape=0, seed=None):
        """Create a frozen multivariate skew-normal-distribution. See
        `multivariate_skew_normal_frozen` for parameters.
        """
        return multivariate_skew_normal_frozen(loc=loc, scale=scale,
                                               shape=shape, seed=seed)

    def pdf(self, x, loc=None, scale=1, shape=0):
        """Multivariate skew-normal-distribution probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        loc : array_like, optional
            Location of the distribution (default zero).
        scale : array_like, optional
            Positive definite scale matrix. This is the distribution's
            covariance matrix (default one).
        shape : array_like, optional
            Skewness parameters. Positive values skew to the right, negative to
            the left (default zero).

        Returns
        -------
        pdf : Probability density function evaluated at `x`.

        Examples
        --------
        FIXME.
        """
        dim, loc, scale, shape = self._process_parameters(loc, scale, shape)
        x = self._process_quantiles(x, dim)
        scale_info = _PSD(scale)
        scale_diag = np.diag(scale)
        lp = self._logpdf(x, loc, scale_diag, scale_info.U, scale_info.log_pdet, shape, dim)
        return _squeeze_output(np.exp(lp))

    def logpdf(self, x, loc=None, scale=1, shape=0):
        """Log of the multivariate skew-normal-distribution probability
        density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        loc : array_like, optional
            Location of the distribution (default zero).
        scale : array_like, optional
            Positive definite scale matrix. This is the distribution's
            covariance matrix (default one).
        shape : array_like, optional
            Skewness parameters. Positive values skew to the right, negative to
            the left (default zero).

        Returns
        -------
        logpdf : Log of the probability density function evaluated at `x`.

        Examples
        --------
        FIXME.
        """
        dim, loc, scale, shape = self._process_parameters(loc, scale, shape)
        x = self._process_quantiles(x, dim)
        scale_info = _PSD(scale)
        scale_diag = np.diag(scale)
        return _squeeze_output(self._logpdf(
            x, loc, scale_diag, scale_info.U, scale_info.log_pdet, shape, dim))

    def _logpdf(self, x, loc, scale_diag, U, log_pdet, shape, dim):
        """Utility method. See `pdf`, `logpdf` for parameters.
        """
        dev = x - loc  # shape (n,d)
        maha = np.square(np.dot(dev, U)).sum(axis=-1)
        L = np.dot(dev / np.sqrt(scale_diag), shape)

        A = np.log(2)  # 2
        B = dim / 2. * np.log(2. * np.pi)  # (2 pi)^{-dim/2}
        C = 0.5 * log_pdet  # (det scale)^{-1/2}
        D = 0.5 * maha  # f(x)
        E = norm.logcdf(L)  # phi(<shape, (x-loc)/sqrt(diag(scale)>)

        return A - B - C - D + E

    def rvs(self, loc=None, scale=1, shape=0, size=1, random_state=None):
        """Draw random samples from a multivariate skew-normal-distribution.

        Parameters
        ----------
        loc : array_like, optional
            Location of the distribution (default zero).
        scale : array_like, optional
            Positive definite scale matrix. This is the distribution's
            covariance matrix (default one).
        shape : array_like, optional
            Skewness parameters. Positive values skew to the right, negative to
            the left (default zero).
        size : integer, optional
            Number of samples to draw (default one).
        random_state : {None, int, np.random.RandomState, np.random.Generator}, optional
            Used for drawing random variates (default is None).

        Returns
        -------
        samples : Drawn samples from the multivariate skew-normal-distribution.

        Examples
        --------
        FIXME.
        """
        dim, loc, scale, shape = self._process_parameters(loc, scale, shape)
        if random_state is not None:
            rng = check_random_state(random_state)
        else:
            rng = self._random_state

        # generate standard samples first, i.e. without location and scale
        # using the stochastic representation of Azzalini & Capitanio (1999)

        # step 1: find scale matrix for standard distribution
        om_v = np.sqrt(np.diag(scale))
        scale_diag_inv_sqrt = np.diag(1./om_v)
        scale_z = np.matmul(np.matmul(scale_diag_inv_sqrt, scale), scale_diag_inv_sqrt)

        # step 2: create sampling covariance matrix of dimension +1
        scale_shape = np.dot(scale_z, shape)
        delta = scale_shape / np.sqrt(1 + np.dot(shape, scale_shape))
        scale_star = np.block([[1, delta], [delta[np.newaxis].T, scale_z]])

        # step 3: generate samples from standard normal distribution with mean
        # zero and covariance matrix scale_star of dimension +1
        # Note: transposed array to prep step 4
        samples_y = np.transpose(multivariate_normal.rvs(
            mean=np.zeros(dim+1), cov=scale_star, size=size, random_state=rng))

        # step 4: decide which samples need sign change based on positivity of
        # first component, and ditch the first component
        # Note: transpose array back to proper shape
        z = np.transpose(np.sign(samples_y[0]) * samples_y[1:])
        
        # Add location and scale
        samples = loc + np.matmul(z, np.diag(om_v))

        return _squeeze_output(samples)

    def dpdf(self, x, loc=None, scale=1, shape=0):
        """Derivative of the multivariate skew-normal distribution probability
        density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        loc : array_like, optional
            Location of the distribution (default zero).
        scale : array_like, optional
            Positive definite scale matrix. This is the distribution's
            covariance matrix (default one).
        shape : array_like, optional
            Skewness parameters. Positive values skew to the right, negative to
            the left (default zero).

        Returns
        -------
        dpdf : Derivative probability density function evaluated at `x`.

        Examples
        --------
        FIXME.

        """
        dim, loc, scale, shape = self._process_parameters(loc, scale, shape)
        x = self._process_quantiles(x, dim)
        scale_info = _PSD(scale)
        scale_diag = np.diag(scale)

        dev = x - loc  # (n,d)
        maha = np.square(np.dot(dev, scale_info.U)).sum(axis=-1)  # (n,)
        ss = shape / np.sqrt(scale_diag)  # (d,)
        L = np.dot(dev, ss)  # (n,)
        sm = np.dot(dev, scale_info.pinv)  # (n,d)

        A = np.log(2)
        B = dim / 2. * np.log(2. * np.pi)
        C = 0.5 * scale_info.log_pdet
        D = 0.5 * maha  # (n,)
        E = 0.5 * np.log(2. * np.pi)
        F = 0.5 * L**2  # (n,)
        G = norm.logcdf(L)  # univariate normal cdf; (n,)

        return _squeeze_output(
            np.outer(np.exp(A - B - C - D - E - F), ss)  # (n,)x(d,)=(n,d)
            - np.transpose(np.transpose(sm) * np.exp(A - B - C - D + G))
            # (n,) * (n,d)
            )

    def _process_quantiles(self, x, dim):
        """Adjust quantiles array so that last axis labels the components of
        each data point.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]
        return x

    def _process_parameters(self, loc, scale, shape):
        """Infer dimensionality from location vector, scale matrix, and shape
        vector, handle defaults, and ensure compatible dimensions.
        """
        if loc is None and scale is None:
            # check shape vector
            if shape is None:
                shape = np.zeros(1, dtype=float)
                dim = 1
            else:
                shape = np.asarray(shape, dtype=float)

                # check dims
                if shape.ndim > 1:
                    if shape.ndim == 2 and shape.shape[1] == 1:
                        shape.shape = (shape.shape[0],)
                    else:
                        raise ValueError("Array 'shape' must be a vector.")
                elif shape.ndim == 0:
                    shape.shape = (1,)

                dim = shape.shape[0]

            # create default loc and scale arays
            loc = np.zeros(dim, dtype=float)
            scale = np.eye(dim, dtype=float)

        elif loc is None:  # and scale is not None
            # infer dimensions and check correctness
            scale = np.asarray(scale, dtype=float)

            # check dims
            if scale.ndim < 2:
                dim = 1
                scale.shape = (1, 1)
            elif scale.ndim > 2:
                raise ValueError("Array 'scale' must be at most"
                                 " two-dimensional, but scale.ndim = %d" %
                                 scale.ndim)
            if scale.shape[0] != scale.shape[1]:
                raise ValueError("Array 'scale' must be square if it is"
                                 " two-dimensional, but scale.shape = %s." %
                                 str(scale.shape))

            #  create default loc vector
            dim = scale.shape[0]
            loc = np.zeros(dim)

            # check shape vector
            if shape is None:
                shape = np.zeros(dim)
            else:
                shape = np.asarray(shape, dtype=float)

                # check dims
                if shape.ndim > 1:
                    if shape.ndim == 2 and shape.shape[1] == 1:
                        shape.shape = (shape.shape[0],)
                    else:
                        raise ValueError("Array 'shape' must be a vector.")
                elif shape.ndim == 0:
                    shape.shape = (1,)
                if dim != len(shape):
                    raise ValueError("Dimension mismatch: array 'scale' is of"
                                     " shape %s, but 'shape' is a vector of"
                                     " length %d." % (str(scale.shape),
                                                      len(shape)))

        else:
            # loc is not None
            loc = np.asarray(loc, dtype=float)

            # check dims
            if loc.ndim > 1:
                if loc.ndim == 2 and loc.shape[1] == 1:
                    loc.shape = (loc.shape[0],)
                else:
                    raise ValueError("Array 'loc' must be a vector.")
            elif loc.ndim == 0:
                loc.shape = (1,)

            dim = loc.shape[0]

            # check scale array
            if scale is None:
                scale = np.eye(dim)
            else:
                scale = np.asarray(scale, dtype=float)

                # check dims
                if scale.ndim < 2:
                    scale.shape = (1, 1)
                elif scale.ndim > 2:
                    raise ValueError("Array 'scale' must be at most"
                                     " two-dimensional, but scale.ndim = %d" %
                                     scale.ndim)
                if scale.shape[0] != dim:
                    raise ValueError("Dimension mismatch: array 'scale' is of"
                                     " shape %s, but 'loc' is a vector of"
                                     " length %d." % (str(scale.shape), dim))
                if scale.shape[0] != scale.shape[1]:
                    raise ValueError("Array 'scale' must be square if it is"
                                     " two-dimensional, but scale.shape = %s." %
                                     str(scale.shape))

            # check shape vector
            if shape is None:
                shape = np.zeros(dim)
            else:
                shape = np.asarray(shape, dtype=float)

                # check dims
                if shape.ndim > 1:
                    if shape.ndim == 2 and shape.shape[1] == 1:
                        shape.shape = (shape.shape[0],)
                    else:
                        raise ValueError("Array 'shape' must be a vector.")
                elif shape.ndim == 0:
                    shape.shape = (1,)
                if dim != len(shape):
                    raise ValueError("Dimension mismatch: array 'scale' is of"
                                     " shape %s, but 'shape' is a vector of"
                                     " length %d." % (str(scale.shape),
                                                      len(shape)))

        # check for Inf & NaN values in shape vector
        if not np.isfinite(shape).all():
            raise ValueError("Values in shape vector must be finite.")

        return dim, loc, scale, shape


class multivariate_skew_normal_frozen(multi_rv_frozen):

    def __init__(self, loc=None, scale=1, shape=0, seed=None):
        """
        Create a frozen multivariate skew-normal distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        loc : array_like, optional
            Location of the distribution (default zero).
        scale : array_like, optional
            Positive definite scale matrix. This is the distribution's
            covariance matrix (default one).
        shape : array_like
            Skewness parameters. Positive values skew to the right, negative to
            the left (default zero)
        seed: {None, int, np.random.RandomState, np.random.Generator}, optional
            Used for drawing random variates (default is None).

        Examples
        --------
        FIXME.
        """
        self._dist = multivariate_skew_normal_gen(seed)
        dim, loc, scale, shape = self._dist._process_parameters(loc, scale, shape)
        self.dim, self.loc, self.scale, self.shape = dim, loc, scale, shape
        self.scale_info = _PSD(scale)

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        U = self.scale_info.U
        log_pdet = self.scale_info.log_pdet
        scale_diag = np.diag(self.scale)
        return self._logpdf(x, self.loc, scale_diag, U, log_pdet, self.shape, self.dim)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Draw random samples from a multivariate skew-normal distribution.

        Parameters
        ----------
        size : int, optional
            Number of samples to draw (default one).
        random_state : {None, int, np.random.RandomState, np.random.Generator}, optional
            Used for drawing random variates (default is None).

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.
        """
        return self._dist.rvs(loc=self.loc,
                              scale=self.scale,
                              df=self.shape,
                              size=size,
                              random_state=random_state)


# -----------------------------------------------------------------------------
multivariate_skew_normal = multivariate_skew_normal_gen()
