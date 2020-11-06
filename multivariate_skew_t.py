"""Multivariate skew-normal-distribution.

Author: Paul Sarbu 2020. Architecture based on SciPy's
`_multivariate.py` module by Joris Vankerschaver 2013, and
the `multivariate_t.py` class by Gregory Gundersen 2020
"""

import numpy as np
from scipy._lib._util import check_random_state
from scipy.stats._multivariate import _PSD, multi_rv_generic, multi_rv_frozen
from scipy.stats._multivariate import _squeeze_output
from scipy.special import gammaln

from scipy.stats import t as student_t
from multivariate_skew_normal import multivariate_skew_normal
from multivariate_t import multivariate_t


# -----------------------------------------------------------------------------

class multivariate_skew_t_gen(multi_rv_generic):

    def __init__(self, seed=None):
        """Initialize a multivariate skew-t distributed random variable.

        Parameters
        ----------
        seed : {None, int, np.random.RandomState, np.random.Generator}, optional
            Used for drawing random variates (default is None).
        """
        self._random_state = check_random_state(seed)

    def __call__(self, df, loc=None, scale=1, shape=0, seed=None):
        """Create a frozen multivariate skew-t distribution. See
        `multivariate_skew_t_frozen` for parameters.
        """
        return multivariate_skew_t_frozen(df, loc=loc, scale=scale,
                                          shape=shape, seed=seed)

    def pdf(self, x, df, loc=None, scale=1, shape=0):
        """Multivariate skew-t distribution probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        df : int or float
            Degrees of freedom.
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
        self._process_degrees_of_freedom(df)

        if np.inf == df:
            # multivariate skew-normal distribution
            lp = multivariate_skew_normal.logpdf(x, loc=loc, scale=scale,
                                                 shape=shape)
        else:
            dim, loc, scale, shape = self._process_parameters(loc, scale, shape)
            x = self._process_quantiles(x, dim)
            scale_info = _PSD(scale)
            scale_diag = np.diag(scale)

            # generic multivariate skew-t distribution
            lp = self._logpdf(x, df, loc, scale_info.U, scale_info.log_pdet,
                              scale_diag, shape, dim)

        return _squeeze_output(np.exp(lp))

    def logpdf(self, x, df, loc=None, scale=1, shape=0):
        """Log of the multivariate skew-t distribution probability
        density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        df : int or float
            Degrees of freedom.
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
        self._process_degrees_of_freedom(df)

        if np.inf == df:
            # multivariate skew-normal distribution
            lp = multivariate_skew_normal.logpdf(x, loc=loc, scale=scale,
                                                 shape=shape)
        else:
            dim, loc, scale, shape = self._process_parameters(loc, scale, shape)
            x = self._process_quantiles(x, dim)
            scale_info = _PSD(scale)
            scale_diag = np.diag(scale)

            # generic multivariate skew-t distribution
            lp = self._logpdf(x, df, loc, scale_info.U, scale_info.log_pdet,
                              scale_diag, shape, dim)

        return _squeeze_output(lp)

    def _logpdf(self, x, df, loc, U, log_pdet, scale_diag, shape, dim):
        """Utility method. See `pdf`, `logpdf` for parameters.
        """
        dev = x - loc  # Careful! x.shape has dim as last axis!
        maha = np.square(np.dot(dev, U)).sum(axis=-1)  # (n,)

        # log multivariate t distribution
        t = 0.5 * (df + dim)
        A = gammaln(t)
        B = gammaln(0.5 * df)
        C = dim/2. * np.log(df * np.pi)
        D = 0.5 * log_pdet
        E = -t * np.log(1 + (1./df) * maha)
        
        log_t = A - B - C - D + E
        
        # factor of log 2
        F = np.log(2)
        
        # log cdf of reparametrized univariate t distribution
        ss = shape / np.sqrt(scale_diag)  # (d,)
        L = np.dot(dev, ss)  # (n,)
        weight = np.sqrt((df + dim) / (df + maha))  # (n,)
        G = student_t.logcdf(weight * L, df + dim)

        return F + log_t + G

    def rvs(self, df, loc=None, scale=1, shape=0, size=1, random_state=None):
        """Draw random samples from a multivariate skew-t distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        df : int or float
            Degrees of freedom.
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
        samples : Drawn samples from the multivariate skew-t distribution.

        Examples
        --------
        FIXME.
        """
        self._process_degrees_of_freedom(df)
        dim, loc, scale, shape = self._process_parameters(loc, scale, shape)

        if random_state is not None:
            rng = check_random_state(random_state)

        else:
            rng = self._random_state

        if np.inf == df:
            # multivariate normal distribution
            x = np.ones(size)

        else:
            # generic multivariate t-distribution
            x = rng.chisquare(df, size=size) / df
        
        z = multivariate_skew_normal.rvs(loc=np.zeros(dim), scale=scale,
                                         shape=shape, size=size,
                                         random_state=rng)
        # z has dimensions squeezed, so we need to promote to 2D
        z = self._process_quantiles(z, dim)  # (n,d)        
        # add location and scale
        samples = loc + z / np.sqrt(x)[:, None]
        
        return _squeeze_output(samples)

    def dpdf(self, x, df, loc=None, scale=1, shape=0):
        """Derivative of the multivariate skew-t distribution probability
        density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        df : int or float
            Degrees of freedom.
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
        self._process_degrees_of_freedom(df)
        
        if np.inf == df:
            # multivariate skew-normal distribution
            out = multivariate_skew_normal.dpdf(x, loc=loc, scale=scale,
                                                shape=shape)

        else:
            dim, loc, scale, shape = self._process_parameters(
                loc, scale, shape)
            x = self._process_quantiles(x, dim)
            scale_info = _PSD(scale)
            scale_diag = np.diag(scale)
            
            # generic multivariate skew-t-distribution, step by step, 
            # using the product rule for the gradient
            
            # first term contains derivative of regular multivariate t
            dev = x - loc  # (n,d)
            maha = np.square(np.dot(dev, scale_info.U)).sum(axis=-1)  # (n,)

            t = 0.5 * (df + dim)
            A = gammaln(t)
            B = gammaln(0.5 * df)
            C = dim/2. * np.log(df * np.pi)
            D = 0.5 * scale_info.log_pdet
            E = -(t+1) * np.log(1 + (1./df) * maha)  # (n,)
            
            ss = shape / np.sqrt(scale_diag)  # (d,)
            
            # logcdf of univariate student t distribution
            L = np.dot(dev, ss)  # (n,)
            weight = np.sqrt((df + dim) / (df + maha))  # (n,)
            z = weight * L  # (n,)
            F = student_t.logcdf(z, df + dim)  # (n,)
            
            sm = np.dot(dev, scale_info.pinv)  # (n,d)
            
            # First term of product rule
            left = -2. * t / df * np.transpose(
                np.transpose(sm) * np.exp(A - B - C - D + E + F))  # (n,d)
            
            # second term is 2 * multivariate t * univariate t 
            # reparametrized * derivative of reparametrization, so we can
            # reuse a lot of values
            
            G = -t * np.log(1 + (1./df) * maha)  # (n,)
            
            # TODO: could be done explicitly to save some computation
            H = student_t.logpdf(z, df + dim)  # (n,)
            
            t_d_times_t_1_cdf = np.exp(A - B - C - D + G + H)  # (n,)
            
            # now we compute the derivative of reparametrization parts
            J = -0.5 * np.sqrt((df + dim) / (df + maha)**3)  # (n,)
            K = np.sqrt((df + dim) / (df + maha))  # (n,)
            
            # Second term of product rule
            tmp = (
                np.transpose(np.transpose(sm) * (J * z))  # (n,d)
                + np.outer(K, ss)  # (n,)x(d,)=(n,d)
                )
            right = 2 * np.transpose(np.transpose(tmp) * t_d_times_t_1_cdf)

            out = left + right  # (n,d)+(n,d)=(n,d)

        return _squeeze_output(out)

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
    
    def _process_degrees_of_freedom(self, df):
        """Make sure degrees of freedom are valid. Separate treatment to
        avoid duplication of code when df == np.inf
        """
        if not (isinstance(df, (int,float)) and df > 0):
            raise ValueError("'df' must be a positive integer or 'np.inf' "
                             "but is of type %s" % type(df) + " and value "
                             "%s" % str(df))

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


class multivariate_skew_t_frozen(multi_rv_frozen):

    def __init__(self, df, loc=None, scale=1, shape=0, seed=None):
        """
        Create a frozen multivariate skew-t-distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        df : int or float
            Degrees of freedom.
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
        self._dist = multivariate_skew_t_gen(seed)
        self._process_degrees_of_freedom(df)
        dim, loc, scale, shape = self._dist._process_parameters(loc, scale, shape)
        self.dim, self.df, self.loc, self.scale, self.shape = dim, df, loc, scale, shape
        self.scale_info = _PSD(scale)

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        U = self.scale_info.U
        log_pdet = self.scale_info.log_pdet
        return self._dist._logpdf(x, self.df, self.loc, U, log_pdet, self.shape, self.dim)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Draw random samples from a multivariate skew-t distribution.

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
        return self._dist.rvs(self.df,
                              loc=self.loc,
                              scale=self.scale,
                              shape=self.shape,
                              size=size,
                              random_state=random_state)


# -----------------------------------------------------------------------------
multivariate_skew_t = multivariate_skew_t_gen()
