from scipy.stats import norm, cauchy,laplace, stats
from scipy.optimize import minimize, root_scalar
from statsmodels.robust.scale import mad
import warnings
import torch
from scipy.stats import norm

torch.set_printoptions(precision=15)


def beta_cauchy(x):
    """
    Find the function beta for the mixed normal prior with Cauchy
    tails.  It is assumed that the noise variance is equal to one.
    
    Parameters:
    x - a real value or vector
    """
    x = torch.tensor(x,dtype=torch.float64)

    phix = torch.tensor(norm.pdf(x, loc=0, scale=1))
    
    j = (x != 0)

    beta = x.clone()

    beta = torch.where(j == False, -1/2, beta)

    beta[j] = (torch.tensor(norm.pdf(0, loc=0, scale=1)) / phix[j] - 1) / (x[j] ** 2) - 1

    return beta


def beta_laplace(x, s=1, a=0.5):
    """
    The function beta for the Laplace prior given parameter a and s (sd)
    
    x - the value or vector of data values (torch tensor)
    s - the value or vector of standard deviations; if vector, must have the same length as x
    a - the scale parameter of the Laplace distribution
    """
    # Compute xpa and xma
    x = torch.abs(x)

    xpa = x / s + s * a

    xma = x / s - s * a

    rat1 = torch.tensor(1 / xpa, dtype=torch.float64)

    rat1[xpa < 35] = torch.tensor(norm.cdf(-xpa[xpa < 35], loc=0, scale=1) / norm.pdf(xpa[xpa < 35], loc=0, scale=1))

    rat2 = torch.tensor(1 / torch.abs(xma), dtype=torch.float64)

    xma = torch.where(xma > 35, torch.tensor(35.0), xma)

    rat2[xma > -35] = torch.tensor(norm.cdf(xma[xma > -35], loc=0, scale=1) / norm.pdf(xma[xma > -35], loc=0, scale=1))
    
    beta = (a * s) / 2 * (rat1 + rat2) - 1
    
    return beta

def ebayesthresh(x, prior="laplace", a=0.5, bayesfac=False, sdev=None, verbose=False, threshrule="median", universalthresh=True, stabadjustment=None):
    pr = prior[0:1]

    if sdev is None: sdev = torch.tensor([float('nan')])
    else: sdev = torch.tensor([sdev])
    
    if len(sdev) == 1:
        if stabadjustment is not None:
            raise ValueError("Argument stabadjustment is not applicable when variances are homogeneous.")
        if torch.isnan(torch.tensor(sdev)):
            sdev = mad(x, center=0)
        stabadjustment_condition = True
    else:
        if pr == "c":
            raise ValueError("Standard deviation has to be homogeneous for Cauchy prior.")
        if sdev.numel() != x.numel():
            raise ValueError("Standard deviation has to be homogeneous or have the same length as observations.")
        if stabadjustment is None:
            stabadjustment = False
        stabadjustment_condition = stabadjustment

    if stabadjustment_condition:
        m_sdev = torch.mean(sdev.float()) 
        s = sdev / m_sdev 
        x = x / m_sdev
    else:
        s = sdev

    if pr == "l" and a is None:
        pp = wandafromx(x, s, universalthresh)
        w = pp["w"]
        a = pp["a"]
    else:
        w = wfromx(x, s, prior, a, universalthresh)

    if pr != "m" or verbose:
        tt = tfromw(w, s, prior=prior, bayesfac=bayesfac, a=a)[0]
        if stabadjustment_condition:
            tcor = tt * m_sdev
        else:
            tcor = tt

    if threshrule == "median":
        muhat = postmed(x, s, w, prior=prior, a=a)
    elif threshrule == "mean":
        muhat = postmean(x, s, w, prior=prior, a=a)
    elif threshrule == "hard":
        muhat = threshld(x, tt)
    elif threshrule == "soft":
        muhat = threshld(x, tt, hard=False)
        if threshrule == "none":
            muhat = None
    else:
        raise ValueError(f"Unknown threshold rule: {threshrule}")

    if stabadjustment_condition:
        muhat = muhat * m_sdev

    if not verbose:
        return muhat
    retlist = {
            'muhat': muhat,
            'x': x,
            'threshold.sdevscale': tt,
            'threshold.origscale': tcor,
            'prior': prior,
            'w': w,
            'a': a,
            'bayesfac': bayesfac,
            'sdev': sdev,
            'threshrule': threshrule
        }
    if pr == "c":
        del retlist['a']
    if threshrule == "none":
        del retlist['muhat']
    return retlist

    
def cauchy_medzero(x, z, w):
    hh = z - x

    dnhh = torch.tensor(norm.pdf(hh, loc=0, scale=1))

    yleft = torch.tensor(norm.cdf(hh, loc=0, scale=1)) - z * dnhh + ((z * x - 1) * dnhh * torch.tensor(norm.cdf(-x, loc=0, scale=1)) ) / torch.tensor(norm.pdf(x, loc=0, scale=1))

    yright2 = 1 + torch.exp(-z**2 / 2) * (z**2 * (1/w - 1) - 1)

    return yright2 / 2 - yleft


def cauchy_threshzero(z, w):
    y = (torch.tensor(norm.cdf(z, loc=0, scale=1)) - z * torch.tensor(norm.pdf(z, loc=0, scale=1)) - 0.5 - (z**2 * torch.exp(-z**2 / 2) * (1/w - 1)) / 2)
    return y


def isotone(x, wt=None, increasing=False):
    """
    Find the weighted least squares isotone fit to the sequence x,
    the weights given by the sequence wt. If increasing == True,
    the curve is set to be increasing, otherwise to be decreasing.
    The vector ip contains the indices on the original scale of the
    breaks in the regression at each stage.
    """
    if wt is None:
        wt = torch.ones_like(x)
        
    nn = len(x)

    if nn == 1:
        x = x

    if not increasing:
        x = -x
 
    ip = torch.arange(nn)
    dx = torch.diff(x)
    nx = len(x)

    while (nx > 1) and (torch.min(dx) < 0):
        jmax = torch.where((torch.cat([dx <= 0, torch.tensor([False])]) & torch.cat([torch.tensor([True]), dx > 0])))[0]
        jmin = torch.where((torch.cat([dx > 0, torch.tensor([True])]) & torch.cat([torch.tensor([False]), dx <= 0])))[0]
        

        for jb in range(len(jmax)):
            ind = torch.arange(jmax[jb], jmin[jb] + 1)
            wtn = torch.sum(wt[ind])
            x[jmax[jb]] = torch.sum(wt[ind] * x[ind]) / wtn
            wt[jmax[jb]] = wtn
            x[jmax[jb] + 1:jmin[jb] + 1] = torch.nan

        ind = ~torch.isnan(x)
        x = x[ind]
        wt = wt[ind]
        ip = ip[ind]
        dx = torch.diff(x)
        nx = len(x)

    jj = torch.zeros(nn, dtype=torch.int32)
    jj[ip] = 1
    z = x[torch.cumsum(jj, dim=0) - 1]

    if not increasing:
        z = -z

    return z.tolist()


def laplace_threshzero(x, s=1, w=0.5, a=0.5):
    """
    This function needs to be zeroed to find the threshold using Laplace prior.

    Parameters:
    x (float): Input value
    s (float): Standard deviation (default: 1)
    w (float): Mean (default: 0.5)
    a (float): Input value where a < 20 (default: 0.5)
    """
    a = min(a, 20)
    
    xma = x / s - s * a
    
    z = z = torch.tensor(norm.cdf(xma, loc=0, scale=1)) - (1 / a) * (1 / s * torch.tensor(norm.pdf(xma, loc=0, scale=1))) * (1 / w + beta_laplace(x, s, a))
    
    return z

def negloglik_laplace(xpar, xx, ss, tlo, thi):
    """
    Marginal negative log likelihood function for Laplace prior. 
    Constraints for thresholds need to be passed externally.

    Parameters:
    xpar : array-like, shape (2,)
        Vector of two parameters:
        xpar[0]: a value between [0, 1] which will be adjusted to range of w 
        xpar[1]: inverse scale (rate) parameter ("a")
    xx : array-like
        Data
    ss : array-like
        Vector of standard deviations
    tlo : array-like
        Lower bound of thresholds
    thi : array-like
        Upper bound of thresholds

    Returns:
    neg_loglik : float
        Negative log likelihood
    """
    a = xpar[1]
    
    wlo = wfromt(thi, ss, a=a)
    whi = wfromt(tlo, ss, a=a)
    wlo = torch.max(wlo)
    whi = torch.min(whi)
    
    loglik = torch.sum(torch.log(1 + (xpar[0] * (whi - wlo) + wlo) *
                           beta_laplace(xx, ss, a)))
    
    return -loglik

def postmean(x, s=1, w=0.5, prior="laplace", a=0.5):
    """
    Find the posterior mean for the appropriate prior for given x, s (sd), w, and a.
    """
    pr = prior[0:1]
    if pr == "l":
        mutilde = postmean_laplace(x, s, w, a=a)
    elif pr == "c":
        if torch.any(s != 1):
            raise ValueError("Only standard deviation of 1 is allowed for Cauchy prior.")
        mutilde = postmean_cauchy(x, w)
    else:
        raise ValueError("Unknown prior type.")
        
    return mutilde

def postmean_cauchy(x, w):
    """
    Find the posterior mean for the quasi-Cauchy prior with mixing
    weight w given data x, which may be a scalar or a vector.
    """
    ind = torch.nonzero(x == 0)
    x = x[x != 0] 
    ex = torch.exp(-x**2/2)
    z = w * (x - (2 * (1 - ex))/x)
    z = z / (w * (1 - ex) + (1 - w) * ex * x**2)
    muhat = z
    muhat[ind] = torch.tensor([0.0])
    
    return muhat


def postmean_laplace(x, s=1, w=0.5, a=0.5):
    a = min(a, 20)
    
    # First find the probability of being non-zero
    w_post = wpost_laplace(w, x, s, a)

    # Now find the posterior mean conditional on being non-zero
    sx = torch.sign(x)
    x = torch.abs(x)
    xpa = x / s + s * a
    xma = x / s - s * a
    xpa = torch.minimum(xpa, torch.tensor(35.0))
    xma = torch.maximum(xma, torch.tensor(-35.0))

    cp1 = torch.tensor(norm.cdf(xma, loc=0, scale=1))
    cp2 = torch.tensor(norm.cdf(-xpa, loc=0, scale=1))
    ef = torch.exp(torch.minimum(2 * a * x, torch.tensor(100.0, dtype=torch.float32)))
    postmean_cond = x - a * s**2 * (2 * cp1 / (cp1 + ef * cp2) - 1)
    
    return sx * w_post * postmean_cond

def postmed(x, s=1, w=0.5, prior="laplace", a=0.5):
    """
    Find the posterior median for the appropriate prior for given x, s (sd), w, and a.

    Parameters:
        x (array-like): Observations.
        s (float): Standard deviation.
        w (float): Weight parameter.
        prior (str): Type of prior ("laplace" or "cauchy").
        a (float): Parameter a.

    Returns:
        array-like: Posterior median estimates.
    """
    pr = prior[0:1]
    if pr == "l":
        muhat = postmed_laplace(x, s, w, a)
    elif pr == "c":
        if np.any(s != 1):
            raise ValueError("Only standard deviation of 1 is allowed for Cauchy prior.")
        muhat = postmed_cauchy(x, w)
    else:
        raise ValueError(f"Unknown prior: {prior}")
    return muhat

def postmed_cauchy(x, w):
    """
    Find the posterior median of the Cauchy prior with mixing weight w,
    pointwise for each of the data points x
    """
    nx = len(x)
    zest = torch.full((nx,), float('nan'))
    w = torch.full((nx,), w)
    ax = torch.abs(x)
    j = ax < 20
    zest[~j] = ax[~j] - 2 / ax[~j]
    
    if torch.sum(j) > 0:
        zest[j] = vecbinsolv(zf=torch.zeros(torch.sum(j)), fun=cauchy_medzero,
                             tlo=0, thi=torch.max(ax[j]), z=ax[j], w=w[j])
                             
    zest[zest < 1e-7] = 0
    zest = torch.sign(x) * zest
    
    return zest

def postmed_laplace(x, s=1, w=0.5, a=0.5):
    """
    Find the posterior median for the Laplace prior for given x (observations), s (sd), w, and a.
    
    Parameters:
        x (array-like): Observations.
        s (float): Standard deviation.
        w (float): Weight parameter.
        a (float): Parameter a.
        
    Returns:
        array-like: Posterior median estimates.
    """
    # Only allow a < 20 for input value
    a = min(a, 20)
    
    # Work with the absolute value of x, and for x > 25 use the approximation
    # to dnorm(x-a)*beta_laplace(x, a)
    sx = torch.sign(x)
    x = torch.abs(x)
    xma = x / s - s * a
    zz = 1 / a * (1 / s * torch.tensor(norm.pdf(xma, loc=0, scale=1))) * (1 / w + beta_laplace(x, s, a))
    zz[xma > 25] = 0.5
    mucor = torch.tensor(norm.ppf(torch.minimum(zz, torch.tensor(1))))
    muhat = sx * torch.maximum(torch.tensor(0), xma - mucor) * s
    
    return muhat

def threshld(x, t, hard=True):
    """
    Threshold the data x using threshold t.
    If hard=True, use hard thresholding.
    If hard=False, use soft thresholding.
    """
    
    if hard:
        z = x * (torch.abs(x) >= t)
    else:
        z = torch.sign(x) * torch.maximum(torch.tensor(0.0), torch.abs(x) - t)
    
    
    return z



def wandafromx(x, s=1, universalthresh=True):
    """
    Find the marginal max lik estimators of w and a given standard  deviation s,
    using a bivariate optimization; If universalthresh=TRUE, the thresholds will 
    be upper bounded by universal threshold adjusted by standard deviation. 
    The threshold is constrained to lie between 0 and sqrt ( 2 log (n)) *   s. 
    Otherwise, threshold can take any nonnegative value;  If running R, 
    the routine optim is used; in S-PLUS the routine is nlminb.
    """
    # Range for thresholds
    if universalthresh:
        thi = torch.sqrt(2 * torch.log(torch.tensor(len(x)))) * s
    else:
        thi = torch.inf

    if isinstance(s, int):
        tlo = torch.zeros(len(str(s)))
    else:
        tlo = np.zeros(len(s))
    lo = torch.tensor([0, 0.04])
    hi = torch.tensor([1, 3])
    startpar = torch.tensor([0.5, 0.5])

    if 'optim' in globals():
        result = minimize(negloglik_laplace, startpar, method='L-BFGS-B', bounds=[(lo[0], hi[0]), (lo[1], hi[1])], args=(x, s, tlo, thi))
        uu = result.x
    else:
        result = minimize(negloglik_laplace, startpar, bounds=[(lo[0], hi[0]), (lo[1], hi[1])], args=(x, s, tlo, thi))
        uu = result.x

    a = uu[1]
    wlo = wfromt(thi, s, a=a)
    whi = wfromt(tlo, s, a=a)
    wlo = torch.max(wlo)
    whi = torch.min(whi)
    w = uu[0] * (whi - wlo) + wlo
    return {'w': w, 'a': a}

def mad(x, center=None, constant=1.4826, na_rm=False, low=False, high=False):
    if na_rm:
        x = x[~torch.isnan(x)]

    if center is None:
        center = torch.median(x)

    if low:
        center = torch.quantile(x, 0.25)
    elif high:
        center = torch.quantile(x, 0.75)

    # Median Absolute Deviation 계산
    mad_value = torch.median(torch.abs(x - center)) * constant
    
    return mad_value



def wfromt(tt, s=1, prior="laplace", a=0.5):
    """
    Find the weight that has posterior median threshold tt, 
    given s (sd) and a.
    
    tt - Threshold value or vector of values.
    s - A single value or a vector of standard deviations if the Laplace prior is used. If
    a vector, must have the same length as tt. Ignored if Cauchy prior is used.
    prior - Specification of prior to be used; can be "cauchy" or "laplace".
    a - Scale factor if Laplace prior is used. Ignored if Cauchy prior is used.
    """
    tt = torch.tensor(tt, dtype=torch.float64)
    
    pr = prior[0:1]
    if pr == "l":
        tma = torch.tensor(tt / s - s * a)
        wi = 1 / torch.abs(tma)
        wi[tma > -35] = torch.tensor(norm.cdf(tma[tma > -35], loc=0, scale=1)/norm.pdf(tma[tma > -35], loc=0, scale=1))
        wi = a * s * wi - beta_laplace(tt, s, a)
        
    if pr == "c":
        dnz = norm.pdf(tt, loc=0, scale=1)
        wi = 1 + (torch.tensor(norm.cdf(tt, loc=0, scale=1)) - tt * dnz - 1/2) / (torch.sqrt(torch.tensor(torch.pi/2)) * dnz * tt**2)
        wi[~torch.isfinite(wi)] = 1
        
    return 1 / wi


def wfromx(x, s=1, prior="laplace", a=0.5, universalthresh=True):
    """
    Given the vector of data x and s (sd),
    find the value of w that zeroes S(w) in the
    range by successive bisection, carrying out nits harmonic bisections
    of the original interval between wlo and 1.  
  
    x - Vector of data.
    s - A single value or a vector of standard deviations if the Laplace prior is used. If
    a vector, must have the same length as x. Ignored if Cauchy prior is used.
    prior - Specification of prior to be used; can be "cauchy" or "laplace".
    a - Scale factor if Laplace prior is used. Ignored if Cauchy prior is used.
    universalthresh - If universalthresh = TRUE, the thresholds will be upper bounded by universal
    threshold; otherwise, the thresholds can take any non-negative values.
    """
    s = torch.tensor(s, dtype=torch.float)

    pr = prior[0:1]
    
    if pr == "c":
        s = torch.tensor(1, dtype=torch.float)

    if universalthresh:
        tuniv = torch.sqrt(2 * torch.log(torch.tensor(len(x)))) * s
        wlo = wfromt(tuniv, s, prior, a)
        wlo = torch.max(wlo)
    else:
        wlo = 0

    if pr == "l":
        beta = beta_laplace(x, s, a)
    elif pr == "c":
        beta = beta_cauchy(x)

    whi = 1
    beta = torch.minimum(beta, torch.tensor(1e20))

    shi = torch.sum(beta / (1 + beta))

    if shi >= 0:
        shi =  1

    slo = torch.sum(beta / (1 + wlo * beta))

    if slo <= 0:
        slo = wlo

    for _ in range(1,31):
        wmid = torch.sqrt(wlo * whi)
        smid = torch.sum(beta / (1 + wmid * beta))
        if smid == 0:
            smid = wmid
        if smid > 0:
            wlo = wmid
        else:
            whi = wmid

    return torch.sqrt(wlo * whi)



def wmonfromx(xd, prior="laplace", a=0.5,  tol=1e-08, maxits=20):
    """
    Find the monotone marginal maximum likelihood estimate of the
    mixing weights for the Laplace prior with parameter a.  It is
    assumed that the noise variance is equal to one.
    Find the beta values and the minimum weight
    Current version allows for standard deviation of 1 only.
    xd - A vector of data.
    prior - Specification of the prior to be used; can be cauchy or laplace.
    a - Scale parameter in prior if prior="laplace". Ignored if prior="cauchy".
    tol - Absolute tolerance to within which estimates are calculated.
    maxits - Maximum number of weighted least squares iterations within the calculation.
    """
    pr = prior[0:1]

    nx = len(xd)

    wmin = wfromt(torch.sqrt(2 * torch.log(torch.tensor(len(xd)))), prior=prior, a=a)

    winit = torch.tensor(1)

    if pr == "l":
        beta = beta_laplace(xd, a=a)
    if pr == "c":
        beta = beta_cauchy(xd)

    w = winit.repeat_interleave(len(beta))

    for j in range(maxits):
        aa = w + 1 / beta
        ps = w + aa
        ww = 1 / aa ** 2
        wnew = torch.tensor(isotone(ps, ww, increasing=False))
        wnew = torch.maximum(wmin, wnew)
        wnew = torch.minimum(torch.tensor(1), wnew)
        zinc = torch.max(torch.abs(torch.diff(wnew)))
        w = wnew
        if zinc < tol:
            return w

    # warnings.filterwarnings("More iterations required to achieve convergence")
    return w


def vecbinsolv(zf, fun, tlo, thi, nits=30, **kwargs):
    """
    Given a monotone function fun, and a vector of values zf find a vector of numbers t such that f(t) = zf.
    The solution is constrained to lie on the interval (tlo, thi).
    
    The function fun may be a vector of increasing functions.
    
    Present version is inefficient because separate calculations are done for each element of z, 
    and because bisections are done even if the solution is outside the range supplied.
    
    It is important that fun should work for vector arguments. 
    Additional arguments to fun can be passed through *args and **kwargs.
    
    Works by successive bisection, carrying out nits harmonic bisections of the interval between tlo and thi.
    
    zf- the right hand side of the equation(s) to be solved

    fun - an increasing function of a scalar argument, or a vector of such functions

    tlo - lower limit of interval over which the solution is sought

    thi-upper limit of interval over which the solution is sought

    nits-number of binary subdivisions carried out
    """
    s = kwargs.get('s', 0) 
    w = kwargs.get('w', 0) 
    a = kwargs.get('a', 0) 
    
    if isinstance(zf, (int, float, str, bool)) :
        nz = len(str(zf))
    else : nz = zf.shape[0]
    
    tlo = torch.full((nz,), tlo, dtype=torch.float32)
    thi = torch.full((nz,), thi, dtype=torch.float32)
    
    if tlo.numel() != nz:
        raise ValueError("Lower constraint has to be homogeneous or have the same length as the number of functions.")
    if thi.numel() != nz:
        raise ValueError("Upper constraint has to be homogeneous or have the same length as the number of functions.")

    # carry out nits bisections
    for _ in range(nits):
        tmid = (tlo + thi) / 2
        if fun == cauchy_threshzero:
            fmid = fun(tmid,w=w)
        elif fun == laplace_threshzero:
            fmid = fun(tmid, s=s,w=w,a=a)
        elif fun == beta_cauchy:
            fmid = fun(tmid)
        elif fun == beta_laplace:
            fmid = fun(tmid, s=s, a=a)
        else:
            fmid = fun(tmid, **kwargs)

        indt = fmid <= zf

        tlo = torch.where(indt, tmid, tlo)
        thi = torch.where(~indt, tmid, thi)
        
    tsol = (tlo + thi) / 2 
    
    return tsol


def tfromw(w, s=1, prior="laplace", bayesfac=False, a=0.5):
    """
    
    This function finds the threshold or threshold vector corresponding to the given weight vector w and standard deviation s under the specified prior distribution.
    If bayesfac is set to True, it finds the Bayes factor thresholds; otherwise, it finds the posterior median thresholds.
    If the Laplace prior distribution is used, a specifies the value of the inverse scale (i.e., rate) parameter where a < 20.

    Parameters:
    w (array-like): Weight vector
    s (float): Standard deviation (default: 1)
    prior (str): Prior distribution (default: "laplace")
    bayesfac (bool): Whether to find the Bayes factor thresholds (default: False)
    a (float): Input value where a < 20 (default: 0.5)

    Returns:
    array-like: Threshold or threshold vector
    """
    pr = prior[0:1]
    
    if not isinstance(w, torch.Tensor):
        w = torch.tensor(w)
    if not isinstance(s, torch.Tensor):
        s = torch.tensor(s)
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if bayesfac:
        z = 1 / w - 2
        if pr == "l":
            if w.dim() == 0 or len(w) >= len(s):
                zz = z
            else:
                zz = z.repeat(len(s))
            tt = vecbinsolv(zz, beta_laplace, 0, 10, s=s, a=a)
        elif pr == "c":
            tt = vecbinsolv(z, beta_cauchy, 0, 10)
    else:
        z = torch.tensor(0.0)
        if pr == "l":
            zz = torch.zeros(max(s.numel(), w.numel()))
            upper_bound = s * (25 + s * a)
            upper_bound = upper_bound.item()
            tt = vecbinsolv(zz, laplace_threshzero, 0, upper_bound, s=s, w=w, a=a)
        elif pr == "c":
            tt = vecbinsolv(z, cauchy_threshzero, 0, 10, w=w)

    return tt


def tfromx(x, s=1, prior="laplace", bayesfac=False, a=0.5, universalthresh=True):
    """
    Given the data x, the prior, and any other parameters, find the
    threshold corresponding to the marginal maximum likelihood
    estimator of the mixing weight.
    
    x - Vector of data.
    s - A single value or a vector of standard deviations if the Laplace prior is used. If
    a vector, must have the same length as x. Ignored if Cauchy prior is used.
    prior - Specification of prior to be used; can be "cauchy" or "laplace".
    bayesfac - Specifies whether Bayes factor threshold should be used instead of posterior
    median threshold.
    a - Scale factor if Laplace prior is used. Ignored if Cauchy prior is used.
    universalthresh - If universalthresh = TRUE, the thresholds will be upper bounded by universal
    threshold; otherwise, the thresholds can take any non-negative values.
    """
    pr = prior[0:1]
    a = torch.tensor(a)
    if pr == "c":
        s = 1
    if pr == "l" and torch.isnan(a):
        wa = wandafromx(x, s, universalthresh)
        w = wa['w']
        a = wa['a']
    else:
        w = wfromx(x, s, prior=prior, a=a)
    return tfromw(w, s, prior=prior, bayesfac=bayesfac, a=a)[0]

def wpost_laplace(w, x, s=1, a=0.5):
    # Calculate the posterior weight for non-zero effect
    laplace_beta = beta_laplace(x, s, a)
    return 1 - (1 - w) / (1 + w * laplace_beta)

def zetafromx(xd, cs, pilo=None, prior="laplace", a=0.5):
    """
    Given a sequence xd, a vector of scale factors cs, and a lower limit pilo,
    find the marginal maximum likelihood estimate of the parameter zeta
    such that the prior probability is of the form median(pilo, zeta*cs, 1).

    If pilo is None, then it is calculated according to the sample size
    to correspond to the universal threshold.

    Find the beta values and the minimum weight if necessary.

    Current version allows for standard deviation of 1 only.
    
    xd - A vector of data.
    cs - A vector of scale factors, of the same length as xd.
    pilo The lower limit for the estimated weights. If pilo=NA it is calculated according
    to the sample size to be the weight corresponding to the universal threshold √ 2 log n.
    prior Specification of prior to be used conditional on the mean being nonzero; can be
    cauchy or laplace.
    a - Scale factor if Laplace prior is used. Ignored if Cauchy prior is used. If, on entry,
    a=NA and prior="laplace", then the scale parameter will also be estimated by
    marginal maximum likelihood. If a is not specified then the default value 0.5
    will be used.
    """
    pr = prior[0:1]
    nx = len(xd)
    if pilo is None:
        pilo = wfromt(torch.sqrt(2 * torch.log(torch.tensor(nx, dtype=torch.float))), prior=prior, a=a)

    if pr == "l":
        beta = beta_laplace(xd, a=a)
    elif pr == "c":
        beta = beta_cauchy(xd)
    
    # Find jump points zj in the derivative of log likelihood as a function of z,
    # and other preliminary calculations
    zs1 = pilo / cs
    zs2 = 1 / cs
    zj =  torch.sort(torch.unique(torch.cat((zs1, zs2)))).values
    cb = cs * beta
    mz = len(zj)
    zlmax = None

    # Find left and right derivatives at each zj and check which are local minima
    lmin = torch.zeros(mz, dtype=torch.bool)
    for j in range(1, mz - 1):
        ze = zj[j]
        cbil = cb[(ze > zs1) & (ze <= zs2)]
        ld = torch.sum(cbil / (1 + ze * cbil))
        if ld <= 0:
            cbir = cb[(ze >= zs1) & (ze < zs2)]
            rd = torch.sum(cbir / (1 + ze * cbir))
            lmin[j] = rd >= 0
    
    # Deal with the two end points in turn, finding right deriv at lower end
    # and left deriv at upper.
    cbir = cb[zj[0] == zs1]
    rd = torch.sum(cbir / (1 + zj[0] * cbir))
    if rd > 0:
        lmin[0] = True
    else:
        zlmax = [zj[0].tolist()]
    
    cbil = cb[zj[mz - 1] == zs2]
    ld = torch.sum(cbil / (1 + zj[mz - 1] * cbil))
    if ld < 0:
        lmin[mz - 1] = True
    else:
        zlmax = [zj[mz - 1].tolist()]

    # Flag all local minima and do a binary search between them to find the local maxima
    zlmin = zj[lmin]
    nlmin = len(zlmin)
    for j in range(1, nlmin):
        zlo = zlmin[j - 1]
        zhi = zlmin[j]
        ze = (zlo + zhi) / 2.0
        zstep = (zhi - zlo) / 2.0
        for nit in range(1,11):
            cbi = cb[(ze >= zs1) & (ze <= zs2)]
            likd = torch.sum(cbi / (1 + ze * cbi))
            zstep /= 2.0
            ze += zstep * torch.sign(likd)
        zlmax.append(ze.item())
    
    # Evaluate all local maxima and find global max;
    # use smaller value if there is an exact tie for the global maximum.
    zlmax = torch.tensor(zlmax)
    zm = torch.full((len(zlmax),), float('nan'))
    for j in range(len(zlmax)):
        pz = torch.clamp(zlmax[j], zs1, zs2)
        zm[j] = torch.sum(torch.log(1 + cb * pz))
    
    zeta_candidates = zlmax[zm == torch.max(zm)]
    zeta = torch.min(zeta_candidates)

    w = torch.clamp(torch.min(torch.tensor([1.0], dtype=torch.float), torch.max(zeta * cs, pilo)), min=0.0, max=1.0)

    return {'zeta': zeta.item(), 'w': w, 'cs': cs, 'pilo': pilo}
