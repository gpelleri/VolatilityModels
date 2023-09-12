import scipy as sp
import numpy as np
from scipy.stats import norm


def bs_value(S, K, r, T, q, sigma, option_type):
    """This function calculates the value of the European option based on Black-Scholes-Merton formula
    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param q: dividend yield
    :param sigma: volatility
    :param option_type: call 1  or put 2 option
    :return : option price
    """
    # determine N(d1) and N(d2)
    d1 = 1 / (sigma * np.sqrt(T)) * (np.log(S / K) + (r - q + sigma ** 2 / 2) * T)
    d2 = d1 - sigma * np.sqrt(T)
    # return based on optionType param
    if option_type == 1:
        phi = 1.0
    elif option_type == 2:
        phi = -1.0
    else:
        print('Wrong option type specified')
        return 0

    val = phi * norm.cdf(phi * d1) * S * np.exp(-q * T) - phi * norm.cdf(phi * d2) * K * np.exp(-r * T)
    return val

def bs_vega(S, K, r, T, q, sigma):
    """"
    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param q: dividend yield
    :param sigma: volatility
    :return: partial derivative w.r.t volatility
    """
    d1 = 1 / (sigma * np.sqrt(T)) * (np.log(S / K) + (r - q + sigma ** 2 / 2) * T)
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)

def volatility(sigma, args):
    S = args[0]
    K = args[1]
    r = args[2]
    T = args[3]
    q = args[4]
    opt_type = args[5]
    mkt_val = args[6]
    diff = bs_value(S, K, r, T, q, sigma, opt_type) - mkt_val
    return diff


def fvega(sigma, args):
    S = args[0]
    K = args[1]
    r = args[2]
    T = args[3]
    q = args[4]

    vega = bs_vega(S, K, r, T, q, sigma)
    return vega
def bisection(func, x1, x2, args, xtol=1e-6, maxIter=100):
    """ Based on Dominic O'kane work on FinancePy.
    Bisection algorithm. You need to supply root brackets x1 and x2. """

    if np.abs(x1-x2) < 1e-10:
        raise Exception("Brackets should not be equal")

    if x1 > x2:
        raise Exception("Bracket x2 should be greater than x1")

    f1 = func(x1, args)
    fmid = func(x2, args)

    if np.abs(f1) < xtol:
        return x1
    elif np.abs(fmid) < xtol:
        return x2

    if f1 * fmid >= 0:
        print("Root not bracketed")
        return None

    for i in range(0, maxIter):

        xmid = (x1 + x2)/2.0
        fmid = func(xmid, args)

        if f1 * fmid < 0:
            x2 = xmid
        else:
            x1 = xmid

        if np.abs(fmid) < xtol:
            return xmid

    print("Bisection exceeded number of iterations", maxIter)
    return None

def implied_volatility_row(row):
    """
    Extracts Implied_Volatilty using a dataframe row as parameter, in order to increase performance.
    N.B. This function allows to compute all implied vol at once by using df.apply(implied_vol_row)
    instead of having to iterate over the DF
    """

    # Extract values from row
    opt_val = row['lastPrice']
    spot = row['Spot']
    strike = row['strike']
    expiry = row['Expiry']
    rf = row['Risk-Free Rate']
    div = row['Dividend']
    if row['optionType'] == "call":
        option_type = 1
    else:
        option_type = 2
    #option_type = row['optionType']

    # sigma initial value
    sig_start = np.sqrt(2 * np.pi / expiry) * opt_val / spot
    arglist = (spot, strike, rf, expiry, div, option_type, opt_val)
    argsv = np.array(arglist)

    sigma = bisection(volatility, 1e-5, 10.0, argsv, xtol=0.00001)

    return sigma