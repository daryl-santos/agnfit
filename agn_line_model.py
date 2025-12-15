# Original code from Jinyi Shangguan

import numpy as np
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from astropy.modeling.core import Fittable1DModel
from astropy.modeling.parameters import Parameter
from scipy.optimize import minimize
import extinction

ls_km = 2.99792458e5  # km/s

__all__ = ['Line_Gaussian', 'line_fwhm', 'find_line_peak']

class Line_Gaussian(Fittable1DModel):
    '''
    The Gaussian line profile with the sigma as the velocity.

    Parameters
    ----------
    x : array like
        Wavelength, units: arbitrary.
    amplitude : float
        The amplitude of the line profile.
    dv : float
        The velocity of the central line offset from wavec, units: km/s.
    sigma : float
        The velocity dispersion of the line profile, units: km/s.
    wavec : float
        The central wavelength of the line profile, units: same as x.
    '''

    amplitude = Parameter(default=1, bounds=(0, None))
    dv = Parameter(default=0, bounds=(-2000, 2000))
    sigma = Parameter(default=200, bounds=(20, 10000))

    wavec = Parameter(default=5000, fixed=True)

    @staticmethod
    def evaluate(x, amplitude, dv, sigma, wavec):
        """
        Gaussian model function.
        """
        v = (x - wavec) / wavec * ls_km  # convert to velocity (km/s)
        f = amplitude * np.exp(-0.5 * ((v - dv)/ sigma)**2)

        return f


def line_fwhm(model, x0, x1, x0_limit=None, x1_limit=None, fwhm_disp=None):
    '''
    Calculate the FWHM of the line profile.

    Parameters
    ----------
    model : Astropy model
        The model of the line profile. It should be all positive.
    x0, x1 : float
        The initial guesses of the wavelengths on the left and right sides.
    x0_limit, x1_limit (optional) : floats
        The left and right boundaries of the search.
    fwhm_disp (optional) : float
        The instrumental dispersion that should be removed from the FWHM, units
        following the wavelength.

    Returns
    -------
    fwhm : float
        The FWHM of the line, units: km/s.
    w_l, w_r : floats
        The wavelength and flux of the peak of the line profile.
    w_peak : float
        The wavelength of the line peak.
    '''
    xc = (x0 + x1) / 2
    w_peak, f_peak = find_line_peak(model, xc)
    f_half = f_peak / 2

    func = lambda x: np.abs(model(x) - f_half)

    if x0_limit is not None:
        bounds = ((x0_limit, w_peak),)
    else:
        bounds = None
    res_l = minimize(func, x0=x0, bounds=bounds)

    if x1_limit is not None:
        bounds = ((w_peak, x1_limit),)
    else:
        bounds = None
    res_r = minimize(func, x0=x1, bounds=bounds)
    w_l = res_l.x[0]
    w_r = res_r.x[0]

    fwhm_w = (w_r - w_l)
    if fwhm_disp is not None:  # Correct for instrumental dispersion
        fwhm_w = np.sqrt(fwhm_w**2 - fwhm_disp**2)

    fwhm = fwhm_w / w_peak * ls_km
    return fwhm, w_l, w_r, w_peak



def find_line_peak(model, x0):
    '''
    Find the peak wavelength and flux of the model line profile.

    Parameters
    ----------
    model : Astropy model
        The model of the line profile. It should be all positive.
    x0 : float
        The initial guess of the wavelength.

    Returns
    -------
    w_peak, f_peak : floats
        The wavelength and flux of the peak of the line profile.
    '''
    func = lambda x: -1 * model(x)
    res = minimize(func, x0=x0)
    w_peak = res.x[0]
    try:
        f_peak = model(w_peak)[0]
    except:
        f_peak = model(w_peak)
    return w_peak, f_peak
