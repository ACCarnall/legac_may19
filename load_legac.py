import numpy as np
import bagpipes as pipes
from astropy.io import fits
from astropy.table import Table
from glob import glob
import pandas as pd

bands = ["IB427", "IB464", "IA484", "IB505", "IA527", "IB574", "IA624", "IA679", "IB709", "IA738", "IA767", "IB827",
         "u", "B", "V", "gp", "rp", "ip", "zp", "Y", "J", "H", "Ks", "ch1", "ch2"]


def load_legac(ID):
    spectrum = load_legac_spec(ID)
    photometry = load_legac_phot(ID.split("_")[1])

    return spectrum, photometry


def load_legac_spec(ID):

    spec_name = "spec1d/legac_" + ID.split("_")[0] + "_v3.6_spec1d_" + ID.split("_")[1] + ".fits"
    err_name = "spec1d/legac_" + ID.split("_")[0] + "_v3.6_wht1d_" + ID.split("_")[1] + ".fits"

    fluxes = fits.open(spec_name)[0].data
    invfluxvars = fits.open(err_name)[0].data
    fluxerrs = np.sqrt(1./invfluxvars)
    fluxerrs[np.isinf(fluxerrs)] = 0.

    wav0 = fits.open(spec_name)[0].header["CRVAL1"]
    dwav = 0.6
    wavs = np.arange(wav0, wav0+0.6*fluxes.shape[0], dwav)

    spectrum = np.c_[wavs, fluxes*10**-19, fluxerrs*10**-19]

    wave_mask = (spectrum[:, 0] > 6300.) & (spectrum[:, 0] < 8800.)
    spectrum = spectrum[wave_mask, :]

    spectrum = bin(spectrum, 2)

    # Get rid of any points with zeros in the error spectrum (hack)
    spectrum[(spectrum[:, 2] == 0.), 1] = 0.
    spectrum[(spectrum[:, 2] == 0.), 2] = 9.9999*10**99
    return spectrum


def load_legac_phot(ID):

    file = np.loadtxt("UVISTA_final_v4.1_fitting_tests.cat")
    cols = open("UVISTA_final_v4.1_fitting_tests.cat").readline().split()[1:]

    table = pd.DataFrame(file, columns=cols)
    table.index = table["id"]

    fluxes = [table.loc[int(ID), b] for b in bands]
    fluxerrs = [table.loc[int(ID), "e" + b] for b in bands]

    apcorr = table.loc[int(ID), "Ks_tot"]/table.loc[int(ID), "Ks"]

    photometry = np.c_[fluxes, fluxerrs]*10**(-1.1/2.5)*apcorr

    snrs = photometry[:, 0]/photometry[:, 1]

    # Limit SNR to 20 sigma in each band
    for i in range(len(photometry)):
        if np.abs(photometry[i,0]/photometry[i,1]) > 20.:
            photometry[i,1] = np.abs(photometry[i,0]/20.)

    # Limit SNR of IRAC1 and IRAC2 channels to 10 sigma.
    for i in range(1,3):
        if np.abs(photometry[-i,0]/photometry[-i,1]) > 10.:
            photometry[-i,1] = np.abs(photometry[-i,0]/10.)

    # blow up the errors associated with any N/A photometry points.
    for i in range(len(photometry)):
        if photometry[i,0] == 0. or photometry[i,1] <= 0:
            photometry[i,0] = 0.
            photometry[i,1] = 9.9*10**99.

    return photometry


def bin(spectrum, binn):
    """ Bins up a two or three column spectrum by a given factor. """

    binn = int(binn)
    nbins = int(len(spectrum)/binn)
    binspec = np.zeros((nbins, spectrum.shape[1]))
    for i in range(binspec.shape[0]):
        binspec[i, 0] = np.mean(spectrum[i*binn:(i+1)*binn, 0])
        binspec[i, 1] = np.mean(spectrum[i*binn:(i+1)*binn, 1])
        if spectrum.shape[1] == 3:
            sq_sum = np.sum(spectrum[i*binn:(i+1)*binn, 2]**2)
            binspec[i,2] = (1./float(binn))*np.sqrt(sq_sum)

    return binspec
