import numpy as np
from load_legac import *


def get_fit_instructions():
    """ Set up the desired fit_instructions dictionary. """

    dust = {}
    dust["type"] = "CF00"
    dust["eta"] = 2.
    dust["Av"] = (0., 8.0)
    dust["n"] = (0.3, 1.5)
    dust["n_prior"] = "Gaussian"
    dust["n_prior_mu"] = 0.7
    dust["n_prior_sigma"] = 0.3

    nebular = {}
    nebular["logU"] = -3.

    dblplaw = {}
    dblplaw["massformed"] = (0., 13.)
    dblplaw["metallicity"] = (0.01, 2.5)
    dblplaw["metallicity_prior"] = "log_10"
    dblplaw["alpha"] = (0.01, 1000.)
    dblplaw["alpha_prior"] = "log_10"
    dblplaw["beta"] = (0.01, 1000.)
    dblplaw["beta_prior"] = "log_10"
    dblplaw["tau"] = (0.1, 15.)

    noise = {}
    noise["type"] = "GP_exp_squared"
    noise["scaling"] = (0.1, 10.)
    noise["scaling_prior"] = "log_10"
    noise["norm"] = (0.0001, 1.)
    noise["norm_prior"] = "log_10"
    noise["length"] = (0.01, 1.)
    noise["length_prior"] = "log_10"

    calib = {}
    calib["type"] = "polynomial_bayesian"

    calib["0"] = (0.5, 1.5)
    calib["0_prior"] = "Gaussian"
    calib["0_prior_mu"] = 1.
    calib["0_prior_sigma"] = 0.25

    calib["1"] = (-0.5, 0.5)
    calib["1_prior"] = "Gaussian"
    calib["1_prior_mu"] = 0.
    calib["1_prior_sigma"] = 0.25

    calib["2"] = (-0.5, 0.5)
    calib["2_prior"] = "Gaussian"
    calib["2_prior_mu"] = 0.
    calib["2_prior_sigma"] = 0.25

    fit_instructions = {}
    fit_instructions["dust"] = dust
    fit_instructions["dblplaw"] = dblplaw
    #fit_instructions["noise"] = noise
    #fit_instructions["calib"] = calib
    fit_instructions["nebular"] = nebular
    fit_instructions["redshift"] = (0., 10.)
    fit_instructions["t_bc"] = 0.01
    #fit_instructions["veldisp"] = (40., 400.)
    #fit_instructions["veldisp_prior"] = "log_10"

    return fit_instructions


filt_list = np.loadtxt("filters/muzzin_filt_list.txt", dtype="str")

file = np.loadtxt("idmask_fitting_tests.cat")
cols = open("idmask_fitting_tests.cat").readline().split()[1:]

table = pd.DataFrame(file, columns=cols)
table.index = "M" + table["mask"].astype(int).astype(str) + "_" + table["id"].astype(int).astype(str)

fit_instructions = get_fit_instructions()

IDs = table["id"].astype(int).astype(str)#table.index
redshifts = np.loadtxt("idmask_fitting_tests.cat", usecols=2)

fit_cat = pipes.fit_catalogue(IDs, fit_instructions, load_legac_phot, run="phot",
                              cat_filt_list=filt_list, vary_filt_list=False,
                              redshifts=redshifts, redshift_sigma=0.025,
                              make_plots=False, time_calls=True,
                              spectrum_exists=False)

fit_cat.fit(verbose=True)
