import numpy as np
from load_legac import *


def get_fit_instructions():
    """ Set up the desired fit_instructions dictionary. """

    dust = {}
    dust["type"] = "Salim"
    dust["eta"] = 2.
    dust["Av"] = (0., 8.)
    dust["delta"] = (-0.3, 0.3)
    dust["delta_prior"] = "Gaussian"
    dust["delta_prior_mu"] = 0.
    dust["delta_prior_sigma"] = 0.1
    dust["B"] = (0., 5.)

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
    fit_instructions["noise"] = noise
    fit_instructions["calib"] = calib
    fit_instructions["nebular"] = nebular
    fit_instructions["redshift"] = (0., 10.)
    fit_instructions["t_bc"] = 0.01
    fit_instructions["veldisp"] = (40., 400.)
    fit_instructions["veldisp_prior"] = "log_10"

    return fit_instructions


def analysis_func(fit):
    import matplotlib.pyplot as plt
    fit.posterior.get_advanced_quantities()

    fig = plt.figure(figsize=(12, 5.))
    ax = plt.subplot()

    y_scale = pipes.plotting.add_spectrum(fit.galaxy.spectrum, ax)
    pipes.plotting.add_spectrum_posterior(fit, ax, y_scale=y_scale)

    noise_post = fit.posterior.samples["noise"]*10**-y_scale
    noise_perc = np.percentile(noise_post, (16, 50, 84), axis=0).T
    noise_max = np.max(np.abs(noise_perc))
    noise_perc -= 1.05*noise_max

    ax.plot(fit.galaxy.spectrum[:,0], noise_perc[:, 1], color="darkorange")

    ax.fill_between(fit.galaxy.spectrum[:,0], noise_perc[:, 0],
                    noise_perc[:, 2], color="navajowhite", alpha=0.7)

    ymax = ax.get_ylim()[1]
    ax.set_ylim(-2.1*noise_max, ymax)
    ax.axhline(0., color="gray", zorder=1, lw=1.)
    ax.axhline(-1.05*noise_max, color="gray", zorder=1, lw=1., ls="--")

    plt.savefig("pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_gp.pdf",
                bbox_inches="tight")

    plt.close()


filt_list = np.loadtxt("filters/muzzin_filt_list.txt", dtype="str")

file = np.loadtxt("idmask_fitting_tests.cat")
cols = open("idmask_fitting_tests.cat").readline().split()[1:]

table = pd.DataFrame(file, columns=cols)
table.index = "M" + table["mask"].astype(int).astype(str) + "_" + table["id"].astype(int).astype(str)

fit_instructions = get_fit_instructions()

IDs = table.index
redshifts = np.loadtxt("idmask_fitting_tests.cat", usecols=2)

fit_cat = pipes.fit_catalogue(IDs, fit_instructions, load_legac, run="v12",
                              cat_filt_list=filt_list, vary_filt_list=False,
                              redshifts=redshifts, redshift_sigma=0.025,
                              make_plots=True, time_calls=False,
                              full_catalogue=True, n_posterior=500,
                              analysis_function=analysis_func)

fit_cat.fit(verbose=False)
