import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling import models, fitting 


def plot_s2d(s2d_file, *, sig = 3):
    """
    Input: s2d file path (.fits)
    
    """
    a = fits.open(s2d_file) ## open up file
    s2d = a[1].data

    plt.figure(figsize = [8,3])
    vmin = np.nanmedian(s2d) - (sig * np.nanstd(s2d))
    vmax = np.nanmedian(s2d) + (sig * np.nanstd(s2d))
    plt.imshow(s2d, vmin=vmin, vmax=vmax,
            origin = "lower", 
            aspect = "auto")

def gaussian_model(centroid, y1_optext, y2_optext, optext_width):

    ## simple Gaussian profile fitted to the centroid profile, so the weights are smooth
    amp = np.max(centroid)
    sigma = optext_width / 2
    mean = (y1_optext + y2_optext) / 2

    x_vals = np.arange(0, len(centroid))

    ## initialize fitter 
    fitter = fitting.LevMarLSQFitter()

    g_init = models.Gaussian1D(amplitude = amp, mean=mean, stddev = sigma)
    g_model = fitter(g_init, x_vals, centroid)

    return x_vals, g_model


def optimal_extract_1d(s2d_file, x1d_file, ax1, ax2, *, optext_width = 8):
    ## grab the expected location of the source using the x1d file 
    with fits.open(x1d_file) as a:
        ## pipeline indexing is 1 based instead of 0 based 
        y1 = a['EXTRACT1D'].header['EXTRYSTR'] - 1
        y2 = a['EXTRACT1D'].header['EXTRYSTP'] 
        y1_optext = np.mean([y1,y2]) - optext_width /2
        y2_optext = y1_optext + optext_width

        print(y1_optext, y2_optext)

    
    with fits.open(s2d_file) as a:
        s2d = a[1].data
        s2d_err = a[2].data
        s2d_wv = a[3].data 

        ## check errors for 0s and set to nan 
        s2d_err[s2d_err == 0] = np.nan

        spatial_profile = np.nanmedian(s2d, axis = 1)

        ## check if spatial profile is nan and replace with negative number 
        where = np.isnan(spatial_profile)
        spatial_profile[where] = 1

        ## now mask out the negative numbers
        spatial_profile[spatial_profile < 0] = 0

        ## now use the y1_optext and y2_optext we definied earlier to grab the central trace
        mask = np.zeros_like(spatial_profile)

        ## y1_optext and y2_optext need to be integers to slice our mask on 
        mask[int(y1_optext):int(y2_optext)] = 1

        ## apply mask 
        centroid = spatial_profile * mask

        x_vals, g_model = gaussian_model(centroid, y1_optext, y2_optext, optext_width)
        
        ax1.plot(centroid)
        ax1.plot(x_vals, g_model(x_vals))

        ## normalize the profile to 1
        centroid_weights = g_model(x_vals) / np.sum(g_model(x_vals))

        ## extract x1d flux and error spectrum
        centroid_weights = np.expand_dims(centroid_weights, axis=1)
        flux_1d = np.nansum(centroid_weights*s2d/s2d_err**2, axis = 0)/np.nansum(centroid_weights**2/s2d_err**2, axis = 0)
        err_1d = np.sqrt(1./np.nansum(centroid_weights**2/s2d_err**2, axis = 0))

        ## flux and err are currently in units of MJy, lets change to Jy 
        flux_1d = flux_1d * 1e6
        err_1d = err_1d * 1e6
        wv = s2d_wv[0]

        ax2.plot(wv, flux_1d, color = "black",
                 drawstyle = "steps-mid",
                 label = "optimal extract")
        ax2.fill_between(wv, flux_1d - err_1d, flux_1d + err_1d, color = "gray",
                         step = "mid")

        plt.legend()
        plt.show()

    
    