import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling import models, fitting 
from jwst.datamodels import ImageModel
from astropy.visualization import ImageNormalize, ZScaleInterval

def get_extraction_bounds(x1d_file):
    """
    Parameters:
    ------------
    x1d_file: auto pipeline extracted x1d file (.fits)

    Returns:
    ------------
    y1: lower extraction bound (float)
    y2: upper extraction bound (float)

    """
    with fits.open(x1d_file) as a:
        y1 = a['EXTRACT1D'].header['EXTRYSTR'] - 1
        y2 = a['EXTRACT1D'].header['EXTRYSTP'] 

    return y1, y2



def plot_s2d(s2d_file, x1d_file):
    """
    Parameters:
    -------------
    s2d_file: file containing the 2d spec from mast (.fits)
    x1d_file: auto pipeline extracted x1d file (.fits)
    """
    ## open up the s2d file 
    with fits.open(s2d_file) as a:
        s2d = a[1].data

    y1,y2 = get_extraction_bounds(x1d_file)

    ## I took this plotting code from pablos notebook, because it makes a nice 2d specta plot 
    im_model = ImageModel(s2d)
    full_s2d = im_model.data

    fig = plt.figure(figsize=(20, 3))
    ax = fig.add_subplot(111)
    norm = ImageNormalize(full_s2d, interval=ZScaleInterval())
    ax.imshow(full_s2d, cmap='viridis', norm=norm,
            origin='lower', interpolation='None',
            aspect='auto')
    ax.set_xlabel(r'$\mathrm{Spectral\ pixel}$')
    ax.set_ylabel(r'$\mathrm{Spatial\ pixel}$')
    ax.set_title(r'$\mathrm{Full\ 2D\ spectrum}$')
    ax.axhline(y1, lw=0.5, c='r', label='Initial trace-searching window')
    ax.axhline(y2, lw=0.5, c='r')
    plt.legend()
    plt.show()

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
    y1, y2 = get_extraction_bounds(x1d_file) 
    y1_optext = np.mean([y1,y2]) - optext_width /2
    y2_optext = y1_optext + optext_width

    
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
        
        ax1.plot(centroid, color = "black", label = "Centroids")
        ax1.plot(x_vals, g_model(x_vals), color = "red", label = "Gaussian Fit")
        ax1.set_xlabel("Spatial Pixel")

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

    
    