import numpy as np 
from astropy.io import fits
import astropy.units as u 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_1d_spec(fits_file):
    """
    load in a jwst 1d spec file. this assumes that the data is in order of: wv, flux, err, which 
    is typical for a jwst 1d spec, and how the files are generated from MAST.
    
    Parameters
    -------
    fits_file: a jwst spec file 

    Returns
    -------
    wv: wavelength array
    flux: flux array
    flux_err: flux error array 

    """
    ## open fits file
    a = fits.open(fits_file)
    
    wv = a[1].data[a[1].header["TTYPE1"]]
    wv_unit = u.Unit(a[1].header["TUNIT1"])
    wv = wv * wv_unit

    fnu = a[1].data[a[1].header["TTYPE2"]]
    fnu_unit = u.Unit(a[1].header["TUNIT2"])

    fnu_err = a[1].data[a[1].header["TTYPE3"]]
    fnu_err_unit = u.Unit(a[1].header["TUNIT3"])

    ## convert to cgs units, from Jy or microJy to erg/s/cm^2/Hz
    try: 
        fnu = (fnu * fnu_unit).to(
            u.erg / u.s / u.cm**2 / u.Hz
        )
        fnu_err = (fnu_err * fnu_err_unit).to(
            u.erg / u.s / u.cm**2 / u.Hz
        )

    except u.UnitConversionError:
        raise ValueError(
            f"Cannot convert {fnu_unit} to flux density"
        )

    flam = fnu.to(
    u.erg / u.s / u.cm**2 / u.AA,
    equivalencies=u.spectral_density(wv)
    )
    
    flam_err = fnu_err.to(
        u.erg / u.s / u.cm**2 / u.AA,
        equivalencies=u.spectral_density(wv)
    )
    
    return wv, fnu, fnu_err, flam, flam_err


def write_to_fits(wv_total, flux_total, err_total, file_235, file_395, source_id):
    """
    Parameters:
    wv_total: full combind wv array, microns
    flux_total: full combined flux array, units of fnu
    err_total: full combined err array, units of fnu
    file_395: fits file, need for header information

    Returns:
    hdul_out: fits file with the combined grating data 
    """

    ## grab header information from 395grat file 
    with fits.open(file_395) as hdul_395:
        primary_header = hdul_395[0].header.copy()
        ext_header = hdul_395[1].header.copy() if len(hdul_395) > 1 else None

        ## grab column names and unit information so we are consistent with our naming scheme
        wv_name = hdul_395[1].header["TTYPE1"]
        wv_unit = u.Unit(hdul_395[1].header["TUNIT1"])
    
        fnu_name = hdul_395[1].header["TTYPE2"]
        fnu_unit = u.Unit(hdul_395[1].header["TUNIT2"])
    
        fnu_err_name = hdul_395[1].header["TTYPE3"]
        fnu_err_unit = u.Unit(hdul_395[1].header["TUNIT3"])

    ## update header information to remove extraneous information regarding the 395 file
    primary_header["FILENAME"] = f's{source_id}_x1d_combgrat.fits'
    
    delete_list = ['ASNTABLE', 'VISITGRP', 'FILTER', "GRATING", 'EXPCOUNT', 'EXPRIPAR', 'EXP_TYPE', 'EXPSTART', 'EXPMID', 'EXPEND',
    'OSF_FILE', 'READPATT', 'HYBRIDFF', 'NOUTPUTS', 'NINTS', 'NGROUPS', 'NFRAMES', 'FRMDIVSR', 'GROUPGAP', 'DRPFRMS1', 'DRPFRMS3', 'NSAMPLES',
    'TSAMPLE', 'TFRAME', 'TGROUP', 'EFFINTTM', 'EFFEXPTM', 'TMEASURE', 'DURATION', 'NRSTSTRT', 'NRESETS', 'ZEROFRAM', 'DATAPROB', 'SCA_NUM',
    'DATAMODE', 'NRS_NORM', 'NRS_REF', 'SCTARATE', 'GAINFACT', 'IS_IMPRT', 'PRIMECRS', 'EXTNCRS', 'SUBARRAY', 'SUBSTRT1', 'SUBSTRT2', 'SUBSIZE1',          
    'SUBSIZE2', 'FASTAXIS', 'SLOWAXIS']
    
    for metadata in delete_list:
        del primary_header[metadata]

    ## remove card names for exposure parameters and subarray parameters
    for i, card in enumerate(primary_header.cards):
        if 'Exposure parameters' in str(card):
            del primary_header[i]
        if 'Subarray parameters' in str(card):
            del primary_header[i]
        
    ## rename extract1d extension name for the combined grat 
    ext_header['EXTNAME'] = 'COMBGRAT'
        
    ## convert flux units back to Jy
    flux_total = flux_total * u.erg / u.s / u.cm**2 / u.Hz
    flux_total = flux_total.to(u.Jy)
    err_total = err_total * u.erg / u.s / u.cm**2 / u.Hz
    err_total = err_total.to(u.Jy)
        
    ## build columns of data for the fits file
    col_wv   = fits.Column(name= wv_name, format='D', array=wv_total, unit= str(wv_unit))
    col_flux = fits.Column(name= fnu_name, format='D', array=flux_total, unit = str(fnu_unit))
    col_err  = fits.Column(name= fnu_err_name, format='D', array=err_total, unit = str(fnu_err_unit))

    ## create the table HDU 
    table_hdu = fits.BinTableHDU.from_columns(
        [col_wv, col_flux, col_err],
        header=ext_header
    )
    table_hdu.header['COMMENT'] = 'Combined G235 + G395 spectrum'

    # Primary HDU keeps the 395 primary header and retains the metadata of the source
    primary_hdu = fits.PrimaryHDU(header=primary_header)

    hdul_out = fits.HDUList([primary_hdu, table_hdu])

    ## append original file_235 and file_395 extensions
    for label, fname in [('G235', file_235), ('G395', file_395)]:
        with fits.open(fname) as hdul_orig:
            for i, hdu in enumerate(hdul_orig):
                hdu_copy = hdu.copy()
                # renaming 
                orig_name = hdu.header.get('EXTNAME', f'HDU{i}')
                hdu_copy.header['EXTNAME'] = f'{label}_{orig_name}'
                hdu_copy.header['EXTVER'] = 1
                hdul_out.append(hdu_copy)

    #print(hdul_out)

    return hdul_out


def combine_filters(file_235, file_395, source_id, *, plot_spec = False):
    """
    Parameters:
    --------
    file_235: fits file of 235 grating data 
    file_395 fits file of 395 grating data 
    source_name: nirspec id, used to name final fits file
    plot_spec: boolean, plots original data + combined spec if set to True

    Returns:
    --------
    comb_file: combined fits file of grating data, retains header information of the reddest filter 
    
    """
    ## load in data files 
    wv_235, fnu_235, fnu_err_235, flam_235, flam_err_235 = load_1d_spec(file_235)
    wv_395, fnu_395, fnu_err_395, flam_395, flam_err_395 = load_1d_spec(file_395)

    ## removing units 
    wv_235, fnu_235, fnu_err_235, flam_235, flam_err_235 = wv_235.value, fnu_235.value, fnu_err_235.value, flam_235.value, flam_err_235.value
    wv_395, fnu_395, fnu_err_395, flam_395, flam_err_395 = wv_395.value, fnu_395.value, fnu_err_395.value, flam_395.value, flam_err_395.value


    ## find overlap between 235 and 395 wavelengths
    min_wv_235, max_wv_235 = np.nanmin(wv_235), np.nanmax(wv_235)
    min_wv_395, max_wv_395 = np.nanmin(wv_395), np.nanmax(wv_395)

    ## minumum and maxium wavelenghts of the overlapping region
    overlapping_min = max(min_wv_235, min_wv_395)
    overlapping_max = min(max_wv_235, max_wv_395)

    ## initialize median_diff to nan, incase there is no overlaps between regions
    median_diff = np.nan
    median_diff_err = np.nan

    ## case 1: the files don't overlap at all, can simply concatenate
    if (overlapping_min > overlapping_max):
        wv_total = np.concatenate((wv_235, wv_395))
        flux_total = np.concatenate((fnu_235, fnu_395))
        err_total = np.concatenate((fnu_err_235, fnu_err_395))
        comb_fits = write_to_fits(wv_total, flux_total, err_total, file_235, file_395, source_id)

    ## case 2: wavelength overlap of the files, need to interpolate the overlapping region onto the redder wv grid
    if (overlapping_min < overlapping_max):
        mask_235 = (wv_235 >= overlapping_min) & (wv_235 <= overlapping_max)
        mask_395 = (wv_395 >= overlapping_min) & (wv_395 <= overlapping_max)

        ## get the regions of the files that have no overlap, because those will stay the same
        wv_235_no_overlap = wv_235[~mask_235]
        flux_235_no_overlap = fnu_235[~mask_235]
        err_235_no_overlap = fnu_err_235[~mask_235]

        wv_395_no_overlap = wv_395[~mask_395]
        flux_395_no_overlap = fnu_395[~mask_395]
        err_395_no_overlap = fnu_err_395[~mask_395]

        ## now get the regions of the files that do overlap
        wv_235_overlap = wv_235[mask_235]
        flux_235_overlap = fnu_235[mask_235]
        err_235_overlap = fnu_err_235[mask_235]

        wv_395_overlap = wv_395[mask_395]
        flux_395_overlap = fnu_395[mask_395]
        err_395_overlap = fnu_err_395[mask_395]

        ## in overlapping region, interpolate 235 data onto 395 wavelength grid 
        interp235 = interp1d(wv_235_overlap, flux_235_overlap, bounds_error=False, fill_value=np.nan)
        interp235_err = interp1d(wv_235_overlap, err_235_overlap, bounds_error=False, fill_value=np.nan)

        flux_235_on_395 = interp235(wv_395_overlap)
        err_235_on_395 = interp235_err(wv_395_overlap)

        flux_avg_235_395 = np.zeros(len(flux_235_on_395))
        err_avg_235_395 = np.zeros(len(flux_235_on_395))
        
        ## take median in overlapping region of both filters 
        median_235 = np.nanmedian(flux_235_overlap)
        median_395 = np.nanmedian(flux_395_overlap)
        median_diff = median_235 - median_395

        ## standard error on the median 
        median_235_err = 1.2533 *(np.nanstd(flux_235_overlap) / np.sqrt(len(flux_235_overlap)))
        median_395_err = 1.2533 *(np.nanstd(flux_395_overlap) / np.sqrt(len(flux_395_overlap)))

        ## error on the difference of the medians 
        median_diff_err = np.sqrt(median_235_err**2 + median_395_err**2)

        for i in range(len(flux_235_on_395)):
            if np.isnan(flux_235_on_395[i]):
                ## 235 interpolator has no coverage here — use 395 value alone, this covers edge cases 
                flux_avg_235_395[i] = flux_395_overlap[i]
                err_avg_235_395[i] = err_395_overlap[i]
                continue

            if np.isnan(flux_395_overlap[i]):
                flux_avg_235_395[i] = flux_235_on_395[i]
                err_avg_235_395[i] = err_235_on_395[i]
                continue

            ## weighted mean of the two spectra at every overlapping pixel
            avg = (flux_235_on_395[i]/err_235_on_395[i]**2) + (flux_395_overlap[i]/err_395_overlap[i]**2)
            avg = avg / ((1/err_235_on_395[i]**2) + (1/err_395_overlap[i]**2))
            flux_avg_235_395[i] = avg

            ## err on the weighted mean 
            err_avg_235_395[i] = 1 / np.sqrt((1/err_235_on_395[i]**2) + (1/err_395_overlap[i]**2))

        ## now put all the arrays together!
        wv_total = np.concatenate((wv_235_no_overlap,  wv_395_overlap, wv_395_no_overlap))
        flux_total = np.concatenate((flux_235_no_overlap, flux_avg_235_395, flux_395_no_overlap))
        err_total = np.concatenate((err_235_no_overlap, err_avg_235_395, err_395_no_overlap))

        # Sort everything by wavelength
        sort_idx = np.argsort(wv_total)
    
        wv_total = wv_total[sort_idx]
        flux_total = flux_total[sort_idx]
        err_total = err_total[sort_idx]
        
        comb_fits = write_to_fits(wv_total, flux_total, err_total, file_235, file_395, source_id)
    
    if plot_spec == True:
        ## plot before and after combining the red different gratings 
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot(wv_235, fnu_235 * 1e23, color='blue', label='G235') ## 1e23 factor coverting from erg/s/cm2/hz back to Jy
        ax1.plot(wv_395, fnu_395 * 1e23, color='red', label='G395')
        ax1.axvspan(overlapping_min, overlapping_max, color='gray', alpha=0.2, label = "overlapping region")
        ax1.set_ylabel(r'$f_\nu$', fontsize = 14)
        ax1.legend()

        ## combined spectrum
        ax2.plot(wv_total, flux_total * 1e23, color='darkviolet', label='combined')
        ax2.fill_between(wv_total, flux_total * 1e23 - err_total * 1e23, flux_total * 1e23 + err_total * 1e23,
                          color='thistle')
        ax2.axvspan(overlapping_min, overlapping_max, color='gray', alpha=0.2)
        ax2.set_xlabel(r'Wavelength [$\mu$m]', fontsize = 14)
        ax2.set_ylabel(r'$f_\nu$', fontsize = 14)
        ax2.legend()

        plt.title(source_id)
        plt.tight_layout()
        #plt.xlim(2.5,3.0)
        plt.show()

    return comb_fits, median_diff, median_diff_err