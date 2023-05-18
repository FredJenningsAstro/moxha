from soxs.utils import mylog, parse_prng, parse_value, \
    get_rot_mat, create_region, get_data_file, ensure_numpy_array
import warnings
from soxs.instrument import make_background
from soxs.events import write_event_file


def make_background_file_with_shift(out_file, exp_time, instrument, sky_center,
                         overwrite=False, foreground=True, instr_bkgnd=True,
                         ptsrc_bkgnd=True, no_dither=False, dither_params=None,
                         subpixel_res=False, input_pt_sources=None, 
                         prng=None,aimpt_shift = None, **kwargs):
    """
    Make an event file consisting entirely of background events. This will be 
    useful for creating backgrounds that can be added to simulations of sources.

    Parameters
    ----------
    exp_time : float, (value, unit) tuple, or :class:`~astropy.units.Quantity`
        The exposure time to use, in seconds. 
    instrument : string
        The name of the instrument to use, which picks an instrument
        specification from the instrument registry. 
    sky_center : array, tuple, or list
        The center RA, Dec coordinates of the observation, in degrees.
    overwrite : boolean, optional
        Whether or not to overwrite an existing file with the same name.
        Default: False
    foreground : boolean, optional
        Whether or not to include the Galactic foreground. Default: True
    instr_bkgnd : boolean, optional
        Whether or not to include the instrumental background. Default: True
    ptsrc_bkgnd : boolean, optional
        Whether or not to include the point-source background. Default: True
    no_dither : boolean, optional
        If True, turn off dithering entirely. Default: False
    dither_params : array-like of floats, optional
        The parameters to use to control the size and period of the dither
        pattern. The first two numbers are the dither amplitude in x and y
        detector coordinates in arcseconds, and the second two numbers are
        the dither period in x and y detector coordinates in seconds. 
        Default: [8.0, 8.0, 1000.0, 707.0].
    subpixel_res: boolean, optional
        If True, event positions are not randomized within the pixels 
        within which they are detected. Default: False
    input_pt_sources : string, optional
        If set to a filename, input the point source positions, fluxes,
        and spectral indices from an ASCII table instead of generating
        them. Default: None
    prng : :class:`~numpy.random.RandomState` object, integer, or None
        A pseudo-random number generator. Typically will only 
        be specified if you have a reason to generate the same 
        set of random numbers, such as for a test. Default is None, 
        which sets the seed based on the system time. 
    """
    if "nH" in kwargs or "absorb_model" in kwargs:
        warnings.warn("The 'nH' and 'absorb_model' keyword arguments"
                      "have been omitted. Please set the 'bkgnd_nH' "
                      "and 'bkgnd_absorb_model' values in the SOXS"
                      "configuration file if you want to change these "
                      "values. ",
                      DeprecationWarning)
    if "input_sources" in kwargs:
        warnings.warn("The 'input_sources' keyword argument has been changed "
                      "to 'input_pt_sources' and is deprecated.", 
                      DeprecationWarning)
        input_pt_sources = kwargs.pop("input_sources")
    prng = parse_prng(prng)
    events, event_params = make_background(exp_time, instrument, sky_center, 
                                           ptsrc_bkgnd=ptsrc_bkgnd, 
                                           foreground=foreground, 
                                           instr_bkgnd=instr_bkgnd,
                                           no_dither=no_dither,
                                           dither_params=dither_params, 
                                           subpixel_res=subpixel_res,
                                           input_pt_sources=input_pt_sources,
                                           prng=prng, aimpt_shift = aimpt_shift)
    write_event_file(events, event_params, out_file, overwrite=overwrite)

    
    
def soxs_plotter(img_file,ax, hdu="IMAGE", stretch='linear', vmin=None, vmax=None,
               facecolor='black', center=None, width=None, figsize=(10, 10),
               cmap=None, plot_cbar = False, reblock = 1, do_contour = False):
    
    from astropy.io import fits
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy import wcs
    from scipy.ndimage import gaussian_filter 

    from matplotlib.colors import PowerNorm, LogNorm, Normalize
    from astropy.wcs.utils import proj_plane_pixel_scales
    from astropy.visualization.wcsaxes import WCSAxes
    if stretch == "linear":
        norm = Normalize(vmin=vmin, vmax=vmax)
    elif stretch == "log":
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif stretch == "sqrt":
        norm = PowerNorm(0.5, vmin=vmin, vmax=vmax)
    else:
        raise RuntimeError(f"'{stretch}' is not a valid stretch!")
    with fits.open(img_file) as f:
        hdu = f[hdu]
        w = wcs.WCS(hdu.header)
        pix_scale = proj_plane_pixel_scales(w)
        if center is None:
            center = w.wcs.crpix
        else:
            center = w.wcs_world2pix(center[0], center[1], 0)
        if width is None:
            dx_pix = 0.5*hdu.shape[0]
            dy_pix = 0.5*hdu.shape[1]
        else:
            dx_pix = width / pix_scale[0]
            dy_pix = width / pix_scale[1]
        print(hdu.data)
        im = ax.imshow(hdu.data/reblock**2, norm=norm, cmap=cmap)
        ax.set_xlim(center[0] - 0.5*dx_pix, center[0] + 0.5*dx_pix)
        ax.set_ylim(center[1] - 0.5*dy_pix, center[1] + 0.5*dy_pix)
        ax.set_facecolor(facecolor)
        ax.set_xticks([])
        ax.set_yticks([])
        if do_contour:
            contour_data = (hdu.data/reblock**2)
            # contour_data[np.where(contour_data ==0)] = 1000
            P = ax.contour(contour_data, levels=(np.array([40,50,80,])), colors='white', alpha=0.25, norm = norm, antialiased=True)
            for level in P.collections:
                for kp,path in enumerate(level.get_paths()):
                    # include test for "smallness" of your choice here:
                    # I'm using a simple estimation for the diameter based on the
                    #    x and y diameter...
                    verts = path.vertices # (N,2)-shape array of contour line coordinates
                    diameter = np.max(verts.max(axis=0) - verts.min(axis=0))

                    if diameter < 40: # threshold to be refined for your actual dimensions!
                        #print 'diameter is ', diameter
                        del(level.get_paths()[kp])  # no remove() for Path objects:(
                        #level.remove() # This does not work. produces V

    return im