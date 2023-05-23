import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea
from astropy.io import fits
import numpy as np
from astropy import wcs
from scipy.ndimage import gaussian_filter 



def soxs_plotter(img_file,ax, hdu="IMAGE", stretch='linear', vmin=None, vmax=None, plot_calibration=False,
               facecolor='black', center=None, width=None, figsize=(10, 10),
               cmap=None, plot_cbar = False, reblock = 1, do_contour = False, cont_levels=[40,50,80,], 
               smooth = False, smooth_sigma = 2, unsharp_sigmas = None, unsharp_method = None):


    from matplotlib.colors import PowerNorm, LogNorm, Normalize
    from astropy.wcs.utils import proj_plane_pixel_scales
    from astropy.visualization.wcsaxes import WCSAxes
    
    with fits.open(img_file) as f:
        hdu = f[hdu]
        print(hdu.header)
        w = wcs.WCS(hdu.header)
        pix_scale = proj_plane_pixel_scales(w)
        # print("pix sclae", pix_scale)
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
            
        if unsharp_sigmas != None:
            if unsharp_method == "divide":
                data = gaussian_filter(hdu.data/reblock**2 , sigma = max(unsharp_sigmas)) 
                data[data<0] = 0
                data[data>0] /= gaussian_filter(hdu.data[data>0]/reblock**2 , sigma = min(unsharp_sigmas))
                data[data>0] = 1/data[data>0]
            if unsharp_method == "subtract":
                data = gaussian_filter(hdu.data/reblock**2 , sigma = max(unsharp_sigmas)) - gaussian_filter(hdu.data/reblock**2 , sigma = min(unsharp_sigmas))
                data[data<0] = 0
        else:
            data = hdu.data/reblock**2
        
        
        
        
        if smooth:
            data = gaussian_filter(data , sigma = smooth_sigma)



        if vmin == None:
            vmin = 0.25*np.percentile(data[data>0].flatten(), 25)    
        if vmax == None:
            vmax = np.percentile(data[data>0].flatten(), 95)   
        print(f"vmin: {vmin}")
        print(f"vmax: {vmax}")
        if stretch == "linear":
            norm = Normalize(vmin=vmin, vmax=vmax)
        elif stretch == "log":
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif stretch == "sqrt":
            norm = PowerNorm(0.5, vmin=vmin, vmax=vmax)
        else:
            raise RuntimeError(f"'{stretch}' is not a valid stretch!")
        im = ax.imshow(data, norm=norm, cmap=cmap)
        ax.set_xlim(center[0] - 0.5*dx_pix, center[0] + 0.5*dx_pix)
        ax.set_ylim(center[1] - 0.5*dy_pix, center[1] + 0.5*dy_pix)
        ax.set_facecolor(facecolor)
        ax.set_xticks([])
        ax.set_yticks([])
        if do_contour:
            contour_data = (data)
            # contour_data[np.where(contour_data ==0)] = 1000
            P = ax.contour(contour_data, levels=(np.array(cont_levels)), colors='white', alpha=0.35, norm = norm, antialiased=True)
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
                        
    ax = plt.gca()
    with astropy.io.fits.open(img_fitsfile) as hdul:
        center = np.array([float(hdul[0].header['CRPIX1']),float(hdul[0].header['CRPIX2'])] )
    chip_width = 1050 ### CHANDRA
    if plot_calibration:
        ax.scatter(center[0],center[1], c = "yellow", marker = "+", s = 1000000, linewidths= 0.5)
        try:
            ax.scatter(center[0]+chip_width[0]/2,center[1]+chip_width[1]/2, c = "red", marker = "+", s = 1000, linewidths= 1)
            ax.scatter(center[0]-chip_width[0]/2,center[1]+chip_width[1]/2, c = "red", marker = "+", s = 1000, linewidths= 1)
            ax.scatter(center[0]+chip_width[0]/2,center[1]-chip_width[1]/2, c = "red", marker = "+", s = 1000, linewidths= 1)
            ax.scatter(center[0]-chip_width[0]/2,center[1]-chip_width[1]/2, c = "red", marker = "+", s = 1000, linewidths= 1)
        except:
            ax.scatter(center[0]+chip_width/2,center[1]+chip_width/2, c = "red", marker = "+", s = 1000, linewidths= 1)
            ax.scatter(center[0]-chip_width/2,center[1]+chip_width/2, c = "red", marker = "+", s = 1000, linewidths= 1)
            ax.scatter(center[0]+chip_width/2,center[1]-chip_width/2, c = "red", marker = "+", s = 1000, linewidths= 1)
            ax.scatter(center[0]-chip_width/2,center[1]-chip_width/2, c = "red", marker = "+", s = 1000, linewidths= 1)


    return im

from matplotlib.offsetbox import AnchoredText
plt.rcParams['text.usetex'] = True 

font = {'family' : 'monospace',
        'monospace':'Courier',
        'weight' : 'normal',
        'size'   : 45}
plt.rc("font", **font)


plot_calibration = False
reblock = 1
snapnum = 148

cmap = "inferno"
stretch = "sqrt"
plot_width = 0.2
vmin = 0#2e-2#1.2 / reblock**2
vmax = 3#0.1#2 #5/ reblock**2
unsharp_sigmas = None # (2,3)
do_contour=False
cont_levels = [55]

idx = 9
img_fitsfile = f""
fig, ax0 = plt.subplots(figsize = (10,10),)
im = soxs_plotter(img_fitsfile,ax0, unsharp_sigmas=unsharp_sigmas, unsharp_method="subtract", smooth = False, smooth_sigma = 8, stretch=stretch, cmap=cmap, vmin = vmin, vmax = vmax, width = plot_width, do_contour=do_contour, cont_levels = cont_levels, reblock = reblock)
cax = fig.add_axes([0.9, 0.11, 0.04, 0.77])
cbar = fig.colorbar(im, cax=cax,)
cbar.set_label("Counts", size = 30)
plt.show()
