from astropy import wcs
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from regions import Regions
from regions import PixCoord, RectanglePixelRegion, PolygonPixelRegion
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib.colors import PowerNorm, LogNorm, Normalize
import scipy
import os
import shutil
from tqdm import tqdm
import soxs
import sys
from pathlib import Path
import datetime
import logging
from soxs.instrument_registry import instrument_registry
from .tools import soxs_plotter
from soxs.instrument import RedistributionMatrixFile
from threeML import *
from threeML.io.package_data import get_path_of_data_file
from astromodels.xspec.factory import XS_bapec,XS_vapec, XS_bvapec, XS_apec, XS_TBabs
from astromodels.xspec.xspec_settings import *
from matplotlib.offsetbox import AnchoredText
from collections import Counter
import subprocess 
# from pympler.tracker import SummaryTracker

import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f GB"
              % (index, frame.filename, frame.lineno, stat.size / 1024**3))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f GB" % (len(other), size / 1024**3))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f GB" % (total / 1024**3))


class VoronoiTesselation():
    '''
    Class for making and fitting Voronoi tiles over the X-ray image, using Michele Cappelari's VorBin package (https://www-astro.physics.ox.ac.uk/~cappellari/software/#ppxf) to create the tile geometry. Files associated with this will be saved under {save_dir}/{self._run_ID}/instrument_name/VTESS
    ------------------------------------------------
    Constructor Positional Arguments:
                    observation: An instance of the Observation class. Many attributes of the observation class are faux-inherited by PostProcess so that we use the correct energies, redshift etc... We also inherit the active instruments and active halos from the Observation.
    Returns: PostProcess object
        
    '''
    def __init__(self,observation, vor_S2N = 50, median_cleaning_thresh = 2, reblock = 1 , needs_cleaning = True, needs_tesselating = True, needs_cutouts = True, just_plot = False, vmin = 3, vmax = 3e3, width = 0.3, cmap = "inferno", overwrite = False):
        
        self.hubble = observation.hubble
        if self.hubble == "from_box":
            self._logger.error(f"You set the hubble param to 'from_box' but you did not load the box!")
            self._logger.warning("h not set. Will use h=0.68")
            self.hubble = 0.68
        self._redshift = observation.redshift        

        self._run_ID = observation._run_ID
        self._top_save_path = observation._top_save_path
        self.emin = observation.emin
        self.emax = observation.emax
        self.active_halos = observation.active_halos
        self.active_instruments = observation.active_instruments
        self._instrument_defaults = observation._instrument_defaults
        
        
        self._logger = logging.getLogger("MOXHA")        
        if (self._logger.hasHandlers()):
            self._logger.handlers.clear()       
        c_handler = logging.StreamHandler()
        c_handler.setLevel(level = logging.INFO)
        self._logger.setLevel(logging.INFO)
        c_format = logging.Formatter('%(name)s - [%(levelname)-8s] --- %(asctime)s  - %(message)s')
        c_handler.setFormatter(c_format)
        self._logger.addHandler(c_handler)
        self._logger.info("Voronoi Tesselation Initialised")
        
        self.vor_S2N = vor_S2N
        self.median_cleaning_thresh = median_cleaning_thresh
        self.reblock = reblock
        
        self.vmin = vmin
        self.vmax = vmax
        self.width = width
        self.cmap = cmap
        
        self._thresh_mult = median_cleaning_thresh
        
        for instrument in self.active_instruments:
            
            self._instrument_name = instrument["Name"]
            if "ID" not in list(instrument.keys()):
                instrument_ID = self._instrument_name
            else:
                instrument_ID = instrument["ID"]
            
            
            for halo in self.active_halos:
                self._halo_idx = halo["index"]
                self._idx_tag = f"{self._run_ID}_h{str(self._halo_idx).zfill(3)}"
                self._idx_instr_tag = f"{self._idx_tag}_{instrument_ID}"
                
                self._bkgrnd_tag = f"{self._run_ID}_blanksky00_{instrument_ID}"
                self._obs_tag          = f"{self._top_save_path}/{self._instrument_name}/OBS/{self._idx_instr_tag}/{self._idx_instr_tag}"
                self._obs_img_fitsfile = f"{self._obs_tag}_img.fits"
                self._obs_evt_fitsfile = f"{self._obs_tag}_evt.fits" 
                self._obs_bkgrnd_evt_fitsfile =  f"{self._top_save_path}/{self._instrument_name}/OBS/{self._bkgrnd_tag}/{self._bkgrnd_tag}_evt.fits"

                self._tess_tag                    = f"{self._top_save_path}/{self._instrument_name}/VTESS/{self._idx_instr_tag}/{self._idx_instr_tag}"
                self._tess_img_fitsfile_uncleaned = f"{self._tess_tag}_uncleaned_img.fits"
                self._tess_evt_fitsfile_uncleaned = f"{self._tess_tag}_uncleaned_evt.fits"
                self._tess_evt_fitsfile_cleaned   = f"{self._tess_tag}_cleaned_evt.fits"
                self._tess_img_fitsfile_cleaned   = f"{self._tess_tag}_cleaned_img.fits"
                self._tess_cleaned_tag            = f"{self._tess_tag}_cleaned"
                self._tess_uncleaned_tag            = f"{self._tess_tag}_uncleaned"
                self._tess_bkgrnd_evt_fitsfile =  f"{self._top_save_path}/{self._instrument_name}/VTESS/{self._bkgrnd_tag}/{self._bkgrnd_tag}_evt.fits"
                
                
                os.makedirs(f"./{self._top_save_path}/{self._instrument_name}/VTESS/{self._idx_instr_tag}/", exist_ok = True)       
                os.makedirs(f"{self._top_save_path}/{self._instrument_name}/VTESS/{self._bkgrnd_tag}/", exist_ok = True)  
                
                
                if just_plot:
                    self._plot()
                    continue
                
                
                if needs_cleaning:
                    print("Cleaning Map")
                    self._clean_map()
                if needs_tesselating:
                    print("Creating Tesselations")
                    self._make_tiles()
                if needs_cutouts:
                    print("Making Cutouts")
                    self._tile_cutouts()
                self._fit_tiles_mwe(overwrite)
                    
                    
                

            

    def _clean_map(self):
        
        shutil.copy(self._obs_evt_fitsfile, self._tess_evt_fitsfile_uncleaned)
        shutil.copy(f"{self._obs_tag}_img.png", f"{self._tess_tag}_whole.png")
        if not os.path.exists(self._tess_bkgrnd_evt_fitsfile):
            shutil.copy(self._obs_bkgrnd_evt_fitsfile, self._tess_bkgrnd_evt_fitsfile)

        soxs.write_image(self._tess_evt_fitsfile_uncleaned, self._tess_img_fitsfile_uncleaned,  emin=self.emin, emax=self.emax, overwrite=True, reblock = self.reblock)
        fig, ax = soxs.plot_image(self._tess_img_fitsfile_uncleaned, stretch='log', cmap='cubehelix', )
        plt.savefig(f"{self._tess_uncleaned_tag}.png")  
        
        with fits.open(self._tess_img_fitsfile_uncleaned) as hdul:
            pixel_median  = np.median(np.array(hdul[0].data[np.where(hdul[0].data > 0)] ) )
            hot_pixels = list(np.array(np.where(hdul[0].data > self._thresh_mult * pixel_median)))
            hot_pixels = np.array(list(zip(hot_pixels[1], hot_pixels[0])))  #In FITS x and y are mixed up
            hottest = np.unravel_index(np.argmax(hdul[0].data), np.shape(hdul[0].data))

        if len(hot_pixels) > 0 :
            for i,hot_pixel in tqdm(enumerate(hot_pixels), desc = f"MOXHA - [INFO    ] ---{''.ljust(26,' ')}- Removing {len(hot_pixels)} Bad Pixels Pre-Tesselation".ljust(60,' '), total = len(hot_pixels)):
                reg = f"# Region file format: DS9\nimage\n-box({( (hot_pixel[0]+0.5) * self.reblock)+1}i, {( (hot_pixel[1]+0.5) * self.reblock)+1}i, {self.reblock}i,{self.reblock}i,0)\n" 
                # reg = f"# Region file format: DS9\nimage\n-circle({( (hot_pixel[0]+0.5) * self.reblock)+1}i, {( (hot_pixel[1]+0.5) * self.reblock)+1}i, {self.reblock/2}i,0)\n" 
                if i == 0:
                    total_region = Regions.parse(reg, format='ds9')[0]
                else:
                    pixel_region = Regions.parse(reg, format='ds9')[0]
                    total_region = total_region ^ pixel_region     
            print("Filtering...")
            soxs.filter_events(self._tess_evt_fitsfile_uncleaned, self._tess_evt_fitsfile_uncleaned, region=total_region, emin = self.emin, emax = self.emax, overwrite=True)
     
        shutil.move(self._tess_evt_fitsfile_uncleaned, self._tess_evt_fitsfile_cleaned)
        soxs.write_image(self._tess_evt_fitsfile_cleaned, self._tess_img_fitsfile_cleaned,  emin=self.emin, emax=self.emax, overwrite=True) 

        fig, ax = soxs.plot_image(self._tess_img_fitsfile_cleaned, stretch='log', cmap='cubehelix', )
        plt.savefig(f"{self._tess_cleaned_tag}.png")  
        fig.clear()   
        plt.close(fig)


        with fits.open(self._tess_img_fitsfile_cleaned) as hdul:
            hot_pixels_post_clean = list(np.array(np.where(hdul[0].data > self._thresh_mult * pixel_median)))
            if len(hot_pixels_post_clean) > 0:
                self._logger.warning(f"{len(hot_pixels_post_clean)} hot pixels remain after cleaning!")


                
#         ''' Now clean the background using the same pixel median thresh'''
#         soxs.write_image(self._tess_bkgrnd_evt_fitsfile_uncleaned, self._tess_bkgrnd_img_fitsfile_uncleaned,  emin=self.emin, emax=self.emax, overwrite=True, reblock = self.reblock)
#         with fits.open(self._tess_bkgrnd_img_fitsfile_uncleaned) as hdul:
#             print(np.max(hdul[0].data))

#         with fits.open(self._tess_bkgrnd_img_fitsfile_uncleaned) as hdul:
#             hot_pixels = list(np.array(np.where(hdul[0].data > self._thresh_mult * pixel_median)))
#             hot_pixels = np.array(list(zip(hot_pixels[1], hot_pixels[0])))  #In FITS x and y are mixed up
#             hottest = np.unravel_index(np.argmax(hdul[0].data), np.shape(hdul[0].data))

            
#         shutil.copy()
#         if len(hot_pixels) > 0 :
#             hot_pixels_chunked = np.array_split(hot_pixels, len(hot_pixels)//5)
#             # print(hot_pixels)
#             # print(hot_pixels_chunked)
#             for i,hot_pixel_chunk in tqdm(enumerate(hot_pixels_chunked), desc = f"MOXHA - [INFO    ] ---{''.ljust(26,' ')}- Removing Bad Pixels Pre-Tesselation".ljust(60,' '), total = len(hot_pixels_chunked)):
#                 reg = f"# Region file format: DS9\nimage"
#                 for hot_pixel in hot_pixel_chunk:
#                     # print(hot_pixel)
#                     reg += f"\n-box({( (hot_pixel[0]+0.5) * self.reblock)+1}i, { ((hot_pixel[1]+0.5)* self.reblock)+1}i, {6*self.reblock}i,{6*self.reblock}i,0)\n" 
#                 soxs.filter_events(self._tess_bkgrnd_evt_fitsfile_uncleaned, self._tess_bkgrnd_evt_fitsfile_uncleaned , region=reg, emin = self.emin, emax = self.emax, overwrite=True)
#         else:
#             self._logger.info(f"No Bad Pixels Identified in Annulus")
#             pass

#         shutil.move(self._tess_bkgrnd_evt_fitsfile_uncleaned, self._tess_bkgrnd_evt_fitsfile_cleaned)
#         soxs.write_image(self._tess_bkgrnd_evt_fitsfile_cleaned, self._tess_bkgrnd_img_fitsfile_cleaned,  emin=self.emin, emax=self.emax, overwrite=True) 

#         fig, ax = soxs.plot_image(self._tess_bkgrnd_img_fitsfile_cleaned, stretch='log', cmap='cubehelix', )
#         plt.savefig(f"{self._tess_bkgrnd_cleaned_tag}.png")  
#         fig.clear()   
#         plt.close(fig)
                
                
                
                
    def _plot(self):  
        width_param = 0.18
        norm = LogNorm() #Normalize() #
        kT_data = np.load(f"{self._tess_cleaned_tag}_kT_map.npy")
        fig = plt.figure(figsize = (10,10))
        ax0 = plt.gca()
        center = [kT_data.shape[0]/2, kT_data.shape[1]/2]
        dx_pix = 0.5*kT_data.shape[0]
        dy_pix = 0.5*kT_data.shape[1]
        im = plt.imshow(scipy.ndimage.gaussian_filter(kT_data,1), norm = norm, cmap = "magma" )
        ax0.set_xlim(center[0] - width_param*dx_pix, center[0] + width_param*dx_pix)
        ax0.set_ylim(center[1] - width_param*dy_pix, center[1] + width_param*dy_pix)    
        #     for tile2 in centre_out_tiles:
        #         vertices = tile2["vertices"]
        #         for i in range(len(vertices)):
        #             points_x = vertices[0] 
        #             points_y = vertices[1] 

        #             ax0.plot(points_x,points_y,ls = "dashed", color = "white", alpha = 0.5)
        #             ax0.plot([points_x[0],points_x[-1]],[points_y[0],points_y[-1]],ls = "dashed", color = "white", alpha = 0.5)
        cax = fig.add_axes([0.9, 0.11, 0.05, 0.77])
        cbar = fig.colorbar(im, cax=cax,)
        # current_cmap = matplotlib.cm.get_cmap()
        # current_cmap.set_bad(color='black')
        cbar.set_label("kT")
        plt.clim(vmin = 0.7, vmax=1.5)
        plt.show()                
                
                
                
                
                
                
                
                


    def _make_tiles(self,):
        
        
        smoothing = False
        pixsize = 1
        tile_list = []

        with fits.open(self._tess_img_fitsfile_cleaned) as hdul:
            if smoothing:
                data =  scipy.ndimage.gaussian_filter(hdul[0].data, sigma=3)
            else:
                data = hdul[0].data
                
        print(data.shape)
        x = np.shape(data)[1]
        y = np.shape(data)[0]  #x and y are reversed in the FITS file I'm pretty sure.
        x = np.arange(0,x)
        y = np.arange(0,y)
        signal = data
        xx, yy = np.meshgrid(x, y)
        xx = xx.reshape(xx.size)
        yy = yy.reshape(yy.size)
        signal = signal.reshape(signal.size)
        noise = np.array([ max(x**0.5, 10) for x in signal])
        all_data = np.array(list(zip(xx,yy,signal,noise)))
        all_data_clean = [datum for datum in all_data if x.shape[0]*0.44<datum[0]<x.shape[0]*0.56 and y.shape[0]*0.44<datum[1]<y.shape[0]*0.56 ]#and 10**(i+1) > x[2] > 0]
        xx = np.array([x[0] for x in all_data_clean])
        yy = np.array([x[1] for x in all_data_clean])
        signal = np.array([x[2] for x in all_data_clean])
        noise = np.array([x**0.5 for x in signal])
        total = np.sum(signal)
        print("total", total)
        # target_sn = np.sqrt(total / 150)
        # target_sn = np.sqrt(min_sig)
        noise = np.array([x[3] for x in all_data_clean])

        print(len(xx),len(yy),len(signal),len(noise))


        xx = np.array([xx[i] for i in range(len(signal)) if signal[i]>1])
        yy = np.array([yy[i] for i in range(len(signal)) if signal[i]>1])
        noise = np.array([noise[i] for i in range(len(signal)) if signal[i]>1])
        signal = np.array([signal[i] for i in range(len(signal)) if signal[i]>1])

        bin_number, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(
        xx, yy, signal, noise, target_sn = self.vor_S2N, cvt=True, pixelsize=pixsize, plot=False,
        quiet=True, sn_func=None, wvt=False, )
        fig = plt.gcf()
        fig.set_size_inches(10, 10)  #If plot too small not all bins are shown
        fig.set_dpi(200)
        plt.show()
        print("bin number",bin_number)

        nbins = max(bin_number)+1
        # print("nbins", nbins)
        for tile_num in range(0,nbins):
            tile_dict = {}
            pixels = np.array([(int(xx[i]),int(yy[i])) for i in range(len(bin_number)) if bin_number[i] == tile_num])
            if len(pixels) < 0:
                continue
            try:
                tile_dict["tile_num"] = tile_num
                tile_dict["pixels"] = pixels
                tile_list.append(tile_dict)
                hull = ConvexHull(pixels,incremental = True, qhull_options = "Pp")
                tile_dict["vertices"] = [np.array(pixels[hull.vertices,0]), np.array(pixels[hull.vertices,1])]
            except Exception as e:
                print("Error in qhull", e)
                pass


        np.save(f"{self._tess_cleaned_tag}_tiles.npy", tile_list)    


        fig = plt.figure(figsize = (10,10))
        fig.tight_layout(pad = -2.5)
        width = 0.3
        ax0 = plt.gca()
        im = soxs_plotter(self._tess_img_fitsfile_cleaned,ax0, stretch='log', cmap=self.cmap, vmin = self.vmin, vmax = self.vmax, width = self.width)
        cax = fig.add_axes([0.9, 0.11, 0.05, 0.77])
        cbar = fig.colorbar(im, cax=cax,)
        cbar.set_label("Counts (0.05-2keV)")

        for tile in tile_list:
            try:
                vertices = tile["vertices"]
            except:
                continue
            for i in range(len(vertices)):
                points_x = vertices[0] 
                points_y = vertices[1] 

                ax0.plot(points_x,points_y,ls = "dashed", color = "white", alpha = 0.2)
                ax0.plot([points_x[0],points_x[-1]],[points_y[0],points_y[-1]],ls = "dashed", color = "white", alpha = 0.2)

        # plt.xlim(128-80,128+80)
        # plt.ylim(128-80,128+80)
        ax = plt.gca()
        ax.set_facecolor("black")
        plt.savefig(f"{self._tess_cleaned_tag}_tess_overlay.png")
        plt.show()

        
        
    def _tile_cutouts(self,):
        from regions import Regions
        self._tess_cleaned_tag_fits = f"./{self._top_save_path}/{self._instrument_name}/VTESS/{self._idx_instr_tag}/fits_and_phas/{self._idx_instr_tag}"
        self._tess_bkgrnd_cleaned_tag_fits = f"./{self._top_save_path}/{self._instrument_name}/VTESS/{self._idx_instr_tag}/fits_and_phas/{self._bkgrnd_tag}_h{str(self._halo_idx).zfill(3)}"
        
        os.makedirs(f"./{self._top_save_path}/{self._instrument_name}/VTESS/{self._idx_instr_tag}/fits_and_phas/", exist_ok = True)   
        tiles = np.load(f"{self._tess_cleaned_tag}_tiles.npy", allow_pickle=True)
        print("We don't yet clean the background too")
        for tile in tqdm(tiles, total = len(tiles)):
            i = tile["tile_num"]
            try:
                vertex_coords = list(zip(tile["vertices"][0], tile["vertices"][1]))
            except:
                continue
            vertex_coords = [(x+0.5) * self.reblock for x in np.array(vertex_coords).flatten()]
            vertex_coords_str = " ".join(map(str,vertex_coords))
            reg = f"# Region file format: DS9\nimage\npolygon {vertex_coords_str}"

            region = Regions.parse(reg, format='ds9')[0]
            region.write(f'./tessregion{i}.reg', overwrite = True)

            soxs.filter_events(self._tess_evt_fitsfile_cleaned, f"{self._tess_cleaned_tag_fits}_t{str(i).zfill(3)}_evt.fits" , region=f'./tessregion{i}.reg', emin = self.emin, emax = self.emax, overwrite=True)
            soxs.write_spectrum(f"{self._tess_cleaned_tag_fits}_t{str(i).zfill(3)}_evt.fits", f"{self._tess_cleaned_tag_fits}_t{str(i).zfill(3)}_evt.pha", overwrite=True)
            
            soxs.write_image(f"{self._tess_cleaned_tag_fits}_t{str(i).zfill(3)}_evt.fits", f"{self._tess_cleaned_tag_fits}_t{str(i).zfill(3)}_img.fits",  emin=self.emin, emax=self.emax, overwrite=True) 
            fig, ax = soxs.plot_image(f"{self._tess_cleaned_tag_fits}_t{str(i).zfill(3)}_img.fits", stretch='log', cmap='cubehelix', vmin = self.vmin, vmax = self.vmax, width = self.width )

            plt.savefig(f"{self._tess_cleaned_tag_fits}_t{str(i).zfill(3)}.png")   
            fig.clear()   
            plt.close(fig)
            
            os.remove(f"{self._tess_cleaned_tag_fits}_t{str(i).zfill(3)}_evt.fits")
            os.remove(f"{self._tess_cleaned_tag_fits}_t{str(i).zfill(3)}_img.fits")

            
            '''Cut out same region in the background'''
            
            soxs.filter_events(self._tess_bkgrnd_evt_fitsfile, f"{self._tess_bkgrnd_cleaned_tag_fits}_t{str(i).zfill(3)}_evt.fits" , region=f'./tessregion{i}.reg', emin = self.emin, emax = self.emax, overwrite=True)
            soxs.write_spectrum(f"{self._tess_bkgrnd_cleaned_tag_fits}_t{str(i).zfill(3)}_evt.fits", f"{self._tess_bkgrnd_cleaned_tag_fits}_t{str(i).zfill(3)}_evt.pha", overwrite=True)
            
            soxs.write_image(f"{self._tess_bkgrnd_cleaned_tag_fits}_t{str(i).zfill(3)}_evt.fits", f"{self._tess_bkgrnd_cleaned_tag_fits}_t{str(i).zfill(3)}_img.fits",  emin=self.emin, emax=self.emax, overwrite=True) 
            fig, ax = soxs.plot_image(f"{self._tess_bkgrnd_cleaned_tag_fits}_t{str(i).zfill(3)}_img.fits", stretch='log', cmap='cubehelix', vmin = self.vmin, vmax = self.vmax, width = self.width )

            plt.savefig(f"{self._tess_bkgrnd_cleaned_tag_fits}_t{str(i).zfill(3)}.png")   
            fig.clear()   
            plt.close(fig)
            
            os.remove(f"{self._tess_bkgrnd_cleaned_tag_fits}_t{str(i).zfill(3)}_evt.fits")
            os.remove(f"{self._tess_bkgrnd_cleaned_tag_fits}_t{str(i).zfill(3)}_img.fits")
            os.remove(f'./tessregion{i}.reg')
            
            

            
            
    def _fit_tiles_mwe(self, overwrite):
        import os, psutil 

        
        import matplotlib
        from threeML import (
            silence_logs,
            silence_warnings,
            activate_logs,
            activate_warnings,
            update_logging_level,
        )
        # update_logging_level("DEBUG")

        
        
        self._tess_cleaned_tag_spectra = f"./{self._top_save_path}/{self._instrument_name}/VTESS/{self._idx_instr_tag}/spectral_fits/{self._idx_instr_tag}"
        os.makedirs(f"./{self._top_save_path}/{self._instrument_name}/VTESS/{self._idx_instr_tag}/spectral_fits/", exist_ok = True)
        self._tess_cleaned_tag_fits = f"./{self._top_save_path}/{self._instrument_name}/VTESS/{self._idx_instr_tag}/fits_and_phas/{self._idx_instr_tag}"        
        tiles = np.load(f"{self._tess_cleaned_tag}_tiles.npy", allow_pickle=True)
        centre_out_tiles = sorted(tiles, key=lambda d: len(d['pixels']), reverse = True)


            

        modapec = APEC() #XS_bvapec()
        modTbAbs = XS_TBabs()
        modapec.redshift.value = self._redshift # Source redshift
        modapec.redshift.fix = True
        modapec.kT.fix = False
        modapec.abund.value = 0.3
        modapec.abund.fix = False
        modTbAbs.nh.value = 0.018 # A value of 1 corresponds to 1e22 cm-2
        modTbAbs.nh.fix = True # NH is fixed   
        absorbed_apec = modapec*modTbAbs
        pts = PointSource("mysource", 30, 45, spectral_shape=absorbed_apec)
        model = Model(pts)

#         tracemalloc.start(25)

        if not overwrite:
            existing_tiles = [int(x.split("_")[-3].split("t")[-1]) for x in os.listdir(f"./{self._top_save_path}/{self._instrument_name}/VTESS/{self._idx_instr_tag}/spectral_fits/") if ".png" in x]
            if len(existing_tiles) == 0:
                overwrite = True
            else:
                centre_out_tiles = [x for x in centre_out_tiles if int(x["tile_num"]) not in existing_tiles]
                print("existing_tiles = ", existing_tiles)
                self._kT_data = np.load(f"{self._tess_cleaned_tag}_kT_map.npy")
            
        if overwrite:
            with fits.open(self._tess_img_fitsfile_cleaned) as hdul:
                self._kT_data = np.full_like(hdul[0].data, fill_value=3e-2)
            
        for tile in centre_out_tiles:
            print(f"Mem Usage: {round(float(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3),2)} GB")
            i = tile["tile_num"]
            pha_file = f"{self._tess_cleaned_tag_fits}_t{str(i).zfill(3)}_evt.pha"
            with fits.open(pha_file) as hdul:
                rmf = hdul["SPECTRUM"].header.get("RESPFILE", None)
                arf = hdul["SPECTRUM"].header.get("ANCRFILE", None)
            ogip_data = OGIPLike("ogip", observation= pha_file, response=f"./CODE/instr_files/{rmf}", arf_file = f"./CODE/instr_files/{arf}")
            ogip_data.set_active_measurements(f"{1.1*self.emin}-{0.9*self.emax}")
            jl = JointLikelihood(model, DataList(ogip_data))  
            _ = jl.fit()
            results = jl.results
            
            


            tile_params = {}
            results_dict = results.get_data_frame(error_type='equal tail', cl=0.68).to_dict(orient='index')
            tile_params["tile_num"] = i  
            tile_params["kT"] = results_dict['mysource.spectrum.main.composite.kT_1']
            tile_params["norm"] = results_dict['mysource.spectrum.main.composite.K_1']
            tile_params["abund"] = results_dict['mysource.spectrum.main.composite.abund_1']


            # self.fit_value_arrays.append(tile_params)  
            count_fig1 = ogip_data.display_model(step=False)
            plt.xlim(0.8*self.emin, 1.2*self.emax)
            plt.xscale("linear")

            at = AnchoredText(
                f"log10 kT = {np.log10(modapec.kT.value)}, log10 norm = {np.log10(modapec.K.value)}", prop=dict(size=8), frameon=True, loc='lower right')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0")
            ax = plt.gca()
            ax.add_artist(at)    
            count_fig1.axes[0].set_ylim(bottom = 1e-4)
            # self._logger.info(f"Saving Fitting Plot under {self.annuli_path}/spectral_fits/{self.idx_instr_tag}_{str(self.tile).zfill(2)}_deprojected_fit.png")
            plt.savefig(f"{self._tess_cleaned_tag_spectra}_t{str(i).zfill(3)}_spectral_fit.png")
            plt.clf()
            plt.close() 

            for pix in tile["pixels"]:
                self._kT_data[pix[1],pix[0]] = modapec.kT.value


            fig = plt.figure(figsize = (10,10))
            ax0 = plt.gca()
            center = [self._kT_data.shape[0]/2, self._kT_data.shape[1]/2]
            dx_pix = 0.5*self._kT_data.shape[0]
            dy_pix = 0.5*self._kT_data.shape[1]
            im = plt.imshow(self._kT_data, norm = LogNorm(), cmap = "magma" )
            ax0.set_xlim(center[0] - 0.25*dx_pix, center[0] + 0.25*dx_pix)
            ax0.set_ylim(center[1] - 0.25*dy_pix, center[1] + 0.25*dy_pix)    
        #     for tile2 in centre_out_tiles:
        #         vertices = tile2["vertices"]
        #         for i in range(len(vertices)):
        #             points_x = vertices[0] 
        #             points_y = vertices[1] 

        #             ax0.plot(points_x,points_y,ls = "dashed", color = "white", alpha = 0.5)
        #             ax0.plot([points_x[0],points_x[-1]],[points_y[0],points_y[-1]],ls = "dashed", color = "white", alpha = 0.5)
            cax = fig.add_axes([0.9, 0.11, 0.05, 0.77])
            cbar = fig.colorbar(im, cax=cax,)
            # current_cmap = matplotlib.cm.get_cmap()
            # current_cmap.set_bad(color='black')
            cbar.set_label("kT")
            plt.clim(vmin = 0.07)
            plt.savefig(f"{self._tess_cleaned_tag}_tesselation.png")
            plt.clf()
            plt.close() 
            # self._logger.info(f"Spectral fit saved under {self.annuli_path}/spectral_fits/{self.idx_instr_tag}_{str(self.tile).zfill(2)}_deprojected_fit")

            fig = plt.figure(figsize = (10,10))
            ax0 = plt.gca()
            center = [self._kT_data.shape[0]/2, self._kT_data.shape[1]/2]
            dx_pix = 0.5*self._kT_data.shape[0]
            dy_pix = 0.5*self._kT_data.shape[1]
            im = plt.imshow(scipy.ndimage.gaussian_filter(self._kT_data, sigma=1.5), norm = LogNorm(), cmap = "magma")
            ax0.set_xlim(center[0] - 0.25*dx_pix, center[0] + 0.25*dx_pix)
            ax0.set_ylim(center[1] - 0.25*dy_pix, center[1] + 0.25*dy_pix)    
        #     for tile2 in centre_out_tiles:
        #         vertices = tile2["vertices"]
        #         for i in range(len(vertices)):
        #             points_x = vertices[0] 
        #             points_y = vertices[1] 

        #             ax0.plot(points_x,points_y,ls = "dashed", color = "white", alpha = 0.5)
        #             ax0.plot([points_x[0],points_x[-1]],[points_y[0],points_y[-1]],ls = "dashed", color = "white", alpha = 0.5)
            cax = fig.add_axes([0.9, 0.11, 0.05, 0.77])
            cbar = fig.colorbar(im, cax=cax, )
            # current_cmap = matplotlib.cm.get_cmap()
            # current_cmap.set_bad(color='black')
            cbar.set_label("kT")
            plt.clim(vmin = 0.07)
            plt.clim(vmin = 0.6*min(self._kT_data.flatten()))
            plt.savefig(f"{self._tess_cleaned_tag}_smoothed_tesselation.png")
            plt.clf()
            plt.close() 
            np.save(f"{self._tess_cleaned_tag}_kT_map.npy", self._kT_data)             
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            # print(type(result))
            # result.analysis_type
            # result.optimized_model
            # result.write_to("./test_mle.fits", overwrite=True)
            
        

        #     snapshot = tracemalloc.take_snapshot()
        #     display_top(snapshot)
        # top_stats = snapshot.statistics('traceback')
        # stat = top_stats[0]

       
                
 
                
                
#     def _fit_tiles(self,):
        
#         self._tess_cleaned_tag_spectra = f"./{self._top_save_path}/{self._instrument_name}/VTESS/{self._idx_instr_tag}/spectral_fits/{self._idx_instr_tag}"
#         os.makedirs(f"./{self._top_save_path}/{self._instrument_name}/VTESS/{self._idx_instr_tag}/spectral_fits/", exist_ok = True)
#         self._tess_cleaned_tag_fits = f"./{self._top_save_path}/{self._instrument_name}/VTESS/{self._idx_instr_tag}/fits_and_phas/{self._idx_instr_tag}"
        
        
#         from threeML import (
#             silence_logs,
#             silence_warnings,
#             activate_logs,
#             activate_warnings,
#             update_logging_level,
#         )
#         from threeML import silence_progress_bars, activate_progress_bars, toggle_progress_bars
#         from threeML.utils.progress_bar import trange
        
#         from threeML import quiet_mode
#         # silence_logs
#         # silence_warnings
#         # quiet_mode
#         update_logging_level("CRITICAL")
#         # update_logging_level("DEBUG")
        
#         tiles = np.load(f"{self._tess_cleaned_tag}_tiles.npy", allow_pickle=True)
#         centre_out_tiles = sorted(tiles, key=lambda d: len(d['pixels']), reverse = True)

#         with fits.open(self._tess_img_fitsfile_cleaned) as hdul:
#             self._kT_data = np.full_like(hdul[0].data, fill_value=3e-2)
            
            
            

#         modapec = APEC() #XS_bvapec()
#         modTbAbs = XS_TBabs()
#         modapec.redshift.value = self._redshift # Source redshift
#         modapec.redshift.fix = True
#         modapec.kT.fix = False
#         modapec.abund.value = 0.3
#         modapec.abund.fix = False
#         modTbAbs.nh.value = 0.018 # A value of 1 corresponds to 1e22 cm-2
#         modTbAbs.nh.fix = True # NH is fixed   
#         absorbed_apec = modapec*modTbAbs
#         pts = PointSource("mysource", 30, 45, spectral_shape=absorbed_apec)
#         model = Model(pts)
        
#         tracemalloc.start(25)
#         import matplotlib
#         analysis = []
#         for self._cum_tile_num, self._tile in enumerate(centre_out_tiles):

            
#             i = self._tile["tile_num"]
#             self._pha_file = f"{self._tess_cleaned_tag_fits}_t{str(i).zfill(3)}_evt.pha"
#             with fits.open(self._pha_file) as hdul:
#                 rmf = hdul["SPECTRUM"].header.get("RESPFILE", None)
#                 arf = hdul["SPECTRUM"].header.get("ANCRFILE", None)

#             rmf = f"./CODE/instr_files/{rmf}"
#             arf = f"./CODE/instr_files/{arf}"
            
#             try:
#                 self.ogip_data = OGIPLike(
#                     "ogip",
#                     observation=self._pha_file,
#                     response=rmf,
#                     arf_file = arf
#                 )
#             except Exception as e:
#                 self._logger.error(f"Error loading OGIP", e)
#             # print("files open post ogip", self._cum_tile_num, )
#             # os.system(f'lsof -p {os.getpid()} | wc -l')
#                 # return

#             # self._logger.info("3ML OGIP Loaded Correctly")
#             # self._logger.info(f"Will fit between {1.1*self.emin}-{0.9*self.emax}")
#             self.ogip_data.set_active_measurements(f"{1.1*self.emin}-{0.9*self.emax}")


#             # print("files open post models", self._cum_tile_num, )
#             # os.system(f'lsof -p {os.getpid()} | wc -l')

#             jl = JointLikelihood(model, DataList(self.ogip_data))  

#             open_files= subprocess.getoutput(f'lsof -F n -F fd  -p {os.getpid()}').splitlines()[1:]
#             open_files1_dict = dict(zip([x[1:] for x in open_files if x[0] == 'n'], [x[1:] for x in open_files if x[0] == 'f']))

#             result = jl.fit()

#             open_files= subprocess.getoutput(f'lsof -F n -F fd  -p {os.getpid()}').splitlines()[1:]
#             open_files2_dict = dict(zip([x[1:] for x in open_files if x[0] == 'n'], [x[1:] for x in open_files if x[0] == 'f']))

#             new_opens = list(set(open_files2_dict.keys()) - set(open_files1_dict.keys())) #Counter(list(open_files2_dict) - Counter(list(open_files1_dict)
#             new_opens = list(new_opens)

#             print("-" *10)
#             print(len(open_files1_dict), len(open_files2_dict), len(open_files2_dict)-len(open_files1_dict) ,len(new_opens))
#             print(new_opens)
#             # for new in new_opens:
#             #     print(new)        



#             analysis.append(jl.results.optimized_model)


#             snapshot = tracemalloc.take_snapshot()
#             display_top(snapshot)
#             top_stats = snapshot.statistics('traceback')

#             # pick the biggest memory block
#             stat = top_stats[0]
#             print("\n %s memory blocks: %.1f GB" % (stat.count, stat.size / 1024**3))
#             for line in stat.traceback.format():
#                 print(line)           
    


#         tile_params = {}
#         results_dict = results.get_data_frame(error_type='equal tail', cl=0.68).to_dict(orient='index')
#         tile_params["tile_num"] = i  
#         tile_params["kT"] = results_dict['mysource.spectrum.main.composite.kT_1']
#         tile_params["norm"] = results_dict['mysource.spectrum.main.composite.K_1']
#         tile_params["abund"] = results_dict['mysource.spectrum.main.composite.abund_1']


#         # self.fit_value_arrays.append(tile_params)  
#         count_fig1 = self.ogip_data.display_model(step=False)
#         plt.xlim(0.8*self.emin, 1.2*self.emax)
#         plt.xscale("linear")

#         at = AnchoredText(
#             f"log10 kT = {np.log10(modapec.kT.value)}, log10 norm = {np.log10(modapec.K.value)}", prop=dict(size=8), frameon=True, loc='lower right')
#         at.patch.set_boxstyle("round,pad=0.,rounding_size=0")
#         ax = plt.gca()
#         ax.add_artist(at)    
#         count_fig1.axes[0].set_ylim(bottom = 1e-4)
#         # self._logger.info(f"Saving Fitting Plot under {self.annuli_path}/spectral_fits/{self.idx_instr_tag}_{str(self.tile).zfill(2)}_deprojected_fit.png")
#         plt.savefig(f"{self._tess_cleaned_tag_spectra}_t{str(i).zfill(3)}_spectral_fit.png")
#         plt.clf()
#         plt.close() 

#         for pix in self._tile["pixels"]:
#             self._kT_data[pix[1],pix[0]] = modapec.kT.value


#         fig = plt.figure(figsize = (10,10))
#         ax0 = plt.gca()
#         center = [self._kT_data.shape[0]/2, self._kT_data.shape[1]/2]
#         dx_pix = 0.5*self._kT_data.shape[0]
#         dy_pix = 0.5*self._kT_data.shape[1]
#         im = plt.imshow(self._kT_data, norm = LogNorm(), cmap = "inferno")
#         ax0.set_xlim(center[0] - 0.25*dx_pix, center[0] + 0.25*dx_pix)
#         ax0.set_ylim(center[1] - 0.25*dy_pix, center[1] + 0.25*dy_pix)    
#     #     for tile2 in centre_out_tiles:
#     #         vertices = tile2["vertices"]
#     #         for i in range(len(vertices)):
#     #             points_x = vertices[0] 
#     #             points_y = vertices[1] 

#     #             ax0.plot(points_x,points_y,ls = "dashed", color = "white", alpha = 0.5)
#     #             ax0.plot([points_x[0],points_x[-1]],[points_y[0],points_y[-1]],ls = "dashed", color = "white", alpha = 0.5)
#         cax = fig.add_axes([0.9, 0.11, 0.05, 0.77])
#         cbar = fig.colorbar(im, cax=cax,)
#         current_cmap = matplotlib.cm.get_cmap()
#         current_cmap.set_bad(color='black')
#         cbar.set_label("kT")
#         plt.savefig(f"{self._tess_cleaned_tag}_tesselation.png")
#         plt.clf()
#         plt.close() 
#         # self._logger.info(f"Spectral fit saved under {self.annuli_path}/spectral_fits/{self.idx_instr_tag}_{str(self.tile).zfill(2)}_deprojected_fit")

#         fig = plt.figure(figsize = (10,10))
#         ax0 = plt.gca()
#         center = [self._kT_data.shape[0]/2, self._kT_data.shape[1]/2]
#         dx_pix = 0.5*self._kT_data.shape[0]
#         dy_pix = 0.5*self._kT_data.shape[1]
#         im = plt.imshow(scipy.ndimage.gaussian_filter(self._kT_data, sigma=1.5), norm = LogNorm(), cmap = "inferno")
#         ax0.set_xlim(center[0] - 0.25*dx_pix, center[0] + 0.25*dx_pix)
#         ax0.set_ylim(center[1] - 0.25*dy_pix, center[1] + 0.25*dy_pix)    
#     #     for tile2 in centre_out_tiles:
#     #         vertices = tile2["vertices"]
#     #         for i in range(len(vertices)):
#     #             points_x = vertices[0] 
#     #             points_y = vertices[1] 

#     #             ax0.plot(points_x,points_y,ls = "dashed", color = "white", alpha = 0.5)
#     #             ax0.plot([points_x[0],points_x[-1]],[points_y[0],points_y[-1]],ls = "dashed", color = "white", alpha = 0.5)
#         cax = fig.add_axes([0.9, 0.11, 0.05, 0.77])
#         cbar = fig.colorbar(im, cax=cax,)
#         cbar.set_label("kT")
#         plt.clim(vmin = 0.6*min(self._kT_data.flatten()))
#         plt.savefig(f"{self._tess_cleaned_tag}_smoothed_tesselation.png")
#         plt.clf()
#         plt.close() 
        np.save(f"{self._tess_cleaned_tag}_kT_map.npy", self._kT_data) 
        # print("files open post plots", self._cum_tile_num, )
        # os.system(f'lsof -p {os.getpid()} | wc -l')
        # self.ogip_data = None

        

        
        # print(open_files_dict)
        for new_open in new_opens:
            if "/atomdb/APED/" in new_open or "/site-packages/threeML/" in new_open  or "/site-packages/astromodels/" in new_open:
                print(open_files2_dict[new_open])
                print(f"Closing {new_open}")
                os.close(int(open_files2_dict[new_open]))
                