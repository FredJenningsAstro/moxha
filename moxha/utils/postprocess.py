import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
# from astropy.cosmology import FLRW
from soxs.instrument import RedistributionMatrixFile
import soxs
from soxs.instrument_registry import instrument_registry
import os
from astropy.io import fits
from astropy.units import arcmin, Mpc
from pathlib import Path
from scipy import ndimage
from os.path import exists
from astropy.wcs import WCS
from matplotlib.offsetbox import AnchoredText
import datetime
import shutil
import sys
from tqdm import tqdm
import logging
import math
import time
import astropy
from regions import PixCoord, PixelRegion, Region, Regions, SkyRegion, CirclePixelRegion

class PostProcess():
    '''
    Class for Post-Processing the observations made through the Observation class. We will clean and deproject the observation into radial annuli based on signal-to-noise, and subtract off the background, with an end-product being spectra that can be fit by the FitRadialSpectra class.
    PostProcess files will be saved under {save_dir}/{run_ID}/instrument_name/ANNULI
    ------------------------------------------------
    Constructor Positional Arguments:
                    observation: An instance of the Observation class. Many attributes of the observation class are faux-inherited by PostProcess so that we use the correct energies, redshift etc... We also inherit the active instruments and active halos from the Observation.
    Returns: PostProcess object
        
    '''
    def __init__(self,observation):
        self._redshift = observation.redshift
        self.hubble = observation.hubble

        self.cosmo = FlatLambdaCDM(H0 = 100*self.hubble, Om0 = 0.3, Ob0 = 0.048, Tcmb0=2.7 )
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
        self._logger.info("Post-Processing Initialised")
        
        if self.hubble == "from_box":
            self._logger.error(f"You set the hubble param to 'from_box' but you did not load the box!")
            sys.exit()        

    def add_instrument(self, Name, exp_time, ID = None, reblock=1, aim_shift=None, chip_width=None):
        '''
        Function for adding instruments to the PostProcess. Should be an instrument with which you've made an Observation.
        ------------------------------------------------
        Positional Arguments:
                        Name: Instrument name as found in SOXS Instrument Registry (https://hea-www.cfa.harvard.edu/soxs/users_guide/instrument.html), e.g. "athena_wfi"
                        exp_time: Exposure time in ks. Must be < photon_sample_exp as passed to MakePhotons()

        Keyword Arguments:
                        ID: Extra identification tag for this instrument. Useful if e.g. you are making observations with two different exposure times with the same instrument. 
                            In this case you can put the exposure time in this ID string and the filenames will reflect this. Default = None
                        reblock: Reblock factor for SOXS when writing images. N x N image pixels will be rebinned into (N/reblock x N/reblock) pixels for some img.fits files in 
                                the deprojection stage, where having fewer pixels to filter significantly speeds up the annulus cleaning of CXB sources. Default = 1 (no reblocking)
                        aim_shift: Shift from the normal aim point in units of arcseconds. Corresponds to aimpt_shift in SOXS (https://hea-www.cfa.harvard.edu/soxs/api/instrument.html#soxs.instrument.instrument_simulator). 
                                    Default = select from self._instrument_defaults
                        chip_width: Chip width in pixels. Used for cutting out the region of interest in the deprojection stage, to avoid chip gaps. Default = select from self._instrument_defaults     
        Returns: 

        '''
        defaults = [x for x in self._instrument_defaults if x["Name"] == Name ][0]
        if len(self.active_instruments) == 0:
            self.active_instruments = []
        if aim_shift == None:
            aim_shift = defaults["aim_shift"]
        if chip_width == None:
            chip_width = defaults["chip_width"]
        if ID == None:
            ID = f"{Name}_{exp_time}ks" 
        self.active_instruments.append({"Name":Name,"ID":ID, "exp_time": exp_time, "reblock":reblock, "aim_shift":aim_shift, "chip_width":chip_width})
            
            
    def clear_instruments(self):
        '''
        Clear all currently active PostProcess instruments.
        ------------------------------------------------
        Returns:
        '''
        self.active_instruments = [] 
        
        
    def set_active_halos(self, halos):
        '''
        Set active halos for the PostProcessing. Should be in the set of halos for which you've made observations.
        ------------------------------------------------
        Positional Arguments:
                    halos: Dictionary or list of dictionaries where each dictionary should contain at a minimum
                    { "index": halo_idx}
        Returns:
        '''
        if isinstance(halos, dict):
            self.active_halos = [halos,]
        else:
            self.active_halos = halos        
        
        
    def generate_annuli(self, num_backgrounds = 1, S2N_threshold = 200**0.5, S2N_range = (1,2), S2N_good_fraction = 0.7, hot_pix_thresh = 2, simple_bkngd_subtract = True, max_ann_width = 20, overwrite = False, edge_rstep_mult = 4):
        '''
        Function to use generate cleaned, deprojected annuli and record their spectras from the observations. Observations corresponding to currently active halos and instruments will be processed.
        ------------------------------------------------
        Keyword Arguments:
                    num_backgrounds: if > 1 will average use this number of blank-sky observations to average the noise spectrum. Default = 1
                    S2N_threshold: Threshold (poissonian) signal to noise ratio in the range specified by S2N_range that a fraction equal to S2N_good_fraction of the channels must achieve for the annulus to be accepted. Default = 5
                    S2N_range: Range over which to test the signal to noise for annulus acceptance. Default = (1,2)
                    S2N_good_fraction: The fraction of channels in the testing range that must obey the signal to noise criterion for an accepted annulus. Default = 0.7
                    hot_pix_threshold: Pixels in an annulus with a photon count larger than hot_pix_threshold * annulus median will be masked. Default = 2
                    simple_bkngd_subtract: Simply subtract the (average) noise spectrum from the cleaned blank-sky observations from the signal spectrum. Default = True
                    max_ann_width: The maximum width of the outer annulus you want to allow in arcsecs. If the outer annulus reaches this width without the S2N threshold being achived then the outer radius is moved in by 25% of max_ann_width. Default = 10
                    
        Returns: 

        '''            
        self._edge_rstep_mult = edge_rstep_mult
        for i,halo in enumerate(self.active_halos):
            halo_idx = halo["index"]
            self._simple_bkgnd_subtract = simple_bkngd_subtract
            self._num_backgrounds = int(num_backgrounds)
            self.idx_tag = f"{self._run_ID}_h{str(halo_idx).zfill(3)}"
            self._postprocess_log = self._top_save_path/"LOGS"/f"postprocess_log_{self.idx_tag}.log"
            self._hot_pix_thresh = hot_pix_thresh
            self._max_ann_width = max_ann_width
            
            for self._instrument in self.active_instruments:
                self.instrument_name = self._instrument["Name"]
                try:
                    self._reblock = self._instrument["reblock"]
                except:
                    self._reblock = 1
                try:
                    pixel_steps = self._instrument["pixel_steps"]
                except:
                    pixel_steps = 2
                try:
                    self.S2N = self._instrument["signal_to_noise_threshold"]
                except:
                    self.S2N = S2N_threshold
                try:
                    self.S2N_range = self._instrument["signal_to_noise_range"]
                except:
                    self.S2N_range = S2N_range               
                try:
                    self.S2N_good_fraction = self._instrument["signal_to_noise_good_fraction"]
                except:
                    self.S2N_good_fraction = S2N_good_fraction
                    
                try:
                    self._chip_width = self._instrument["chip_width"]
                except:
                    self._chip_width = None
                if "ID" not in list(self._instrument.keys()):
                    self._instrument_ID = self.instrument_name
                else:
                    self._instrument_ID = self._instrument["ID"]
                self.idx_instr_tag = f"{self.idx_tag}_{self._instrument_ID}"
                self.evts_path = Path(self._top_save_path/self.instrument_name/"OBS"/self.idx_instr_tag)
                spectra_path = Path(self._top_save_path/self.instrument_name/"SPECTRA"/self.idx_instr_tag)
                os.makedirs(spectra_path, exist_ok = True)  
                if not exists(f"{self.evts_path}/{self.idx_instr_tag}_evt.fits"):
                    self._logger.warning(f"{self.evts_path}/{self.idx_instr_tag}_evt.fits not found!")
                    continue
                    
                self.annuli_path = Path(self._top_save_path/self.instrument_name/"ANNULI"/self.idx_instr_tag)
                if os.path.exists(self.annuli_path/"spectral_fits/") and not overwrite:
                    self._logger.info(f"{self.annuli_path}/spectral_fits/ already exists and overwrite == False, so we will skip annuli generation.")
                    continue
                    
                self._bkgnd_annuli_path = self.annuli_path/"background_files/"
                os.makedirs(f"{self.annuli_path}/deproj_spectras", exist_ok = True)
                os.makedirs(f"{self.annuli_path}/fits_and_phas", exist_ok = True)
                os.makedirs(f"{self.annuli_path}/map_pngs", exist_ok = True)
                os.makedirs(f"{self.annuli_path}/DATA", exist_ok = True)
                os.makedirs(self.annuli_path/"background_files/", exist_ok = True)  
                os.makedirs(f"{self._bkgnd_annuli_path}/map_pngs/", exist_ok = True)
                os.makedirs(f"{self._bkgnd_annuli_path}/deproj_spectras/", exist_ok = True)                   
                os.makedirs(self.annuli_path, exist_ok = True)
                
                bkgnd_idx_tag = f"{self._run_ID}_blanksky{str(0).zfill(2)}"
                self._bkgnd_idx_instr_tag = f"{bkgnd_idx_tag}_{self._instrument_ID}"
                self._bkgnd_evts_path = Path(self._top_save_path/self.instrument_name/"OBS"/self._bkgnd_idx_instr_tag)
                shutil.copy(f"{self._bkgnd_evts_path}/{self._bkgnd_idx_instr_tag}_img.png", f"{self._bkgnd_annuli_path}/{self._bkgnd_idx_instr_tag}_img.png")     
                self._logger.info(f" copying {self._bkgnd_evts_path}/{self._bkgnd_idx_instr_tag}_img.png to {self._bkgnd_annuli_path}/{self._bkgnd_idx_instr_tag}_img.png" )

                self._find_central_pixel( smoothing = False,)
                self._logger.info(f"Brightest spot = {self.brightest_pos_pixels}      {self.brightest_pos_degrees}")
                self._logger.info(f"Wrtiting whole image spectrum. This can take a while, and a fair amount of memory...")
                os.makedirs(f"{spectra_path}/", exist_ok = True, )
                soxs.write_spectrum(f"{self.evts_path}/{self.idx_instr_tag}_evt.fits", f"{spectra_path}/{self.idx_instr_tag}_whole_image_evt.pha", overwrite=True)
                with fits.open(f"{spectra_path}/{self.idx_instr_tag}_whole_image_evt.pha") as hdul:
                    total_counts = np.sum(hdul[1].data.field('COUNTS'))    
                    self._write_log(f"Total counts = {total_counts}")
                shutil.copy(f"{self.evts_path}/{self.idx_instr_tag}_img.png", f"{self.annuli_path}/{self.idx_instr_tag}_entire_obs_img.png")           
                
                if self._chip_width == None:
                    self._r_i_out = round(self.img_width_pixels,5) #round(width_from_brightest_degrees,5)
                else:
                    self._r_i_out = self._chip_width
                self._r_step = pixel_steps             
                self._do_deprojections()
                self._logger.info(f"Finished Creating Deprojected Annuli for {self.idx_instr_tag}")
                

                
                
                
                
    def _write_log(self, message):
        now = datetime.datetime.now()
        f = open(self._postprocess_log, 'a')
        f.write("[{0}] ".format(now.strftime("%Y-%m-%d %H:%M:%S")) + str(message))
        f.write('\n')
        f.close()


    def _write_warning(self, message):
        now = datetime.datetime.now()
        f = open(self._postprocess_log, 'a')
        f.write("[{0}] >>>>> WARNING! >>>> ".format(now.strftime("%Y-%m-%d %H:%M:%S")) + str(message))
        f.write('\n')
        f.close()
        
    def _find_central_pixel(self, smoothing = False,):
        self._ = f"{self.evts_path}/{self.idx_instr_tag}_evt.fits"
        img_fitsfile = f"{self.evts_path}/{self.idx_instr_tag}_img.fits"
        if smoothing == True:
            shutil.copy(self._, f"{self.evts_path}/{self.idx_instr_tag}_smoothed_evt.fits")
            self._ = f"{self.evts_path}/{self.idx_instr_tag}_smoothed_evt.fits"
            img_fitsfile = f"{self.evts_path}/{self.idx_instr_tag}_smoothed_img.fits"
            soxs.write_image(self._, img_fitsfile,  emin=self.emin, emax=self.emax, overwrite=True, )
            f = fits.open(img_fitsfile, mode = 'update')
            f[hdu].data = ndimage.gaussian_filter(f[hdu].data, sigma=3)
            f.close()
            fig, ax = soxs.plot_image(img_fitsfile, stretch='log', cmap='cubehelix',)
            plt.savefig(f"{self.evts_path}/{self.idx_instr_tag}_smoothed_img.png")   
            fig.clear() 
            plt.close(fig)

        hdu="IMAGE"
        with fits.open(img_fitsfile) as hdul:
            wcs = WCS(hdul[0].header)
            where_max  = np.where(hdul[0].data == np.amax(hdul[0].data))
            where_max = np.array(list(zip(where_max[0], where_max[1])))
            if len(where_max) > 1:
                self._logger.info("Multiple brightest pixels detected. Will just use center of image...")
                self.brightest_pos_pixels = np.array([float(hdul[0].header['CRPIX1']),float(hdul[0].header['CRPIX2'])] )
            self.brightest_pos_pixels = np.array(where_max[0])
            self.img_width = 0.5*hdul[0].data.shape[0]
            # self.brightest_pos_pixels = np.array([float(hdul[0].header['CRPIX1']),float(hdul[0].header['CRPIX2'])] )
            # if abs(float(hdul[0].header['CRPIX1'])- self.brightest_pos_pixels[0]) > self.img_width/10 or abs(float(hdul[0].header['CRPIX2'])- self.brightest_pos_pixels[1]) > self.img_width/50:  #MAKE SURE crpix1/2 right way around for non-symmetric plane
            self.brightest_pos_pixels = np.array([float(hdul[0].header['CRPIX1']),float(hdul[0].header['CRPIX2'])] ) 
            print(self.brightest_pos_pixels)
            print(np.array(hdul[0].data.shape)/2)
            self._logger.warning(f"Setting center at {self.brightest_pos_pixels} which is the center of the chip, coinciding with user-provided halo true center.")
            self.brightest_pos_degrees = wcs.pixel_to_world(self.brightest_pos_pixels[0],self.brightest_pos_pixels[1])
            '''Find all edge positions of the X-ray map, so we can figure out the maximum radius from the central pixel we can use. '''
            self.img_width_pixels = 0.5*hdul[0].data.shape[0]
 
   
    def _do_deprojections(self, keep_annuli_evts = False, keep_annuli_img = False):
        self._keep_annuli_evts = keep_annuli_evts
        self._annuli_radius_arr = []
        self._write_log("-------------------------------")
        self._write_log("Starting Deprojection Process...")
        self._write_log(f"Reblock = {self._reblock}")
        self._write_log(f"Pixel Steps = {self._r_step}")
        self._annulus_number = 0  
        self._r_i_in = self._r_i_out     
        self._dirty_pix = []
        self._edge_attempt = -1
        nx = instrument_registry[self.instrument_name]["num_pixels"]
        plate_scale_arcsec_per_pix = 60 * instrument_registry[self.instrument_name]["fov"]/nx   #in arcmin   

        self._evts_fitsfile = f"{self.evts_path}/{self.idx_instr_tag}_evt.fits"
        self._make_chip_cutout()
         
        while self._r_i_in >= self._r_step:   
            if self._annulus_number != 0:
                self._r_i_in -= self._r_step
            else:
                self._edge_attempt += 1
                self._r_i_in -= self._edge_rstep_mult * self._r_step  # Fixed width to find the edge         
            if self._r_i_in <= 0:
                self._logger.info("We have reached the center of the cut-out region. Returning...")
                break     
            if self._r_i_in <= self._r_step:
                self._r_i_in = 0.01
            self._r_i_in = round(self._r_i_in,6)  
            
            with fits.open(f"{self.evts_path}/{self.idx_instr_tag}_img.fits") as hdul:
                wcs = WCS(hdul[0].header)   
            central_pos_pixel = [float(self.brightest_pos_pixels[0]+1), float(self.brightest_pos_pixels[1]+1)]
            r_i_in_kpc, r_i_out_kpc, _, _ = self._convert_to_kpc(central_pos_pixel=central_pos_pixel, r_i_in_pos_pixel=self._r_i_in, r_i_out_pos_pixel=self._r_i_out,wcs=wcs  )
            
            self._get_cleaned_signal_annulus()
            self._get_cleaned_bkgnd_annulus()
            
            final_counts, spec_fig = self._deproject_annulus(r_i_in_kpc=r_i_in_kpc, r_i_out_kpc=r_i_out_kpc)
            low_idx,high_idx,counts_in_range,all_counts,good_S2N_fraction = self._determine_good_fraction(final_counts)
            
            if good_S2N_fraction >= self.S2N_good_fraction or self._r_i_in <= self._r_step:
                self._S2N_is_good(spec_fig,low_idx,high_idx,counts_in_range,all_counts,good_S2N_fraction)
            else:
                self._S2N_is_bad(spec_fig,low_idx,high_idx,counts_in_range,all_counts,good_S2N_fraction, plate_scale_arcsec_per_pix)                                                                
            continue
                
        if self._annulus_number == 0:
            self._logger.warning(f"No Deprojected Annuli Accepted for {self.idx_instr_tag}! We are done...")
            return
        
        self._plot_cleaned_cutout_region(keep_annuli_img)
        np.save( f"{self.annuli_path}/DATA/{self.idx_instr_tag}_pixel_radii.npy", self._annuli_radius_arr )
        self._convert_pixcoords_to_kpc()

        
        
        
    def _make_chip_cutout(self,):
        self._logger.info(f"Making a chip cutout with width {self._chip_width}")
        ds9_circle_reg = str(f"# Region file format: DS9\nimage\ncircle({float(self.brightest_pos_pixels[0]+1.5)}, {float(self.brightest_pos_pixels[1]+1.5)},{self._chip_width})\n")

        soxs.filter_events(self._evts_fitsfile, f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_considered_region_evt.fits", region=ds9_circle_reg, emin = self.emin, emax = self.emax, overwrite=True)
        soxs.write_image(f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_considered_region_evt.fits", f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_considered_region_img.fits",  emin=self.emin, emax=self.emax, overwrite=True)          
        fig, ax = soxs.plot_image(f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_considered_region_img.fits", stretch='log', cmap='cubehelix', )
        ax = self._append_markers(ax,f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_considered_region_img.fits")
        plt.savefig(f"{self.annuli_path}/{self.idx_instr_tag}_considered_region.png")           
        plt.close()          
        
        

    def _get_cleaned_signal_annulus(self,):
        self._raw_uncleaned_candidate = f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_{str(self._annulus_number).zfill(2)}_raw_uncleaned_canditate"
        self._raw_cleaned_candidate = f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_{str(self._annulus_number).zfill(2)}_raw_cleaned_canditate"
        self._accepted_candidate = f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_{str(self._annulus_number).zfill(2)}_deprojected"
        self._ds9_annulus_reg = str(f"# Region file format: DS9\nimage\nannulus({float(self.brightest_pos_pixels[0]+1.5)}, {float(self.brightest_pos_pixels[1]+1.5)}, {self._r_i_in}, {self._r_i_out})\n")
        soxs.filter_events(self._, f"{self._raw_uncleaned_candidate}_evt.fits", region=self._ds9_annulus_reg,emin = self.emin, emax = self.emax, overwrite=True)
        soxs.write_image(f"{self._raw_uncleaned_candidate}_evt.fits", f"{self._raw_uncleaned_candidate}_img.fits",  emin=self.emin, emax=self.emax, overwrite=True, reblock = self._reblock)  
        self._clean_signal_annulus()
        soxs.write_spectrum(f"{self._raw_cleaned_candidate}_evt.fits", f"{self._raw_cleaned_candidate}.pha", overwrite=True)        
        
        
    def _clean_signal_annulus(self):
        with fits.open(f"{self._raw_uncleaned_candidate}_img.fits") as hdul:
            self._pixel_median  = np.median( np.array(hdul[0].data[np.where(hdul[0].data > 0)] ) )
            hot_pixels = list(np.array(np.where(hdul[0].data > self._hot_pix_thresh * self._pixel_median)))
            
            hot_pixels = np.array(list(zip(hot_pixels[1], hot_pixels[0])))  #In FITS x and y are mixed up
            all_annulus_pixels = list(np.array(np.where(hdul[0].data > 0)))
            all_annulus_pixels = np.array(list(zip(all_annulus_pixels[1], all_annulus_pixels[0]))) 
            self._signal_annulus_masking_correction = len(all_annulus_pixels)/ ( len(all_annulus_pixels) - len(hot_pixels) )

        if len(hot_pixels) > 0 :
            self._logger.info(f"Removing {len(hot_pixels)} Bad Signal Annulus Pixels ({self.idx_instr_tag}, Annulus {self._annulus_number})")
            for i,hot_pixel in tqdm(enumerate(hot_pixels), desc = f"MOXHA - [INFO    ] ---{''.ljust(26,' ')}- Removing Bad Signal Annulus Pixels (Annulus {self._annulus_number})    ".ljust(60,' '), total = len(hot_pixels)):
                self._dirty_pix.append(hot_pixel)
                reg = f"# Region file format: DS9\nimage\n-box({( (hot_pixel[0]+0.5) * self._reblock)+1}i, {( (hot_pixel[1]+0.5) * self._reblock)+1}i, {self._reblock}i,{self._reblock}i,0)\n" 
                if i == 0:
                    total_region = Regions.parse(reg, format='ds9')[0]
                else:
                    pixel_region = Regions.parse(reg, format='ds9')[0]
                    total_region = total_region ^ pixel_region                    
            soxs.filter_events(f"{self._raw_uncleaned_candidate}_evt.fits", f"{self._raw_uncleaned_candidate}_evt.fits", region=total_region, emin = self.emin, emax = self.emax, overwrite=True)
            soxs.write_image(f"{self._raw_uncleaned_candidate}_evt.fits", f"{self._raw_uncleaned_candidate}_temp_img.fits",  emin=self.emin, emax=self.emax, overwrite=True, reblock = self._reblock)  
            with fits.open(f"{self._raw_uncleaned_candidate}_temp_img.fits") as hdul:
                orig_hot_pixels = hot_pixels
                hot_pixels = list(np.array(np.where(hdul[0].data > self._hot_pix_thresh * self._pixel_median)))
                hot_pixels = np.array(list(zip(hot_pixels[1], hot_pixels[0])))
                if len(hot_pixels) > 0:
                    self._logger.warning(f"{len(hot_pixels)} bad pixels remain in Signal Annulus after cleaning! ({self.idx_instr_tag}, Annulus {self._annulus_number})")
                    # self._logger.warning(f"Bad Signal Pixels: {hot_pixels} ({self.idx_instr_tag}, Annulus {self._annulus_number})")
                    # self._logger.info(f"Original hot pixels: {orig_hot_pixels}")
                    # raise RuntimeError()
            os.remove(f"{self._raw_uncleaned_candidate}_temp_img.fits") 
            self._logger.info(f"All Bad Signal Annulus Pixels Removed ({self.idx_instr_tag}, Annulus {self._annulus_number})")
        else:
            self._logger.info(f"No Bad Pixels Identified in Signal ({self.idx_instr_tag}, Annulus {self._annulus_number})")
             
        shutil.move(f"{self._raw_uncleaned_candidate}_evt.fits", f"{self._raw_cleaned_candidate}_evt.fits")
                 
            
    def _get_cleaned_bkgnd_annulus(self):
            bkgnd_evt_fitsfile = f"{self._bkgnd_evts_path}/{self._bkgnd_idx_instr_tag}_evt.fits"
            bkgnd_img_fitsfile = f"{self._bkgnd_evts_path}/{self._bkgnd_idx_instr_tag}_img.fits"
            self._bkgnd_uncleaned = f"{self.annuli_path}/fits_and_phas/{self._bkgnd_idx_instr_tag}_{str(self._annulus_number).zfill(2)}_uncleaned"
            self._bkgnd_cleaned = f"{self.annuli_path}/fits_and_phas/{self._bkgnd_idx_instr_tag}_{str(self._annulus_number).zfill(2)}_cleaned"
            soxs.filter_events(bkgnd_evt_fitsfile, f"{self._bkgnd_uncleaned}_evt.fits", region=self._ds9_annulus_reg,emin = self.emin, emax = self.emax, overwrite=True)
            soxs.write_image(f"{self._bkgnd_uncleaned}_evt.fits", f"{self._bkgnd_uncleaned}_img.fits",  emin=self.emin, emax=self.emax, overwrite=True, reblock = self._reblock) 
            fig, ax = soxs.plot_image(f"{self._bkgnd_uncleaned}_img.fits", stretch='log', cmap='cubehelix', )
            ax = self._append_markers(ax, f"{self._bkgnd_uncleaned}_img.fits", reblock = self._reblock)
            plt.savefig(f"{self.annuli_path}/map_pngs/{self._bkgnd_idx_instr_tag}_{str(self._annulus_number).zfill(2)}_uncleaned.png")
            fig.clear()   
            plt.close(fig)
            self._clean_bkgrnd_annulus()
            soxs.write_spectrum(f"{self._bkgnd_cleaned}_evt.fits", f"{self._bkgnd_cleaned}.pha", overwrite=True)
            with fits.open(f"{self._bkgnd_cleaned}.pha") as hdul_bkgrnd:  
              self._bkgrnd_annulus_counts = hdul_bkgrnd["SPECTRUM"].data['COUNTS'] 
              self._bkgrnd_annulus_count_rate =hdul_bkgrnd["SPECTRUM"].data['COUNT_RATE'] 
         
        
    def _clean_bkgrnd_annulus(self):
        with fits.open(f"{self._bkgnd_uncleaned}_img.fits") as hdul:
            '''Here we use the same median value as used for the signal annulus'''
            hot_pixels = list(np.array(np.where(hdul[0].data > self._hot_pix_thresh * self._pixel_median)))
            hot_pixels = np.array(list(zip(hot_pixels[1], hot_pixels[0])))  #In FITS x and y are mixed up
            all_annulus_pixels = list(np.array(np.where(hdul[0].data > 0)))
            all_annulus_pixels = np.array(list(zip(all_annulus_pixels[1], all_annulus_pixels[0]))) 
            self._bkgrnd_annulus_masking_correction = len(all_annulus_pixels)/ ( len(all_annulus_pixels) - len(hot_pixels) )

            
        if len(hot_pixels) > 0 :
            self._logger.info(f"Removing All Bad Bkgrnd Annulus Pixels ({self.idx_instr_tag}, Annulus {self._annulus_number})")
            for i,hot_pixel in tqdm(enumerate(hot_pixels), desc = f"MOXHA - [INFO    ] ---{''.ljust(26,' ')}- Removing Bad Background Annulus Pixels ({self.idx_instr_tag}, Annulus {self._annulus_number})".ljust(60,' '), total = len(hot_pixels)):
                reg = f"# Region file format: DS9\nimage\n-box({( (hot_pixel[0]+0.5) * self._reblock)+1}i, {( (hot_pixel[1]+0.5) * self._reblock)+1}i, {self._reblock}i,{self._reblock}i,0)\n" 
                if i == 0:
                    total_region = Regions.parse(reg, format='ds9')[0]
                else:
                    pixel_region = Regions.parse(reg, format='ds9')[0]
                    total_region = total_region ^ pixel_region            
                    
            soxs.filter_events(f"{self._bkgnd_uncleaned}_evt.fits", f"{self._bkgnd_uncleaned}_evt.fits", region=total_region, emin = self.emin, emax = self.emax, overwrite=True)  
            soxs.write_image(f"{self._bkgnd_uncleaned}_evt.fits", f"{self._bkgnd_uncleaned}_temp_img.fits",  emin=self.emin, emax=self.emax, overwrite=True, reblock = self._reblock)  
            with fits.open(f"{self._bkgnd_uncleaned}_temp_img.fits") as hdul:
                hot_pixels = list(np.array(np.where(hdul[0].data > self._hot_pix_thresh * self._pixel_median)))
                hot_pixels = np.array(list(zip(hot_pixels[1], hot_pixels[0])))
                if len(hot_pixels) > 0:
                    self._logger.warning(f"{len(hot_pixels)} bad pixels remain in background annulus after cleaning! ({self.idx_instr_tag}, Annulus {self._annulus_number})")
                    # self._logger.warning(f"Bad Background Pixels: {hot_pixels} ({self.idx_instr_tag}, Annulus {self._annulus_number})")
                    # raise RuntimeError()
            os.remove(f"{self._bkgnd_uncleaned}_temp_img.fits")             
            
            
            
            self._logger.info(f"All Bad Bkgrnd Annulus Pixels Removed ({self.idx_instr_tag}, Annulus {self._annulus_number})")
        else:
            self._logger.info(f"No Bad Pixels Identified in Background Annulus ({self.idx_instr_tag}, Annulus {self._annulus_number})")      
        shutil.move(f"{self._bkgnd_uncleaned}_evt.fits", f"{self._bkgnd_cleaned}_evt.fits")     
        soxs.write_image(f"{self._bkgnd_cleaned}_evt.fits", f"{self._bkgnd_cleaned}_img.fits",  emin=self.emin, emax=self.emax, overwrite=True, reblock = self._reblock) 
        fig, ax = soxs.plot_image(f"{self._bkgnd_cleaned}_img.fits", stretch='log', cmap='cubehelix', )
        ax = self._append_markers(ax, f"{self._bkgnd_cleaned}_img.fits", reblock = self._reblock)
        plt.savefig(f"{self.annuli_path}/map_pngs/{self._bkgnd_idx_instr_tag}_{str(self._annulus_number).zfill(2)}_cleaned.png")   
        fig.clear()   
        plt.close(fig)
            
            
            
            
            
        
    def _deproject_annulus(self,r_i_in_kpc,r_i_out_kpc):
        with fits.open(f"{self._raw_cleaned_candidate}.pha") as hdul_main:
                fig = plt.figure(figsize = (10,5))
                rmf = hdul_main["SPECTRUM"].header.get("RESPFILE", None)
                rmf = RedistributionMatrixFile(rmf)
                self._energy_bins = 0.5*(rmf.ebounds_data["E_MIN"]+rmf.ebounds_data["E_MAX"])
                plt.plot(self._energy_bins, hdul_main["SPECTRUM"].data['COUNTS'], label = "Raw", color = "red") 
                plt.plot(self._energy_bins, hdul_main["SPECTRUM"].data['COUNTS'] * self._signal_annulus_masking_correction, label = f"Raw * masking correct = {round(self._signal_annulus_masking_correction,3)}", color = "black", ls = 'dotted')
                pixel_area = math.pi * (   self._r_i_out**2 - self._r_i_in**2   )

                if self._num_backgrounds > 0 and self._simple_bkgnd_subtract:
                    # self._logger.warning("Subtracting Background Spectrum in a Simple Way (May not be valid, especially for low counts!)...")
                    self._logger.info(f"Applying Signal masking correction = {round(self._signal_annulus_masking_correction,3)}          Applying bkgrnd masking correction = {round(self._bkgrnd_annulus_masking_correction,3)}")
                    hdul_main["SPECTRUM"].data['COUNTS'] = (self._signal_annulus_masking_correction * np.copy(hdul_main["SPECTRUM"].data.field('COUNTS'))) - (self._bkgrnd_annulus_masking_correction * self._bkgrnd_annulus_counts) 
                    hdul_main["SPECTRUM"].data['COUNT_RATE'] = (self._signal_annulus_masking_correction * np.copy(hdul_main["SPECTRUM"].data.field('COUNT_RATE'))) - (self._bkgrnd_annulus_masking_correction * self._bkgrnd_annulus_count_rate)
                    hdul_main["SPECTRUM"].data['COUNTS'][np.where(hdul_main["SPECTRUM"].data['COUNTS'] < 0 )] = 0
                    hdul_main["SPECTRUM"].data['COUNT_RATE'][np.where(hdul_main["SPECTRUM"].data['COUNT_RATE'] < 0 )] = 0
                    plt.plot(self._energy_bins, self._bkgrnd_annulus_counts * self._bkgrnd_annulus_masking_correction , color = "orange", label = "Background Contribution to Annulus")
                    plt.plot(self._energy_bins, hdul_main["SPECTRUM"].data['COUNTS'], color = "red", ls = "dashed", label = "Projected Signal minus Noise")

                for external_shell in self._annuli_radius_arr:
                        spectrum_number = external_shell[0]
                        r_j_in, r_j_out = external_shell[1], external_shell[2]
                        with fits.open( f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_{str(spectrum_number).zfill(2)}_deprojected.pha") as hdul_contaminant:
                            S_j_rates = hdul_contaminant["SPECTRUM"].data.field('COUNT_RATE') 
                            S_j_counts = hdul_contaminant["SPECTRUM"].data.field('COUNTS')    

                            geometric_factor= ( ( r_j_in**2 - self._r_i_out**2)**(3/2)
                                              + (r_j_out**2 -  self._r_i_in**2)**(3/2) 
                                              - (r_j_out**2 - self._r_i_out**2)**(3/2)
                                              - ( r_j_in**2 -  self._r_i_in**2)**(3/2) )/(
                                                (r_j_out**2 -        r_j_in**2)**(3/2)   )

                            hdul_main["SPECTRUM"].data['COUNTS'] = np.copy(hdul_main["SPECTRUM"].data.field('COUNTS')) - (geometric_factor *  S_j_counts)
                            hdul_main["SPECTRUM"].data['COUNT_RATE'] = np.copy(hdul_main["SPECTRUM"].data.field('COUNT_RATE')) - (geometric_factor *  S_j_rates)     
                            plt.plot(self._energy_bins, geometric_factor *  S_j_counts, label = f"Contribution from external_shell {spectrum_number}", alpha = 0.2)

                if len(np.where(hdul_main['SPECTRUM'].data['COUNTS'] < 0 )[0]) > 0:
                    # self._logger.warning(f"{len(np.where(hdul_main['SPECTRUM'].data['COUNTS'] < 0 )[0])} < 0 energy bins in deprojected spectrum. Setting to 0...")
                    pass
                hdul_main["SPECTRUM"].data['COUNTS'][np.where(hdul_main["SPECTRUM"].data['COUNTS'] < 0 )] = 0
                hdul_main["SPECTRUM"].data['COUNT_RATE'][np.where(hdul_main["SPECTRUM"].data['COUNT_RATE'] < 0 )] = 0
                self._logger.info(f"Deprojected {self.idx_instr_tag} annulus {self._annulus_number} (annulus radii: {r_i_in_kpc.round(1)} to {r_i_out_kpc.round(1)})") 
                plt.plot(self._energy_bins, hdul_main["SPECTRUM"].data['COUNTS'], label = "Final Deproj", color = "green", ls = "solid", alpha = 1)
                final_counts = hdul_main["SPECTRUM"].data['COUNTS']     
                plt.xlim(0,self.emax*1.2)
                plt.yscale("symlog", linthresh = 1)  
                plt.legend()
                plt.xlabel("E (keV)")
                plt.ylabel("Bare Counts (?)")     
                hdul_main.writeto(f"{self._raw_cleaned_candidate}.pha", overwrite = True)
        return final_counts, fig       
  


    def _determine_good_fraction(self,final_counts):
        with fits.open(f"{self._raw_cleaned_candidate}.pha") as hdul:
            if not np.array_equal(hdul["SPECTRUM"].data['COUNTS'] , final_counts):
                self._logger.error("Deprojected spectrum did not save properly. Exiting...")
                sys.exit()
            if len(self._energy_bins) != len(hdul["SPECTRUM"].data['COUNTS']):
                raise Exception() 
                sys.exit()

            '''1-2kev towards high end of emission for groups'''
            low_idx = (np.abs(self._energy_bins - self.S2N_range[0])).argmin()
            high_idx = (np.abs(self._energy_bins - self.S2N_range[1])).argmin()

            ''' Get the count rates in the desired energy range.'''
            all_counts = np.array(hdul[1].data.field('COUNTS'))
            counts_in_range = np.array(hdul[1].data.field('COUNTS'))[low_idx:high_idx]
            num_good = len(counts_in_range[counts_in_range > self.S2N**2])
            good_S2N_fraction = round(num_good/(high_idx-low_idx) , 3)
        self._logger.info(f"Good fraction = {round(good_S2N_fraction,2)}      ({self.idx_instr_tag}, Annulus {self._annulus_number})")
        return low_idx,high_idx,counts_in_range,all_counts,good_S2N_fraction


    
    def _S2N_is_good(self,fig, low_idx,high_idx,counts_in_range,all_counts,good_S2N_fraction):
        self._logger.info(f"Success! The SNR ratio is acceptable for {self.idx_instr_tag} annulus {self._annulus_number}." )
        plt.savefig(f"{self.annuli_path}/deproj_spectras/{self.idx_instr_tag}_projected_contributions_{str(self._annulus_number).zfill(2)}.png")
        fig.clear() 
        plt.close(fig) 
        fig = plt.figure(figsize = (10,5))
        plt.plot( self._energy_bins, all_counts , color = "blue")
        plt.plot( self._energy_bins[low_idx:high_idx], counts_in_range , label = "Deproj Spectrum in S/N testing range", color = "orange")
        at = AnchoredText(
            f"good S/N fraction = {good_S2N_fraction}", prop=dict(size=15), frameon=True, loc='lower left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax = plt.gca()
        ax.add_artist(at)
        plt.xlim(0,self.emax*1.2)
        # plt.ylim(1, 3000)
        
        plt.yscale("log")  
        plt.legend()
        plt.xlabel("E (keV)")
        plt.ylabel("Folded Counts")  
        plt.hlines(y= self.S2N**2, xmin = 0, xmax = 1.1*self.emax, linestyles = "dashed")
        plt.savefig(f"{self.annuli_path}/deproj_spectras/{self.idx_instr_tag}_Annulus_{str(self._annulus_number).zfill(2)}_signal_noise_test_ACCEPTED.png")
        fig.clear()   
        plt.close(fig)
        self._write_log(f"r_i_in = {self._r_i_in}, r_i_out = {self._r_i_out},    Good Fraction = {good_S2N_fraction}   DEPROJ SUCCESSFUL    ({self.idx_instr_tag}, Annulus {self._annulus_number})")          
        shutil.move(f"{self._raw_cleaned_candidate}.pha", f"{self._accepted_candidate}.pha")  
        shutil.move(f"{self._raw_cleaned_candidate}_evt.fits", f"{self._accepted_candidate}_evt.fits")                                 
        fig, ax = soxs.plot_image(f"{self._raw_uncleaned_candidate}_img.fits", stretch='sqrt', cmap='cubehelix', )
        ax = self._append_markers(ax,f"{self._raw_uncleaned_candidate}_img.fits", reblock = self._reblock)

        plt.savefig(f"{self.annuli_path}/map_pngs/{self.idx_instr_tag}_{str(self._annulus_number).zfill(2)}_raw_uncleaned.png")    
        fig.clear() 
        plt.close(fig)

        soxs.write_image(f"{self._accepted_candidate}_evt.fits", f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_{str(self._annulus_number).zfill(2)}_raw_cleaned_img.fits",  emin=self.emin, emax=self.emax, overwrite=True, reblock = self._reblock)      
        fig, ax = soxs.plot_image(f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_{str(self._annulus_number).zfill(2)}_raw_cleaned_img.fits", stretch='sqrt', cmap='cubehelix',)
        ax = self._append_markers(ax,f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_{str(self._annulus_number).zfill(2)}_raw_cleaned_img.fits", reblock = self._reblock)
        plt.savefig(f"{self.annuli_path}/map_pngs/{self.idx_instr_tag}_{str(self._annulus_number).zfill(2)}_cleaned.png")           
        fig.clear()  
        plt.close(fig)   
        if not self._keep_annuli_evts:
            os.remove(f"{self._accepted_candidate}_evt.fits")

        mean_r_i = (self._r_i_in + self._r_i_out)/2
        self._annuli_radius_arr.append( [self._annulus_number, self._r_i_in, self._r_i_out, [float(self.brightest_pos_pixels[0]+1), float(self.brightest_pos_pixels[1]+1)] ] )

        self._write_log(f"r_i_in = {self._r_i_in}, r_i_out = {self._r_i_out},    Good Fraction = {good_S2N_fraction}   DEPROJ SUCCESSFUL   ({self.idx_instr_tag}, Annulus {self._annulus_number})")
        self._annulus_number += 1
        self._r_i_out = self._r_i_in
        os.remove(f"{self._raw_uncleaned_candidate}_img.fits")    
        
        
    
    def _S2N_is_bad(self, fig,low_idx,high_idx,counts_in_range,all_counts,good_S2N_fraction, plate_scale_arcsec_per_pix):
        fig.clear()  
        plt.close(fig)
        fig = plt.figure(figsize = (10,5))
        plt.plot( self._energy_bins, all_counts , color = "blue")
        plt.plot( self._energy_bins[low_idx:high_idx], counts_in_range , label = "Deproj Spectrum in S/N testing range", color = "orange")
        at = AnchoredText(
            f"good S/N fraction = {good_S2N_fraction}", prop=dict(size=15), frameon=True, loc='lower left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax = plt.gca()
        ax.add_artist(at)
        plt.xlim(0,self.emax*1.2)
        # plt.ylim(1, 3000)
        plt.hlines(y= self.S2N**2, xmin = 0, xmax = 1.1*self.emax, linestyles = "dashed")
        plt.yscale("symlog", linthresh = 1)  
        plt.legend()
        plt.xlabel("E (keV)")
        plt.ylabel("Bare Counts (?)")  
        # plt.savefig(f"{self.annuli_path}/deproj_spectras/{self.idx_instr_tag}_Annulus_{str(self._annulus_number).zfill(2)}_edge_attempt_{self._edge_attempt}_signal_noise_test_FAILED.png")
        fig.clear()   
        plt.close(fig)
        self._write_log(f"r_i_in = {self._r_i_in}, r_i_out = {self._r_i_out},    Good Fraction = {good_S2N_fraction}  ({self.idx_instr_tag}, Annulus {self._annulus_number})")  

        # if self._annulus_number == 0:
        #     fig, ax = soxs.plot_image(f"{self._raw_uncleaned_candidate}_img.fits", stretch='sqrt', cmap='twilight',)
        #     plt.savefig(f"{self.annuli_path}/map_pngs/{self.idx_instr_tag}_{str(self._annulus_number).zfill(2)}_uncleaned_edge_attempt_{self._edge_attempt}.png")                        

        os.remove(f"{self._raw_cleaned_candidate}_evt.fits") 
        os.remove(f"{self._raw_cleaned_candidate}.pha")
        os.remove(f"{self._raw_uncleaned_candidate}_img.fits")
        os.remove(f"{self._bkgnd_cleaned}.pha"  )
        os.remove(f"{self._bkgnd_cleaned}_img.fits")
        os.remove(f"{self._bkgnd_cleaned}_evt.fits")
        os.remove(f"{self._bkgnd_uncleaned}_img.fits")
        os.remove(f"{self.annuli_path}/map_pngs/{self._bkgnd_idx_instr_tag}_{str(self._annulus_number).zfill(2)}_uncleaned.png")
        os.remove(f"{self.annuli_path}/map_pngs/{self._bkgnd_idx_instr_tag}_{str(self._annulus_number).zfill(2)}_cleaned.png")               

        if self._annulus_number == 0 and (self._r_i_out- self._r_i_in) >= (plate_scale_arcsec_per_pix * self._max_ann_width):
            self._logger.info(f"Outer annulus width = {(self._r_i_out- self._r_i_in)/plate_scale_arcsec_per_pix} arcsec, which is larger than limit of {self._max_ann_width} arcsec, so shifting outer edge...")
            self._r_i_out = self._r_i_in + 0.75*(self._r_i_out - self._r_i_in)
            self._r_i_in = self._r_i_out        
    
    
    
    def _plot_cleaned_cutout_region(self, keep_annuli_img):
        shutil.copy(f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_{str(0).zfill(2)}_raw_cleaned_img.fits", f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_considered_region_cleaned_img.fits")
        with fits.open(f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_considered_region_cleaned_img.fits") as hdul_main:
            for annulus_number in range(1,self._annulus_number):
                with fits.open(f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_{str(annulus_number).zfill(2)}_raw_cleaned_img.fits") as hdul_an:
                    hdul_main[0].data += hdul_an[0].data
                if not keep_annuli_img:
                    os.remove(f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_{str(annulus_number).zfill(2)}_raw_cleaned_img.fits")
            hdul_main.writeto(f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_considered_region_cleaned_img.fits", overwrite = True) 

        fig, ax = soxs.plot_image(f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_considered_region_cleaned_img.fits", stretch='log', cmap='cubehelix' )
        ax = self._append_markers(ax,f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_considered_region_cleaned_img.fits")
        plt.savefig(f"{self.annuli_path}/{self.idx_instr_tag}_considered_region_cleaned.png")     
        fig.clear()   
        plt.close(fig) 
        
        
    def _convert_pixcoords_to_kpc(self,):
        self._logger.info(f"Converting Annuli Pixel Scales to kpc using angular diameter distance...")
        self._logger.info(f"Whole detector (all chips) fov = {instrument_registry[self.instrument_name]['fov']}'") 
        '''Use the img fits under /OBS/ which MUST NOT BE REBLOCKED'''
        with fits.open(f"{self.evts_path}/{self.idx_instr_tag}_img.fits") as hdul:
            wcs = WCS(hdul[0].header)   
        radii_kpc_arr = []
        for annulus in self._annuli_radius_arr:
            r_i_in_pos_pixel = annulus[1]
            r_i_out_pos_pixel = annulus[2]
            central_pos_pixel = annulus[3]
            r_i_in_kpc, r_i_out_kpc, r_i_in_sep, r_i_out_sep   = self._convert_to_kpc(central_pos_pixel=central_pos_pixel, r_i_in_pos_pixel=r_i_in_pos_pixel, r_i_out_pos_pixel=r_i_out_pos_pixel,wcs=wcs  )

            '''Lets just check we get the same arcmin from using fov/pixels scale...'''
            nx = instrument_registry[self.instrument_name]["num_pixels"]
            plate_scale = instrument_registry[self.instrument_name]["fov"]/nx   #in arcmin         
            dif_to_astropy = np.abs( (plate_scale* r_i_out_pos_pixel) -  r_i_out_sep.arcmin) / r_i_out_sep.arcmin

            if dif_to_astropy > 0.05:
                self._logger.warning("Astropy vs fof/pixels scale disagree on arcmins!")

            self._logger.info(f"Annulus {str(annulus[0]).zfill(2)}: r_i_in = {round(r_i_in_sep.arcmin,3)}' ({round(r_i_in_kpc.value,3)}), r_i_out = {round(r_i_out_sep.arcmin,3)}' ({round(r_i_out_kpc.value,3)} kpc) ")
            radii_kpc_arr.append( {"annulus_num":annulus[0], "radii":[r_i_in_kpc, r_i_out_kpc]} )

        np.save( f"{self.annuli_path}/DATA/{self.idx_instr_tag}_kpc_radii.npy", radii_kpc_arr )    
        
    
    def _convert_to_kpc(self,central_pos_pixel, r_i_in_pos_pixel, r_i_out_pos_pixel,wcs):
        r_i_in_pos_degrees = wcs.pixel_to_world(central_pos_pixel[0], central_pos_pixel[1]+r_i_in_pos_pixel)
        r_i_out_pos_degrees = wcs.pixel_to_world(central_pos_pixel[0], central_pos_pixel[1]+r_i_out_pos_pixel)
        center_pos_degrees = wcs.pixel_to_world(central_pos_pixel[0], central_pos_pixel[1])

        r_i_in_sep = center_pos_degrees.separation(r_i_in_pos_degrees)#.arcmin  * arcmin
        r_i_out_sep = center_pos_degrees.separation(r_i_out_pos_degrees)#.arcmin  * arcmin

        r_i_in_kpc = (r_i_in_sep.radian * self.cosmo.angular_diameter_distance(z = self._redshift)).to("kpc")
        r_i_out_kpc = (r_i_out_sep.radian * self.cosmo.angular_diameter_distance(z = self._redshift)).to("kpc")
        return r_i_in_kpc, r_i_out_kpc, r_i_in_sep, r_i_out_sep   
    
    
    
    
    
    def _append_markers(self, ax, img_file, reblock = 1):
        with astropy.io.fits.open(img_file) as hdul:
            center = np.array([float(hdul[0].header['CRPIX1']),float(hdul[0].header['CRPIX2'])] )
            ax.scatter(center[0],center[1], c = "yellow", marker = "+", s = 1000000, linewidths= 0.5)
            instrument_spec = instrument_registry[self.instrument_name]
            try:
        
                chip_width = float(np.array(instrument_spec["chips"])[1][[3,4]][0])
                ax.scatter(center[0]+chip_width[0]/2,center[1]+chip_width[1]/2, c = "red", marker = "+", s = 1000, linewidths= 1)
                ax.scatter(center[0]-chip_width[0]/2,center[1]+chip_width[1]/2, c = "red", marker = "+", s = 1000, linewidths= 1)
                ax.scatter(center[0]+chip_width[0]/2,center[1]-chip_width[1]/2, c = "red", marker = "+", s = 1000, linewidths= 1)
                ax.scatter(center[0]-chip_width[0]/2,center[1]-chip_width[1]/2, c = "red", marker = "+", s = 1000, linewidths= 1)
            except:
                chip_width = float(np.array(instrument_spec["chips"])[0][[3,4]][0])
                ax.scatter(center[0]+chip_width/2,center[1]+chip_width/2, c = "red", marker = "+", s = 1000, linewidths= 1)
                ax.scatter(center[0]-chip_width/2,center[1]+chip_width/2, c = "red", marker = "+", s = 1000, linewidths= 1)
                ax.scatter(center[0]+chip_width/2,center[1]-chip_width/2, c = "red", marker = "+", s = 1000, linewidths= 1)
                ax.scatter(center[0]-chip_width/2,center[1]-chip_width/2, c = "red", marker = "+", s = 1000, linewidths= 1) 
            return ax    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 