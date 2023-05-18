
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import os
from astropy.io import fits
from pathlib import Path
from os.path import exists
from matplotlib.offsetbox import AnchoredText
import datetime
import shutil
import sys
from tqdm import tqdm
import logging
import math
import fnmatch
from matplotlib.offsetbox import AnchoredText
from soxs.instrument import RedistributionMatrixFile
from threeML import *
from threeML.io.package_data import get_path_of_data_file
from astromodels.xspec.factory import XS_bapec,XS_vapec, XS_bvapec, XS_apec, XS_TBabs
from astromodels.xspec.xspec_settings import *
import warnings
warnings.simplefilter("ignore")
silence_warnings()
set_threeML_style()
from threeML import silence_progress_bars, activate_progress_bars, toggle_progress_bars
from threeML.utils.progress_bar import trange
from threeML import quiet_mode, loud_mode, silence_logs, update_logging_level

''' DO NOT INCLUDE EVTS ANYWHERE HERE> ONLY PHA FILES ARE DEPROJECTED AND NOISE-CORRECTED'''

class FitRadialSpectra():
    '''
    Class for fitting the spectra products produced by the PostProcess class. For this we use threeML and fit using XSPEC models from astromodels. Conveniently, this means that we do not actually have to install XSPEC/Heasoft; see https://threeml.readthedocs.io/en/stable/notebooks/installation.html
    The final data product will be saved under {save_dir}/{run_ID}/instrument_name/ANNULI/spectral_fits/ in the form of .npy file containing a list of dictionary quantities ordered by annulus number, with each dictionary containing the radius, and fit parameter values, for a given annulus. 
    ------------------------------------------------
    Constructor Positional Arguments:
                    observation: An instance of the Observation class. Many attributes of the observation class are faux-inherited by FitRadialSpectra so that we use the correct energies, redshift etc... We also inherit the active instruments and active halos from the Observation. NOTE; this means that you will need to update the active instruments/active halos if you want them to match those used in the deprojection stage if different from those used for the initial observation.
    Returns: FitRadialSpectra object
        
    '''
    def __init__(self,observation):
        self._redshift = observation.redshift
        self.hubble = observation.hubble
        if self.hubble == "from_box":
            self._logger.error(f"You set the hubble param to 'from_box' but you did not load the box!")
            sys.exit()
        self.cosmo = FlatLambdaCDM(H0 = 100*self.hubble, Om0 = 0.3, Ob0 = 0.048, Tcmb0=2.7 )
        xspec_cosmo(H0=100*self.hubble, q0=0.1, lambda_0=70.0)
        self._run_ID = observation._run_ID
        self._top_save_path = observation._top_save_path
        self.emin = observation.emin
        self.emax = observation.emax

        self._logger = logging.getLogger("MOXHA")        
        if (self._logger.hasHandlers()):
            self._logger.handlers.clear()       
        c_handler = logging.StreamHandler()
        c_handler.setLevel(level = logging.INFO)
        self._logger.setLevel(logging.INFO)
        c_format = logging.Formatter('%(name)s - [%(levelname)-8s] --- %(asctime)s  - %(message)s')
        c_handler.setFormatter(c_format)
        self._logger.addHandler(c_handler)
        self._logger.info("Spectral Fitting Initialised")
        self.active_halos = observation.active_halos
        self.active_instruments = observation.active_instruments
        self._instrument_defaults = observation._instrument_defaults
        
        
    def add_instrument(self, Name, exp_time, ID = None, reblock=1, aim_shift=None, chip_width=None):
        '''
        Function for adding instruments to the FitRadialSpectra object
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
        self.active_instruments.append({"Name":Name,"ID":ID, "exp_time": exp_time, "reblock":reblock, "aim_shift":aim_shift, "chip_width":250 })
            
            
    def clear_instruments(self):
        '''
        Clear all currently active FitRadialSpectras instruments. Should be an instrument with which you've carried out PostProcessing on.
        ------------------------------------------------
        Returns:
        '''
        self.active_instruments = []        
        
    def set_active_halos(self, halos):
        '''
        Set active halos for the FitRadialSpectra object. Should be in the set of halos for which you've carried out PostProcessing on.
        ------------------------------------------------
        Positional Arguments:
                    halos: Dictionary or list of dictionaries where each dictionary should contain at a minimum
                    { "index": halo_idx}
        Returns:
        '''
        if isinstance(halos, float) or isinstance(halos,int):
            self.active_halos = [halos,]
        else:
            self.active_halos = halos        
    
        
        

    def fit_spectra(self, fit_emin, fit_emax, n_max = 100, n_min = 4, overwrite = False ):
        '''
        Carry out the spectral fits on the annuli using threeML for the currently active instruments and halos. The annuli files will be found automatically under {save_dir}/{run_ID}/instrument_name/ANNULI/spectral_fits/fits_and_phas/.
        ------------------------------------------------
        Positional Arguments:
                    n_max: The maximum number of annuli to fit. This can save time for large clusters where you may have lots of annuli and you don't want to fit them all. We will include the extremal annuli and so the actual number of fits may be more by 1 or 2. Default = 15
                    n_min: Don't fit for an observation if there are fewer than n_min deprojected annuli availble. Default = 4
        Returns:
        '''
        for i,halo in enumerate(self.active_halos):
            print(halo)
            halo_idx = halo["index"]
            self.fit_emin = fit_emin
            self.fit_emax = fit_emax
            self.idx_tag = f"{self._run_ID}_h{str(halo_idx).zfill(3)}"
            self._obs_log = self._top_save_path/"LOGS"/f"obs_log_{self.idx_tag}.log"
            for self._instrument in self.active_instruments:
                instrument_name = self._instrument["Name"]
                print(instrument_name)
                if "ID" not in list(self._instrument.keys()):
                    self._instrument_ID = instrument_name
                else:
                    self._instrument_ID = self._instrument["ID"]
                self.idx_instr_tag = f"{self.idx_tag}_{self._instrument_ID}"
                self.evts_path = Path(self._top_save_path/instrument_name/"OBS"/self.idx_instr_tag)
                self.spectra_path = Path(self._top_save_path/instrument_name/"SPECTRA"/self.idx_instr_tag)
                os.makedirs(self.spectra_path, exist_ok = True)  
                os.makedirs(self.spectra_path/"SPECTRAL_FITS", exist_ok = True) 
                
                self.annuli_path = Path(self._top_save_path/instrument_name/"ANNULI"/self.idx_instr_tag)
                if os.path.exists(f"{self.annuli_path}/DATA/{self.idx_instr_tag}_fitted_data.npy") and not overwrite:
                    self._logger.info(f"{self.annuli_path}/DATA/{self.idx_instr_tag}_fitted_data.npy already exists and overwrite == False, so we will skip spectral fitting.")
                    continue
                os.makedirs(self.annuli_path/"spectral_fits/", exist_ok = True)
                os.makedirs(f"{self.annuli_path}/DATA", exist_ok = True)
                # accepted_candidate = f"{self.annuli_path}/{self.idx_instr_tag}_{str(self._annulus_number).zfill(2)}_deprojected"
                self._logger.info(f"Auto-finding Deprojected Spectrums under {self.annuli_path}...")
                try:
                    deprojected_annuli = sorted([x.split("_")[-2] for x in fnmatch.filter(os.listdir(f"{self.annuli_path}/fits_and_phas"), f"{self.idx_instr_tag}_*_deprojected.pha")])
                    self._logger.info(f"For {self.idx_instr_tag} I found {len(deprojected_annuli)} deprojected annuli: {','.join(deprojected_annuli)}")
                except Exception as e:
                    self._logger.error(e)
                    continue
                if len(deprojected_annuli)< n_min:
                    self._logger.error(f"Number of deprojected annuli {len(deprojected_annuli)} < n_min {n_min}")
                    f = open(f"{self.annuli_path}/DATA/{self.idx_instr_tag}_reason_for_no_fits.txt", "a")
                    f.write(f"\n {self.idx_instr_tag}: Number of deprojected annuli {len(deprojected_annuli)} < n_min {n_min}")
                    f.close()
                    continue                    
                
                step = int(-(len(deprojected_annuli)//-n_max))
                deprojected_annuli = sorted(list(set([*deprojected_annuli[::step], deprojected_annuli[-1]])))
                self._logger.info(f"n_max = {n_max} so will include extremal annuli and fit spectra for {len(deprojected_annuli)} annuli: {','.join(sorted(deprojected_annuli))}")
                self._logger.info(f"3ML cosmology: {xspec_cosmo()}")
                self._fit_values = {"r":[], "kT":[], "K":[], "abund":[]}
                self.fit_value_arrays = []
                self._kpc_radii = np.load(f"{self.annuli_path}/DATA/{self.idx_instr_tag}_kpc_radii.npy", allow_pickle = True)
                for n, self.annulus in enumerate(deprojected_annuli): #tqdm(enumerate(reversed(deprojected_annuli[0:-4])),desc = f"MOXHA - [    INFO] ---{''.ljust(26,' ')}- Fitting Annuli for {self.idx_instr_tag}".ljust(30,' '), total = len(deprojected_annuli)):
                    self._logger.info(f"Proceeding with fit for {self.idx_instr_tag}, annulus {self.annulus} ...")
                    self._pha_file = f"{self.annuli_path}/fits_and_phas/{self.idx_instr_tag}_{str(self.annulus).zfill(2)}_deprojected.pha"
                    if n == 0:
                        self._load_model()
                    try:
                        self._load_ogip()
                    except Exception as e:
                        self._logger.warning("something has gone wrong with the fitting...")
                        self._logger.warning(e)
                        continue
                    self._logger.info(f"Passing spectrum File {self._pha_file} to 3ML...")
                    self._fit_with_3ML()
                np.save(f"{self.annuli_path}/DATA/{self.idx_instr_tag}_fitted_data.npy", self.fit_value_arrays)
                    
                    
                    
            
    def _fit_with_3ML(self, test_areascal=False):    

        try:
            jl = JointLikelihood(self._model, DataList(self.ogip_data))  
            result = jl.fit( n_samples=100000)
            self._logger.info(f"Completed fit for {self.idx_instr_tag}, annulus {self.annulus} ")
            self._results = jl.results
        except Exception as e:
            self._logger.warning(e)
            return
        self._record_fit_results()
        

    def _load_model(self,):
        
        desired_model = "APEC_tbabs"
        print(f"Using model {desired_model}")
        if desired_model == "APEC_tbabs":
            modapec = APEC() #XS_bapec() #XS_bapec() #APEC() #XS_bvapec()
            modTbAbs = XS_TBabs()

            modapec.kT.min_value = 0.01
            modapec.kT.delta = 0.01
            modapec.abund.fix = False
            modapec.abund.max_value = 10
            modapec.abund.min_value = 0.01
            modapec.abund.delta = 0.01

            modapec.redshift.value = self._redshift # Source redshift
            modapec.redshift.fix = True  
            modTbAbs.nh.value = 0.018 # A value of 1 corresponds to 1e22 cm-2
            modTbAbs.nh.fix = True # NH is fixed   

            absorbed_apec = modapec*modTbAbs
            absorbed_apec.display()
            pts = PointSource("mysource", 30, 45, spectral_shape=absorbed_apec)
            self._model = Model(pts)           
        
        
        
        
        
        if desired_model == "vapec_tbabs":
            modapec = XS_vapec() #APEC() #XS_bapec() #XS_bapec() #APEC() #
            modTbAbs = XS_TBabs()
            modapec.kt.value = 0.3
            modapec.kt.min_value = 0.01
            modapec.kt.delta = 0.01

            for elem in ["He", "C", "N", "O", "Ne", "Mg","Al", "Si", "S", "Ar", "Ca", "Fe", "Ni"]:
                modapec[elem.lower()].fix = False
                modapec[elem.lower()].value = 1
                modapec[elem.lower()].max_value = 3
                modapec[elem.lower()].min_value = 0.01
                modapec[elem.lower()].delta = 0.01             
            modapec.redshift.value = round(self._redshift,2) # Source redshift
            modapec.redshift.fix = True  
            modTbAbs.nh.value = 0.018 # A value of 1 corresponds to 1e22 cm-2
            modTbAbs.nh.fix = True # NH is fixed   

            absorbed_apec = modapec*modTbAbs
            absorbed_apec.display()
            pts = PointSource("mysource", 30, 45, spectral_shape=absorbed_apec)
            self._model = Model(pts) 
            
        if desired_model == "apec_apec_tbabs":
            modapec1 = XS_apec() #APEC() #XS_bapec() #XS_bapec() #APEC() #
            modapec2 = XS_apec()
            modTbAbs = XS_TBabs()
            
            modapec1.kt.value = 0.3
            modapec1.kt.min_value = 0.01
            modapec1.kt.delta = 0.01
            modapec2.kt.value = 0.3
            modapec2.kt.min_value = 0.01
            modapec2.kt.delta = 0.01
         
            modapec1.redshift.value = round(self._redshift,2) # Source redshift
            modapec2.redshift.value = round(self._redshift,2) # Source redshift
            modapec1.redshift.fix = True 
            modapec2.redshift.fix = True 
            
            modTbAbs.nh.value = 0.018 # A value of 1 corresponds to 1e22 cm-2
            modTbAbs.nh.fix = True # NH is fixed   

            absorbed_apec = (modapec1+modapec2)*modTbAbs
            absorbed_apec.display()
            pts = PointSource("mysource", 30, 45, spectral_shape=absorbed_apec)
            self._model = Model(pts)                       
                    
                    
    def _load_ogip(self,):
        with fits.open(self._pha_file) as hdul:
            rmf = hdul["SPECTRUM"].header.get("RESPFILE", None)
            arf = hdul["SPECTRUM"].header.get("ANCRFILE", None)

        rmf = f"./CODE/instr_files/{rmf}"
        arf = f"./CODE/instr_files/{arf}"
        
        if not os.path.exists(rmf):
            raise RuntimeError(f"rmf file {rmf} does not exist!")
        if not os.path.exists(arf):
            raise RuntimeError(f"arf file {arf} does not exist!")
        
        self._logger.info(f" Fitting using rmf: {rmf}")
        self._logger.info(f" Fitting using arf: {arf}")
        
        try:
            self.ogip_data = OGIPLike("ogip",observation=self._pha_file,
                    response=rmf,arf_file = arf)
        except Exception as e:
            self._logger.warning(e)

        self._logger.info(f"Will fit between {self.fit_emin}-{self.fit_emax}")
        self.ogip_data.remove_rebinning()
        self.ogip_data.set_active_measurements(f"{self.fit_emin}-{self.fit_emax}")
        self.ogip_data.rebin_on_source(500)
                
                    
                    
    def _record_fit_results(self,):
        annulus_params = {}
        results_dict = self._results.get_data_frame(error_type='equal tail', cl=0.68).to_dict(orient='index')
        radii = [x for x in self._kpc_radii if int(x["annulus_num"]) == int(self.annulus)][0]["radii"]
        annulus_params["kT"] = results_dict['mysource.spectrum.main.composite.kT_1']
        annulus_params["norm"] = results_dict['mysource.spectrum.main.composite.K_1']
        annulus_params["radii"] = radii      
        self.fit_value_arrays.append(annulus_params)  
        count_fig1 = self.ogip_data.display_model(step=False)
        plt.xlim(0.8*self.fit_emin, 1.2*self.fit_emax)
        plt.xscale("linear")

        at = AnchoredText(
            f"log10 kT = {round(np.log10(self._model.mysource.spectrum.main.composite.kT_1.value),3)}, log10 K = {round(np.log10(self._model.mysource.spectrum.main.composite.K_1.value),3)}", prop=dict(size=8), frameon=True, loc='lower right')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0")
        ax = plt.gca()
        ax.add_artist(at)        
        self._logger.info(f"Saving Fitting Plot under {self.annuli_path}/spectral_fits/{self.idx_instr_tag}_{str(self.annulus).zfill(2)}_deprojected_fit_new.png")
        plt.savefig(f"{self.annuli_path}/spectral_fits/{self.idx_instr_tag}_{str(self.annulus).zfill(2)}_deprojected_fit_new.png")
        plt.clf()
        plt.close() 
        self._logger.info(f"Spectral fit saved under {self.annuli_path}/spectral_fits/{self.idx_instr_tag}_{str(self.annulus).zfill(2)}_deprojected_fit")