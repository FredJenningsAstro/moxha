from scipy.optimize import minimize
import astropy.io.fits as pyfits
from scipy.signal import savgol_filter
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import emcee
import logging
from tqdm import tqdm
from IPython.display import display, Math
import os
import math
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import caesar
from scipy import odr
import corner
import scipy.stats as st
import pathlib
from pathlib import Path
import h5py
import unyt
from tqdm import tqdm
from lmfit import Model, Parameter, minimize, Parameters, fit_report
import lmfit
from scipy import integrate
from scipy.signal import find_peaks
from scipy.ndimage import median_filter, percentile_filter
import random
# import corner
import pandas as pd
import shutil
import scipy
from matplotlib.offsetbox import AnchoredText   
from astropy.constants import G, m_p, k_B
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm
from matplotlib.ticker import MultipleLocator



class HaloAnalysis:
    
    def __init__(self, load_dir , run_ID,  snap_num, R500_truth, M500, redshift, emin, emax, rho_crit = None, h = None, min_length = 5, inner_R500_frac = 0.1, outer_R500_frac = 2.5, max_chisqr = 50, do_MCMC = True, nsteps = 20000, nwalkers = 100, nburn = 1000, thin = 200, num_samples_for_errors = 100, remove_outermost = 0, monotonic_penalty = 1e10):
        
        self.remove_outermost = remove_outermost
        self._logger = logging.getLogger("MOXHA")
        if (self._logger.hasHandlers()):
            self._logger.handlers.clear()     
        c_handler = logging.StreamHandler()
        c_handler.setLevel(level = logging.INFO)
        self._logger.setLevel(logging.INFO)
        c_format = logging.Formatter('%(name)s - [%(levelname)8s] --- %(asctime)s  - %(message)s')
        c_handler.setFormatter(c_format)
        self._logger.addHandler(c_handler)

        self._top_save_path = Path(f"{load_dir}/{run_ID}")
        self.run_ID = f"{run_ID}_sn{str(snap_num).zfill(3)}"
        self.snap_num = snap_num
        self.R500_truth = R500_truth.to("kpc")
        self.M500_truth = M500.to("solMass")
        self.min_length = min_length
        self.inner_R500_frac = inner_R500_frac
        self.outer_R500_frac = outer_R500_frac
        self.min_radius = inner_R500_frac*self.R500_truth
        self.max_radius = outer_R500_frac*self.R500_truth
        self.max_chisqr = max_chisqr
        self.redshift = redshift 
        self.emin = emin
        self.emax = emax
        if h == None:
            self._logger.warning("h not set. Will use h=0.68")
            self.hubble = 0.68
        else:
            self.hubble = h
            
        self.cosmo = FlatLambdaCDM(H0 = 100*self.hubble, Om0 = 0.3, Ob0 = 0.048)    
        if rho_crit == None:
            self.rho_crit = self.cosmo.critical_density(self.redshift).to("solMass/kpc**3")
            self._logger.warning(f"Critical density not provided. Using Flat LCDM cosmology at z={round(self.redshift,2)} to calculate rho_crit = {self.rho_crit}.")  
        else:
            self.rho_crit = rho_crit.to("solMass/kpc**3")
            
        self.do_MCMC = do_MCMC
        self.emcee_nsteps = nsteps
        self.emcee_walkers = nwalkers
        self.emcee_burn = nburn
        self.emcee_thin = thin
        self._mcmc_mass_samples = int(num_samples_for_errors)
        print(f"Chain length = {nsteps}, with burn-in = {nburn}, and thinning = {thin}, and {nwalkers} walkers")
        print(f"num_mass_samples = {self._mcmc_mass_samples}")
        self.marker_colour = "black"
        
        self.r_values_fine = np.linspace(self.min_radius.to('kpc').value, self.max_radius.to('kpc').value, 1000) * u.kpc
        self.monotonic_penalty = monotonic_penalty
        with open('./halo_fitting_record.out','a') as f:
            pass
        
        
        
    def load_data(self, halo_idx, instrument_ID, instrument_name, chip_rad_arcmin = None):
        self._chip_rad_arcmin = chip_rad_arcmin
        
        self.halo_idx = halo_idx
        self.instrument_name = instrument_name
        self.idx_tag = f"{self.run_ID}_h{str(self.halo_idx).zfill(3)}"
        self.idx_instr_tag = f"{self.idx_tag}_{instrument_ID}"
        self.annuli_path = Path(self._top_save_path/self.instrument_name/"ANNULI"/self.idx_instr_tag)
        self.evts_path = Path(self._top_save_path/self.instrument_name/"OBS"/self.idx_instr_tag)
        
        self._bkgrnd_idx_tag = f"{self.run_ID}_blanksky{str(0).zfill(2)}"
        self._bkgrnd_idx_instr_tag = f"{self._bkgrnd_idx_tag}_{instrument_ID}"
        self._bkgrnd_evts_path = Path(self._top_save_path/self.instrument_name/"OBS"/self._bkgrnd_idx_instr_tag)
        
        
        try:
            data_arr = np.load( f"{self.annuli_path}/DATA/{self.idx_instr_tag}_fitted_data.npy"  , allow_pickle = True)
        except Exception as e:
            with open('./halo_fitting_record.out','a') as f:
                f.write(f"\n {e}")
                raise NameError()

        # print(data_arr)
        self.shell_radii = [x["radii"] for x in data_arr]
        self.shell_radii = [[x[0].value,x[1].value] for x in self.shell_radii] * self.shell_radii[0][0].unit
        self.shell_radii = self.shell_radii.to("kpc")

        self.r =[ (x[0] + x[1])/2  for x in self.shell_radii]
        self.r = [x.value for x in self.r] * self.r[0].unit       
        self.r = self.r.to("kpc")
        
        self.r_err = [ [(x[1]-x[0])/2,(x[1]-x[0])/2 ] for x in self.shell_radii]
        self.r_err = [[x[0].value,x[1].value] for x in self.r_err] * self.r_err[0][0].unit
        self.r_err = self.r_err.to("kpc")
        
        self.ne_norms = np.array([x["norm"]["value"] for x in data_arr])
        self.norm_bounds = np.array([ [x["norm"]["negative_error"],x["norm"]["positive_error"]] for x in data_arr]) 
        
        self.ne_y = [self._norm_to_ne(self.ne_norms[i], self.shell_radii[i][0], self.shell_radii[i][1]) for i,_ in enumerate(self.ne_norms)]
        self.ne_y = [x.value for x in self.ne_y] * self.ne_y[0].unit
        self.ne_y = self.ne_y.to("cm**-3")
        
        self.ne_yerr = [ [ self._ne_calc_error(self.norm_bounds[i][0],self.shell_radii[i][0], self.shell_radii[i][1], self.ne_norms[i]),  self._ne_calc_error(self.norm_bounds[i][1],self.shell_radii[i][0], self.shell_radii[i][1], self.ne_norms[i]) ]   for i,_ in enumerate(self.ne_norms) ]
        self.ne_yerr = [ [x[0].value, x[1].value ] for x in self.ne_yerr] * self.ne_yerr[0][0].unit
        self.ne_yerr = self.ne_yerr.to("cm**-3")
        
        
        self.kT_y = [x["kT"]["value"] for x in data_arr] * data_arr[0]["kT"]["unit"]
        self.kT_y = self.kT_y.to("keV")
        
        self.kT_yerr = [ [x["kT"]["negative_error"],x["kT"]["positive_error"]] for x in data_arr] * data_arr[0]["kT"]["unit"]
        self.kT_yerr = self.kT_yerr.to("keV")
     
        ### WE MUST BE CAREFUL TO USE THE FULL PROJECTED LUMINOSITY SO WE COLLLECT ALL THE RADIATED ENERGY FROM ALL SHELLS
        # self.lumins = [x[0] * 10**44 for x in np.load(self.halo_path / f"{self.idx_instr_tag }__bapec_lumin_tuple_raw.npy", allow_pickle=True)]

        included_idxs = tuple([ (np.isnan(self.kT_y) != True) & (np.isnan(self.ne_y) != True) & (self.r > self.min_radius) & (self.r < self.max_radius)])
        

        self.r = np.flip(self.r[included_idxs])
        self.shell_radii = np.flip(self.shell_radii[included_idxs])
        self.r_err = np.flip(self.r_err[included_idxs])
        self.ne_y = np.flip(self.ne_y[included_idxs])
        self.ne_norms = np.flip(self.ne_norms[included_idxs])
        self.ne_yerr = np.flip(self.ne_yerr[included_idxs])
        self.kT_y = np.flip(self.kT_y[included_idxs])
        self.kT_yerr = np.flip(self.kT_yerr[included_idxs])        
        # self.lumins = np.flip(np.array(self.lumins)[included_idxs]) 
        
        
        if self.remove_outermost != 0:
            self.r = self.r[0:-self.remove_outermost]
            self.shell_radii = self.shell_radii[0:-self.remove_outermost]
            self.r_err = self.r_err[0:-self.remove_outermost]
            self.ne_y = self.ne_y[0:-self.remove_outermost]
            self.ne_norms = self.ne_norms[0:-self.remove_outermost]
            self.ne_yerr = self.ne_yerr[0:-self.remove_outermost]
            self.kT_y = self.kT_y[0:-self.remove_outermost]
            self.kT_yerr = self. kT_yerr[0:-self.remove_outermost]           
        
        self._check_data()        

        
        
        
        
        
        
    def load_yT_data(self,  halo_idx, emin, emax, emin_for_total_Lx = None, emax_for_total_Lx = None,  smooth = False):
        self.halo_idx = halo_idx
        self.idx_tag = f"{self.run_ID}_h{str(self.halo_idx).zfill(3)}"
        self.emin_for_EW_values = emin
        self.emax_for_EW_values = emax
        if emin_for_total_Lx == None: emin_for_total_Lx = emin
        if emax_for_total_Lx == None: emax_for_total_Lx = emax
        
        self.emin_for_total_Lx = emin_for_total_Lx
        self.emax_for_total_Lx = emax_for_total_Lx
        
        
        yt_data_path = Path(self._top_save_path/"YT_DATA"/self.idx_tag)
        
        self.yT_path = pathlib.Path(f"{yt_data_path}/{self.idx_tag}_yt_data_ver3.npy")
        self.yT_data = np.load(self.yT_path, allow_pickle = True)
        
        
        self.MW_kT_x = [x["radius"] for x in self.yT_data if x["Name"] == f"kT_filtGasMassW" ][0]
        self.MW_kT_x = [x.value for x in self.MW_kT_x] * self.MW_kT_x[0].unit
        self.MW_kT_x = self.MW_kT_x.to("kpc")

        self.MW_kT_y = [x["values"] for x in self.yT_data if x["Name"] == f"kT_filtGasMassW" ][0]
        self.MW_kT_y = [x.value for x in self.MW_kT_y] * self.MW_kT_y[0].unit 
        self.MW_kT_y = self.MW_kT_y.to(u.keV, equivalencies=u.temperature_energy())    

        included_idxs = tuple([  (self.MW_kT_x > self.min_radius) & (self.MW_kT_x < self.max_radius)])
        self.MW_kT_x = self.MW_kT_x[included_idxs]
        self.MW_kT_y = self.MW_kT_y[included_idxs]
        
        self.MW_ne_x = [x["radius"] for x in self.yT_data if x["Name"] == f"ne_filtGasMassW" ][0]
        self.MW_ne_x = [x.value for x in self.MW_ne_x] * self.MW_ne_x[0].unit
        self.MW_ne_x = self.MW_ne_x.to("kpc")

        self.MW_ne_y = [x["values"] for x in self.yT_data if x["Name"] == f"ne_filtGasMassW" ][0]
        self.MW_ne_y = [x.value for x in self.MW_ne_y] * self.MW_ne_y[0].unit 
        self.MW_ne_y = self.MW_ne_y.to("cm**-3")
        
        included_idxs = tuple([  (self.MW_ne_x > self.min_radius) & (self.MW_ne_x < self.max_radius)])
        self.MW_ne_x = self.MW_ne_x[included_idxs]
        self.MW_ne_y = self.MW_ne_y[included_idxs]

        self.EW_kT_x = [x["radius"] for x in self.yT_data if x["Name"] == f"kT_EW_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV" ][0]
        self.EW_kT_x = [x.value for x in self.EW_kT_x] * self.EW_kT_x[0].unit
        self.EW_kT_x = self.EW_kT_x.to("kpc")

        self.EW_kT_y = [x["values"] for x in self.yT_data if x["Name"] == f"kT_EW_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV" ][0]
        self.EW_kT_y = [x.value for x in self.EW_kT_y] * self.EW_kT_y[0].unit 
        self.EW_kT_y = self.EW_kT_y.to(u.keV, equivalencies=u.temperature_energy())    

        included_idxs = tuple([  (self.EW_kT_x > self.min_radius) & (self.EW_kT_x < self.max_radius)])
        self.EW_kT_x = self.EW_kT_x[included_idxs]
        self.EW_kT_y = self.EW_kT_y[included_idxs]
        
        self.EW_ne_x = [x["radius"] for x in self.yT_data if x["Name"] == f"ne_EW_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV" ][0]
        self.EW_ne_x = [x.value for x in self.EW_ne_x] * self.EW_ne_x[0].unit
        self.EW_ne_x = self.EW_ne_x.to("kpc")

        self.EW_ne_y = [x["values"] for x in self.yT_data if x["Name"] == f"ne_EW_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV" ][0]
        self.EW_ne_y = [x.value for x in self.EW_ne_y] * self.EW_ne_y[0].unit 
        self.EW_ne_y = self.EW_ne_y.to("cm**-3")
        
        included_idxs = tuple([  (self.EW_ne_x > self.min_radius) & (self.EW_ne_x < self.max_radius)])
        self.EW_ne_x = self.EW_ne_x[included_idxs]
        self.EW_ne_y = self.EW_ne_y[included_idxs]

        self.LW_kT_x = [x["radius"] for x in self.yT_data if x["Name"] == f"kT_LuminW_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV" ][0]
        self.LW_kT_x = [x.value for x in self.LW_kT_x] * self.LW_kT_x[0].unit
        self.LW_kT_x = self.LW_kT_x.to("kpc")

        self.LW_kT_y = [x["values"] for x in self.yT_data if x["Name"] == f"kT_LuminW_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV" ][0]
        self.LW_kT_y = [x.value for x in self.LW_kT_y] * self.LW_kT_y[0].unit 
        self.LW_kT_y = self.LW_kT_y.to(u.keV, equivalencies=u.temperature_energy())    

        included_idxs = tuple([  (self.LW_kT_x > self.min_radius) & (self.LW_kT_x < self.max_radius)])
        self.LW_kT_x = self.LW_kT_x[included_idxs]
        self.LW_kT_y = self.LW_kT_y[included_idxs]
        
        self.LW_ne_x = [x["radius"] for x in self.yT_data if x["Name"] == f"ne_LuminW_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV" ][0]
        self.LW_ne_x = [x.value for x in self.LW_ne_x] * self.LW_ne_x[0].unit
        self.LW_ne_x = self.LW_ne_x.to("kpc")

        self.LW_ne_y = [x["values"] for x in self.yT_data if x["Name"] == f"ne_LuminW_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV" ][0]
        self.LW_ne_y = [x.value for x in self.LW_ne_y] * self.LW_ne_y[0].unit 
        self.LW_ne_y = self.LW_ne_y.to("cm**-3")
        
        included_idxs = tuple([  (self.LW_ne_x > self.min_radius) & (self.LW_ne_x < self.max_radius)])
        self.LW_ne_x = self.LW_ne_x[included_idxs]
        self.LW_ne_y = self.LW_ne_y[included_idxs]        
        

        self.yT_profile_mass_x = [x["radius"] for x in self.yT_data if x["Name"] == "Total_Mass" ][0]
        self.yT_profile_mass_x = [x.value for x in self.yT_profile_mass_x] * self.yT_profile_mass_x[0].unit
        self.yT_profile_mass_x = self.yT_profile_mass_x.to("kpc")

        self.yT_profile_mass_y = [x["values"] for x in self.yT_data if x["Name"] == "Total_Mass" ][0]
        self.yT_profile_mass_y = [x.value for x in self.yT_profile_mass_y] * self.yT_profile_mass_y[0].unit 
        self.yT_profile_mass_y = self.yT_profile_mass_y.to("solMass")
        
        included_idxs = tuple([  (self.yT_profile_mass_x > self.min_radius) & (self.yT_profile_mass_x < self.max_radius)])
        self.yT_profile_mass_x = self.yT_profile_mass_x[included_idxs]
        self.yT_profile_mass_y = self.yT_profile_mass_y[included_idxs]   
        
        self.Lx_in_R500_truth = [ x["value"] for x in self.yT_data if x["Name"] == f"total_Lx_in_R500_{self.emin_for_total_Lx}_{self.emax_for_total_Lx}_keV"][0]
        
        print(f"Total luminosity field is -- total_Lx_in_R500_{self.emin_for_total_Lx}_{self.emax_for_total_Lx}_keV" )
        
        assert( self.MW_kT_x.unit == u.kpc )
        assert( self.MW_ne_x.unit == u.kpc )
        assert( self.MW_kT_y.unit == u.keV )
        assert( self.MW_ne_y.unit == u.cm**-3 )
        
        assert( self.EW_kT_x.unit == u.kpc )
        assert( self.EW_ne_x.unit == u.kpc )
        assert( self.EW_kT_y.unit == u.keV )
        assert( self.EW_ne_y.unit == u.cm**-3 )
        
        assert( self.LW_kT_x.unit == u.kpc )
        assert( self.LW_ne_x.unit == u.kpc )
        assert( self.LW_kT_y.unit == u.keV )
        assert( self.LW_ne_y.unit == u.cm**-3 )
        
        assert( self.yT_profile_mass_x.unit == u.kpc )
        assert( self.yT_profile_mass_y.unit == u.solMass )   
        assert( self.Lx_in_R500_truth.unit == u.erg/u.s)
        

        
    def _check_data(self):
        if self._chip_rad_arcmin != None:
            chip_width_kpc = (self._chip_rad_arcmin*(u.arcmin/u.radian)*self.cosmo.angular_diameter_distance(z = self.redshift)).to("kpc").value
            if self.R500_truth.to('kpc').value > 1.2 * chip_width_kpc:
                raise ValueError(f"True R500 > 1.2 * Chip radius")  
                
        if max(self.r) < 0.45*self.R500_truth:
            raise ValueError(f"Max Radius is too small = {round(max(self.r.value/self.R500_truth.value),3)} < 0.45")   
        if len(self.r) < self.min_length:
            with open('./halo_fitting_record.out','a') as f:
                f.write(f"\n {self.halo_idx}: Number of data points within R500 {len(self.r)} < min_length" )
            raise ValueError(f'Number of data points within R500 {len(self.r)} < min_length')
        elif len(self.r) >= 7: 
            self._models_to_use = "full"
            self._logger.info(f"Enough data points {len(self.r)} for full model")
        elif len(self.r) < 7 and len(self.r) >= 4:
            self._models_to_use = "reduced_params"
            self._logger.info(f"Too few data points {len(self.r)} for full model but enough for reduced")
        else:
            with open('./halo_fitting_record.out','a') as f:
                f.write(f"\n {self.halo_idx}: Number of data points within R500 > min_length but too few points even for reduced model" )
            raise ValueError('Number of data points within R500 < min_length')   
        if min(self.r) > 0.3*self.R500_truth:
            with open('./halo_fitting_record.out','a') as f:
                f.write(f"\n {self.halo_idx}: Min Radius is too large = {min(self.r/self.R500_truth)}" )
            raise ValueError(f"Min Radius is too large = {min(self.r/self.R500_truth)}")
            
 
        
        assert( self.r.unit == u.kpc )
        assert( self.shell_radii.unit == u.kpc )
        assert( self.r_err.unit == u.kpc )
        assert( self.ne_y.unit == u.cm**-3 )
        assert( self.ne_yerr.unit == u.cm**-3 )
        assert( self.kT_y.unit == u.keV )
        assert( self.kT_yerr.unit == u.keV )
        
        
        

    def _ne_calc_error(self, norm_bound,  r_in, r_out, norm):
        assert((r_in.unit == u.kpc) &  (r_out.unit == u.kpc))
        D_a = self.cosmo.angular_diameter_distance(self.redshift).to(u.cm)
        V = ((4/3) * math.pi * ((r_out**2) - (r_in**2))**(3/2))
        norm = norm * (u.cm**-5)
        ne = np.sqrt(((10**14) * 4 *  math.pi * norm * (D_a * (1+self.redshift))**2 )/(0.82*V))
        ne_error = (ne/2) * (abs(norm_bound)/norm.value)
        return ne_error.to("cm**-3") 
    
    
    def _norm_to_ne(self, norm,  r_in, r_out):
        assert((r_in.unit == u.kpc) &  (r_out.unit == u.kpc))
        D_a = self.cosmo.angular_diameter_distance(self.redshift).to(u.cm)
        V = (4/3) * math.pi * ((r_out**2) - (r_in**2))**(3/2)
        norm = norm * (u.cm**-5)
        ne = np.sqrt(( (10**14) * 4 *  math.pi * norm * (D_a * (1+self.redshift))**2 )/(0.82*V))
        return ne.to("cm**-3")     
    
    
    
    
    
    def joint_fit_MW_kT_ne(self, kT_redchi_thresh = 0.2, ne_redchi_thresh = 0.2, smoothness_regularization_power = 0, kT_smoothness_radius_limit = None,):
        if  max(abs(self.MW_ne_x.value/self.MW_kT_x.value)) != 1 or min(abs(self.MW_ne_x.value/self.MW_kT_x.value)) != 1:
            print("kT x and ne x not same for yt profiles!")
            return

        self._logger.info(f"Currently fitting for halo {self.halo_idx}")
        self._fit_data_type = "profile"
        self._smoothness_regularization_power = smoothness_regularization_power
        self._kT_smoothness_radius_limit = kT_smoothness_radius_limit
        params = Parameters()
        params.add("log_kT_0", value = np.log10(1), min = -2, max = 2)
        params.add("log_kT_min", value = np.log10(0.3), min = -2, max = 2)
        params.add("log_r_cool", value = np.log10(1000), min = np.log10(200), max = np.log10(3000))
        params.add("log_r_t", value = np.log10(1000), min = np.log10(500), max = np.log10(5000) )
        params.add("a_cool", value = 5, max = 20, min = 1e-1)
        params.add("a",value = 0.3, max = 10, min = 1e-2)
        params.add("b", value = 5, max = 20, min = 1e-1 )
        params.add("c",value = 1.3, max = 10, min = 1e-2 )
        
        params.add("log_ne_0", value = -2, min = -4, max = 0)
        params.add("log_r_c", value = np.log10(150), min = np.log10(5), max = np.log10(800) )
        params.add("beta", value = 0.5, min = 0.1 , max = 2 )
        params.add("log_r_s", value = np.log10(700), min = np.log10(200), max = np.log10(4000) )
        # params.add("gamma", value = 3, min = 2.99 , max = 3 )
        params.add("eps", value = 1, min = 0 , max = 5 )        
        

        slice_n = len(self.MW_kT_y.value)//20
        self.yT_slice = slice_n
        log_yT_kT_y = np.log10(self.MW_kT_y.value)[::slice_n]
        log_yT_ne_y = np.log10(self.MW_ne_y.value)[::slice_n]
        log_x = np.log10(self.MW_ne_x.value)[::slice_n]
        # No errors on values measured from yT
        log_yT_kT_err = np.ones_like(log_x)
        log_yT_ne_err = np.ones_like(log_x)
        
        self._logger.info("Fitting for the Mass-Weighted Values")
        minim, self.MW_kT_model_best_fit_pars, self.MW_ne_model_best_fit_pars, log_x, log_yT_kT_y, log_yT_ne_y, log_yT_kT_yerr, log_yT_ne_yerr, self.MW_non_outlier_idxs = self._do_joint_minimization_with_leave_one_out(params, log_x, log_yT_kT_y, log_yT_ne_y, log_yT_kT_err, log_yT_ne_err, kT_redchi_thresh, ne_redchi_thresh) 
        # print("profile_non_outlier_idxs", self.MW_non_outlier_idxs)
        if self.do_MCMC:
            minim.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(5))
            result_emcee = lmfit.minimize(self._joint_log_kT_log_ne_minim, method='emcee', nan_policy='omit',
                     params=minim.params, is_weighted=False, progress=True, steps=self.emcee_nsteps, burn=self.emcee_burn, thin=self.emcee_thin, nwalkers = self.emcee_walkers, args = (log_x,), kws={"log_kT_y": log_yT_kT_y , "log_ne_y": log_yT_ne_y, "log_kT_yerr":np.ones_like(log_x),"log_ne_yerr":np.ones_like(log_x)})
            self.MW_joint_redchisqr = result_emcee.redchi
            flattened_chain = result_emcee.flatchain
            draw = np.floor(np.random.uniform(0,len(flattened_chain),size=self.emcee_nsteps))
            self.MW_kT_thetas = flattened_chain.iloc[draw].drop(["__lnsigma", *self.MW_ne_model_best_fit_pars.keys()] , axis = 1)
            self.MW_ne_thetas = flattened_chain.iloc[draw].drop(["__lnsigma", *self.MW_kT_model_best_fit_pars.keys()] , axis = 1)
            self.MW_kT_models = [self.kT_model(self.r_values_fine.value, **i) for i in self.MW_kT_thetas.to_dict(orient = 'records')]
            self.MW_ne_models = [self.ne_model(self.r_values_fine.value, **i) for i in self.MW_ne_thetas.to_dict(orient = 'records')]
            
            # self.MW_kT_spread = np.std(self.MW_kT_models,axis=0) * u.keV
            # self.MW_ne_spread = np.std(self.MW_ne_models,axis=0) * u.cm**-3   
            
            self.MW_kT_spread  = (abs(np.percentile(self.MW_kT_models,(16,84), axis=0) - self.kT_model(self.r_values_fine.value, **self.MW_kT_model_best_fit_pars,)) * u.keV).to("keV")
            self.MW_ne_spread  = (abs(np.percentile(self.MW_ne_models,(16,84), axis=0) - self.ne_model(self.r_values_fine.value, **self.MW_ne_model_best_fit_pars,)) * u.cm**-3).to("cm**-3")

        self.calculate_MW_derived_mass_profile()
        self._logger.info("Calculated MW Mass Profile")
        self.calculate_MW_derived_entropy()
        self._logger.info("Calculated MW S Profile")
        self.calculate_MW_derived_pressure()
        self._logger.info("Calculated MW Pressure Profile")
        self.calculate_weighted_kT_MW()
        self._logger.info("Calculated Weighted MW kT and kT(R500)")
        self.calculate_weighted_S_MW()
        self._logger.info("Calculated Weighted MW S")
        self.calculate_MW_mgas()
        self._logger.info("Calculated MW Mgas")
    
    
        
    def joint_fit_EW_kT_ne(self, kT_redchi_thresh = 0.2, ne_redchi_thresh = 0.2, smoothness_regularization_power = 0, kT_smoothness_radius_limit = None,):
        if  max(abs(self.EW_ne_x.value/self.EW_kT_x.value)) != 1 or min(abs(self.EW_ne_x.value/self.EW_kT_x.value)) != 1:
            print("kT x and ne x not same for yt profiles!")
            return

        self._logger.info(f"Currently fitting for halo {self.halo_idx}")
        self._fit_data_type = "profile"
        self._smoothness_regularization_power = smoothness_regularization_power
        self._kT_smoothness_radius_limit = kT_smoothness_radius_limit
        params = Parameters()
        params.add("log_kT_0", value = np.log10(1), min = -2, max = 2)
        params.add("log_kT_min", value = np.log10(0.3), min = -2, max = 2)
        params.add("log_r_cool", value = np.log10(1000), min = np.log10(200), max = np.log10(3000))
        params.add("log_r_t", value = np.log10(1000), min = np.log10(500), max = np.log10(5000) )
        params.add("a_cool", value = 5, max = 20, min = 1e-1)
        params.add("a",value = 0.3, max = 10, min = 1e-2)
        params.add("b", value = 5, max = 20, min = 1e-1 )
        params.add("c",value = 1.3, max = 10, min = 1e-2 )
        
        params.add("log_ne_0", value = -2, min = -4, max = 0)
        params.add("log_r_c", value = np.log10(150), min = np.log10(5), max = np.log10(800) )
        params.add("beta", value = 0.5, min = 0.1 , max = 2 )
        params.add("log_r_s", value = np.log10(700), min = np.log10(200), max = np.log10(4000) )
        # params.add("gamma", value = 3, min = 2.99 , max = 3 )
        params.add("eps", value = 1, min = 0 , max = 5 )       
        

        slice_n = len(self.EW_kT_y.value)//20
        self.yT_slice = slice_n
        log_yT_kT_y = np.log10(self.EW_kT_y.value)[::slice_n]
        log_yT_ne_y = np.log10(self.EW_ne_y.value)[::slice_n]
        log_x = np.log10(self.EW_ne_x.value)[::slice_n]
        # No errors on values measured from yT
        log_yT_kT_err = np.ones_like(log_x)
        log_yT_ne_err = np.ones_like(log_x)
        
        self._logger.info("Fitting for the Emission-Weighted Values")
        minim, self.EW_kT_model_best_fit_pars, self.EW_ne_model_best_fit_pars, log_x, log_yT_kT_y, log_yT_ne_y, log_yT_kT_yerr, log_yT_ne_yerr, self.EW_non_outlier_idxs = self._do_joint_minimization_with_leave_one_out(params, log_x, log_yT_kT_y, log_yT_ne_y, log_yT_kT_err, log_yT_ne_err, kT_redchi_thresh, ne_redchi_thresh) 
        # print("profile_non_outlier_idxs", self.EW_non_outlier_idxs)
        if self.do_MCMC:
            minim.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(5))
            result_emcee = lmfit.minimize(self._joint_log_kT_log_ne_minim, method='emcee', nan_policy='omit',
                     params=minim.params, is_weighted=False, progress=True, steps=self.emcee_nsteps, burn=self.emcee_burn, thin=self.emcee_thin, nwalkers = self.emcee_walkers, args = (log_x,), kws={"log_kT_y": log_yT_kT_y , "log_ne_y": log_yT_ne_y, "log_kT_yerr":np.ones_like(log_x),"log_ne_yerr":np.ones_like(log_x)})
            self.EW_joint_redchisqr = result_emcee.redchi
            flattened_chain = result_emcee.flatchain
            draw = np.floor(np.random.uniform(0,len(flattened_chain),size=self.emcee_nsteps))
            self.EW_kT_thetas = flattened_chain.iloc[draw].drop(["__lnsigma", *self.EW_ne_model_best_fit_pars.keys()] , axis = 1)
            self.EW_ne_thetas = flattened_chain.iloc[draw].drop(["__lnsigma", *self.EW_kT_model_best_fit_pars.keys()] , axis = 1)
            self.EW_kT_models = [self.kT_model(self.r_values_fine.value, **i) for i in self.EW_kT_thetas.to_dict(orient = 'records')]
            self.EW_ne_models = [self.ne_model(self.r_values_fine.value, **i) for i in self.EW_ne_thetas.to_dict(orient = 'records')]
            
            self.EW_kT_spread  = (abs(np.percentile(self.EW_kT_models,(16,84), axis=0) - self.kT_model(self.r_values_fine.value, **self.EW_kT_model_best_fit_pars,)) * u.keV).to("keV")
            self.EW_ne_spread  = (abs(np.percentile(self.EW_ne_models,(16,84), axis=0) - self.ne_model(self.r_values_fine.value, **self.EW_ne_model_best_fit_pars,)) * u.cm**-3).to("cm**-3")

        self.calculate_EW_derived_mass_profile()
        self._logger.info("Calculated EW Mass Profile")
        self.calculate_EW_derived_entropy()
        self._logger.info("Calculated EW S Profile")
        self.calculate_EW_derived_pressure()
        self._logger.info("Calculated EW Pressure Profile")
        self.calculate_weighted_kT_EW()
        self._logger.info("Calculated Weighted EW kT and kT(R500)")
        self.calculate_weighted_S_EW()
        self._logger.info("Calculated Weighted EW S")
        self.calculate_EW_mgas()
        self._logger.info("Calculated EW Mgas")
        
    def joint_fit_LW_kT_ne(self, kT_redchi_thresh = 0.2, ne_redchi_thresh = 0.2, smoothness_regularization_power = 0, kT_smoothness_radius_limit = None,):
        if  max(abs(self.LW_ne_x.value/self.LW_kT_x.value)) != 1 or min(abs(self.LW_ne_x.value/self.LW_kT_x.value)) != 1:
            print("kT x and ne x not same for yt profiles!")
            return

        self._logger.info(f"Currently fitting for halo {self.halo_idx}")
        self._fit_data_type = "profile"
        self._smoothness_regularization_power = smoothness_regularization_power
        self._kT_smoothness_radius_limit = kT_smoothness_radius_limit
        params = Parameters()
        params.add("log_kT_0", value = np.log10(1), min = -2, max = 2)
        params.add("log_kT_min", value = np.log10(0.3), min = -2, max = 2)
        params.add("log_r_cool", value = np.log10(1000), min = np.log10(200), max = np.log10(3000))
        params.add("log_r_t", value = np.log10(1000), min = np.log10(500), max = np.log10(5000) )
        params.add("a_cool", value = 5, max = 20, min = 1e-1)
        params.add("a",value = 0.3, max = 10, min = 1e-2)
        params.add("b", value = 5, max = 20, min = 1e-1 )
        params.add("c",value = 1.3, max = 10, min = 1e-2 )
        
        params.add("log_ne_0", value = -2, min = -4, max = 0)
        params.add("log_r_c", value = np.log10(150), min = np.log10(5), max = np.log10(800) )
        params.add("beta", value = 0.5, min = 0.1 , max = 2 )
        params.add("log_r_s", value = np.log10(700), min = np.log10(200), max = np.log10(4000) )
        # params.add("gamma", value = 3, min = 2.99 , max = 3 )
        params.add("eps", value = 1, min = 0 , max = 5 )     
        

        slice_n = len(self.LW_kT_y.value)//20
        self.yT_slice = slice_n
        log_yT_kT_y = np.log10(self.LW_kT_y.value)[::slice_n]
        log_yT_ne_y = np.log10(self.LW_ne_y.value)[::slice_n]
        log_x = np.log10(self.LW_ne_x.value)[::slice_n]
        # No errors on values measured from yT
        log_yT_kT_err = np.ones_like(log_x)
        log_yT_ne_err = np.ones_like(log_x)
        
        self._logger.info("Fitting for the Luminosity-Weighted Values")
        minim, self.LW_kT_model_best_fit_pars, self.LW_ne_model_best_fit_pars, log_x, log_yT_kT_y, log_yT_ne_y, log_yT_kT_yerr, log_yT_ne_yerr, self.LW_non_outlier_idxs = self._do_joint_minimization_with_leave_one_out(params, log_x, log_yT_kT_y, log_yT_ne_y, log_yT_kT_err, log_yT_ne_err, kT_redchi_thresh, ne_redchi_thresh) 
        # print("profile_non_outlier_idxs", self.LW_non_outlier_idxs)
        if self.do_MCMC:
            minim.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(5))
            result_emcee = lmfit.minimize(self._joint_log_kT_log_ne_minim, method='emcee', nan_policy='omit',
                     params=minim.params, is_weighted=False, progress=True, steps=self.emcee_nsteps, burn=self.emcee_burn, thin=self.emcee_thin, nwalkers = self.emcee_walkers, args = (log_x,), kws={"log_kT_y": log_yT_kT_y , "log_ne_y": log_yT_ne_y, "log_kT_yerr":np.ones_like(log_x),"log_ne_yerr":np.ones_like(log_x)})
            self.LW_joint_redchisqr = result_emcee.redchi
            flattened_chain = result_emcee.flatchain
            draw = np.floor(np.random.uniform(0,len(flattened_chain),size=self.emcee_nsteps))
            self.LW_kT_thetas = flattened_chain.iloc[draw].drop(["__lnsigma", *self.LW_ne_model_best_fit_pars.keys()] , axis = 1)
            self.LW_ne_thetas = flattened_chain.iloc[draw].drop(["__lnsigma", *self.LW_kT_model_best_fit_pars.keys()] , axis = 1)
            self.LW_kT_models = [self.kT_model(self.r_values_fine.value, **i) for i in self.LW_kT_thetas.to_dict(orient = 'records')]
            self.LW_ne_models = [self.ne_model(self.r_values_fine.value, **i) for i in self.LW_ne_thetas.to_dict(orient = 'records')]
            
            self.LW_kT_spread  = (abs(np.percentile(self.LW_kT_models,(16,84), axis=0) - self.kT_model(self.r_values_fine.value, **self.LW_kT_model_best_fit_pars,)) * u.keV).to("keV")
            self.LW_ne_spread  = (abs(np.percentile(self.LW_ne_models,(16,84), axis=0) - self.ne_model(self.r_values_fine.value, **self.LW_ne_model_best_fit_pars,)) * u.cm**-3).to("cm**-3") 
        self.calculate_LW_derived_mass_profile()
        self._logger.info("Calculated LW Mass Profile")
        self.calculate_LW_derived_entropy()
        self._logger.info("Calculated LW S Profile")
        self.calculate_LW_derived_pressure()
        self._logger.info("Calculated LW Pressure Profile")
        self.calculate_weighted_kT_LW()
        self._logger.info("Calculated Weighted LW kT and kT(R500)")
        self.calculate_weighted_S_LW()
        self._logger.info("Calculated Weighted LW S")
        self.calculate_LW_mgas()
        self._logger.info("Calculated LW Mgas")
        


    def joint_fit_Xray_kT_ne(self, kT_redchi_thresh = 0.2, ne_redchi_thresh = 0.2, smoothness_regularization_power = 0, kT_smoothness_radius_limit = None, initial_values = {}):
        
        self._logger.info(f"Currently fitting for halo {self.halo_idx}")
        self._fit_data_type = "Xray"
        self._smoothness_regularization_power = smoothness_regularization_power
        self._kT_smoothness_radius_limit = kT_smoothness_radius_limit
        params = Parameters()
        params.add("log_kT_0", value = initial_values.get("log_kT_0", np.log10(1)), min = -2, max = 2)
        params.add("log_kT_min", value = initial_values.get("log_kT_min",np.log10(0.3)), min = -2, max = 2)
        params.add("log_r_cool", value = initial_values.get("log_r_cool", np.log10(1000)), min = np.log10(200), max = np.log10(3000))
        params.add("log_r_t", value = initial_values.get("log_r_t", np.log10(1000)), min = np.log10(500), max = np.log10(5000) )
        params.add("a_cool", value = initial_values.get("a_cool",5), max = 20, min = 1e-1)
        params.add("a",value = initial_values.get("a",0.3), max = 10, min = 1e-2)
        params.add("b", value = initial_values.get("b",5), max = 20, min = 1e-1 )
        params.add("c",value = initial_values.get("c",1.3), max = 10, min = 1e-2 )
        
        params.add("log_ne_0", value = initial_values.get("log_ne_0",-2), min = -4, max = 0)
        params.add("log_r_c", value = initial_values.get("log_r_c",np.log10(150)), min = np.log10(5), max = np.log10(800) )
        params.add("beta", value = initial_values.get("beta",0.5), min = 0.1 , max = 2 )
        params.add("log_r_s", value = initial_values.get("log_r_s",np.log10(700)), min = np.log10(200), max = np.log10(4000) )
        # params.add("gamma", value = 3, min = 2.99 , max = 3 )
        params.add("eps", value = initial_values.get("eps",1), min = 0 , max = 5 )        
        log_kT_y = np.log10(self.kT_y.value)
        log_ne_y = np.log10(self.ne_y.value)
        log_x = np.log10(self.r.value)
        self._logger.info("Using max +/- error as error for X-ray ne and kT fit")
        
        kT_err = np.array( [max(abs(x)) for x in self.kT_yerr.value] )
        ne_err = np.array( [max(abs(x)) for x in self.ne_yerr.value] )

        
        ### Turn the errors in true values to errors in the logarithm, since we perform the minimisation in log space
        log_kT_err = np.array( [kT_err[i]/(float(self.kT_y.value[i])*np.log(10)) for i in range(len(self.kT_y.value))])
        log_ne_err = np.array( [ne_err[i]/(float(self.ne_y.value[i])*np.log(10)) for i in range(len(self.ne_y.value))])
        
        self._logger.info("Fitting for the X-ray Values")

        minim, self.Xray_kT_model_best_fit_pars, self.Xray_ne_model_best_fit_pars, self.radii_used_to_fit_Xray, log_kT_y, log_ne_y, log_kT_yerr, log_ne_yerr, self.Xray_non_outlier_idxs = self._do_joint_minimization_with_leave_one_out(params, log_x, log_kT_y, log_ne_y, log_kT_err, log_ne_err, kT_redchi_thresh, ne_redchi_thresh) 
        # if self._models_to_use = "reduced_params"   
            
        self._logger.info(f"X-ray_non_outlier_idxs: {self.Xray_non_outlier_idxs}")
        self._logger.info(f"Xray_kT_model_best_fit_pars: {self.Xray_kT_model_best_fit_pars}")
        if self.do_MCMC:
            minim.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(5))
            
            result_emcee = lmfit.minimize(self._joint_log_kT_log_ne_minim, method='emcee', nan_policy='omit',
                     params=minim.params, is_weighted=False, progress=True, steps=self.emcee_nsteps, burn=self.emcee_burn, thin=self.emcee_thin, nwalkers = self.emcee_walkers, args = (self.radii_used_to_fit_Xray,), kws={"log_kT_y": log_kT_y , "log_ne_y": log_ne_y, "log_kT_yerr":np.ones_like(self.radii_used_to_fit_Xray),"log_ne_yerr":np.ones_like(self.radii_used_to_fit_Xray)})
            self.Xray_joint_redchisqr = result_emcee.redchi
            flattened_chain = result_emcee.flatchain
            self._logger.info(f"size of chain = {np.shape(flattened_chain)}. Will take {self._mcmc_mass_samples} samples for error calculations")
            draw = np.floor(np.random.uniform(0,len(flattened_chain),size=self.emcee_nsteps))
            
            ### Drop the keys of the other model in turn
            self.Xray_kT_thetas = flattened_chain.iloc[draw].drop(["__lnsigma", *self.Xray_ne_model_best_fit_pars.keys()] , axis = 1)
            self.Xray_ne_thetas = flattened_chain.iloc[draw].drop(["__lnsigma", *self.Xray_kT_model_best_fit_pars.keys()] , axis = 1)
            self.Xray_kT_models = [self.kT_model(self.r_values_fine.value, **i) for i in self.Xray_kT_thetas.to_dict(orient = 'records')]
            self.Xray_ne_models = [self.ne_model(self.r_values_fine.value, **i) for i in self.Xray_ne_thetas.to_dict(orient = 'records')]
            self.Xray_kT_spread  = (abs(np.percentile(self.Xray_kT_models,(16,84), axis=0) - self.kT_model(self.r_values_fine.value, **self.Xray_kT_model_best_fit_pars,)) * u.keV).to("keV")
            self.Xray_ne_spread  = (abs(np.percentile(self.Xray_ne_models,(16,84), axis=0) - self.ne_model(self.r_values_fine.value, **self.Xray_ne_model_best_fit_pars,)) * u.cm**-3).to("cm**-3")          
        self._logger.info("Successfully joint fit Xray profiles")   
        self.calculate_Xray_derived_mass_profile()
        self._logger.info("Calculated X-ray Mass Profile")
        self.calculate_Xray_derived_entropy()
        self._logger.info("Calculated X-ray S Profile")
        self.calculate_Xray_derived_pressure()
        self._logger.info("Calculated X-ray Pressure Profile")
        self.calculate_weighted_kT_Xray()
        self._logger.info("Calculated Weighted X-ray kT and kT(R500)")
        self.calculate_weighted_S_Xray()
        self._logger.info("Calculated Weighted X-ray S")
        self.calculate_Xray_mgas()
        self._logger.info("Calculated X-ray Mgas")
        with open('./halo_fitting_record.out','a') as f:
            f.write(f"\n {self.halo_idx}: successfully joint fit Xray profiles" )


    def _joint_log_kT_log_ne_minim(self, p, log_r,  log_kT_y=None, log_ne_y=None, log_kT_yerr = None, log_ne_yerr = None ):
        r = 10**log_r
        kT_y = 10**log_kT_y
        ne_y = 10**log_ne_y
        log_kT_model = self._log_kT_model(log_r, p["log_kT_0"], p["log_kT_min"], p["log_r_cool"],p["log_r_t"], p["a_cool"], p["a"], p["b"], p["c"] )
        log_ne_model = self._log_ne_model(log_r, p["log_ne_0"], p["log_r_c"],  p["beta"] , p["log_r_s"],  p["eps"])
        log_kT_model_fine = self._log_kT_model(np.log10(self.r_values_fine.value), p["log_kT_0"], p["log_kT_min"], p["log_r_cool"],p["log_r_t"], p["a_cool"], p["a"], p["b"], p["c"] )
        
        resid1 = (log_kT_y - log_kT_model)/log_kT_yerr
        resid2 = (log_ne_y - log_ne_model)/log_ne_yerr
        monotonicity_penalty, mass_profile = self._check_monotonic(p)
        smoothness_penalty = self._check_smoothness(mass_profile, log_kT_model_fine)
        
        resid1 *= monotonicity_penalty
        resid2 *= monotonicity_penalty        
        resid1 *= smoothness_penalty
        resid2 *= smoothness_penalty  
        if self._kT_smoothness_radius_limit != None:
            kT_gradient_penalty = self._check_kT_gradient(self._kT_smoothness_radius_limit, log_kT_model_fine)
            resid1 *= kT_gradient_penalty
            resid2 *= kT_gradient_penalty  
        
        zeros_pad = np.zeros(max(self._num_params-(len(resid1) + len(resid2)),0))

        ###eturn error-scaled residuals a la https://stackoverflow.com/questions/15255928/how-do-i-include-errors-for-my-data-in-the-lmfit-least-squares-miniimization-an
        return np.concatenate((resid1, resid2, zeros_pad))

    def _check_monotonic(self,p):
        def _monotonicity_HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
            HSE_log_r = np.log10(HSE_r)
            log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
            log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
            mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
            term1 = -( HSE_r * u.kpc  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
            term2 = log_ne_dash 
            term3 = log_kT_dash 
            mass = ((term1 *(term2 + term3)).to("solMass")).value       
            return mass

        log_kT_model_fine = self._log_kT_model(np.log10(self.r_values_fine.value), p["log_kT_0"], p["log_kT_min"], p["log_r_cool"],p["log_r_t"], p["a_cool"], p["a"], p["b"], p["c"] )
        log_ne_model_fine = self._log_ne_model(np.log10(self.r_values_fine.value), p["log_ne_0"], p["log_r_c"],  p["beta"] , p["log_r_s"],  p["eps"])
        mass_profile = _monotonicity_HSE_formula(log_kT_model_fine, log_ne_model_fine, self.r_values_fine.value)
        dx = np.diff(mass_profile)
        monotonic = np.all(dx[1:] > 0)
        if monotonic:
            penalty = 1
        if not monotonic:
            penalty = self.monotonic_penalty
        return penalty, mass_profile

    def _check_smoothness(self, mass_profile, log_kT_model_fine, reg_magnitude = 1):
        # normed_first_deriv = np.gradient(np.log10(mass_profile), np.log10(self.r_values_fine.value))
        # second_deriv = np.gradient(normed_first_deriv, np.log10(self.r_values_fine.value))
        normed_first_deriv = np.gradient(log_kT_model_fine, np.log10(self.r_values_fine.value))
        second_deriv = np.gradient(normed_first_deriv, np.log10(self.r_values_fine.value))
        second_deriv /= len(self.r_values_fine.value)
        if np.isnan(np.sum(second_deriv)):
            '''Sometimes the second deriv is badly defined and returns nans'''
            penalty = 1000000
        else:
            penalty = reg_magnitude*abs(np.sum(second_deriv))**self._smoothness_regularization_power
        return penalty
    
    def _check_kT_gradient(self, radius_limit, log_kT_model_fine):
        # normed_first_deriv = np.gradient(np.log10(mass_profile), np.log10(self.r_values_fine.value))
        # second_deriv = np.gradient(normed_first_deriv, np.log10(self.r_values_fine.value))
        normed_first_deriv = np.gradient(log_kT_model_fine, np.log10(self.r_values_fine.value))
        radius_range_idxs = np.where(self.r_values_fine.value > radius_limit)[0]  
        normed_first_deriv_in_radius_range = normed_first_deriv[radius_range_idxs]
        # print(len(np.where(normed_first_deriv_in_radius_range > 0)[0]))
        if np.isnan(np.sum(normed_first_deriv_in_radius_range)):
            penalty = 1e50
        elif len(np.where(normed_first_deriv_in_radius_range > 0)[0]) > 0:
            penalty = 1e30 #1e10 * len(np.where(normed_first_deriv_in_radius_range > 0)[0])
        else:
            penalty = 1
        return penalty
    

    def _do_joint_minimization_with_leave_one_out(self,params,log_x,log_kT_y,log_ne_y,log_kT_err,log_ne_err, kT_redchi_thresh, ne_redchi_thresh,):  
        self._logger.info(f"Halo {self.halo_idx} ---- Original log x: {log_x}")
        self._logger.info(f"Halo {self.halo_idx} ---- Original log ne: {log_ne_y}")
        self._logger.info(f"Halo {self.halo_idx} ---- Original log kT: {log_kT_y}")
        self._logger.info(f"Halo {self.halo_idx} ---- Original params: {params}")
        
        original_log_x = log_x
        original_log_kT_y = log_kT_y
        original_log_kT_err = log_kT_err
        self._num_params = len(params)
        while True:
            ### Perform a minimisation with all the current data
            minim = minimize(self._joint_log_kT_log_ne_minim, params, args = (log_x,), kws={"log_kT_y": log_kT_y , "log_ne_y": log_ne_y, "log_kT_yerr":log_kT_err,"log_ne_yerr":log_ne_err},)
            bf = minim.params.valuesdict()

            log_kT_model_pars = {
                "log_kT_0":   bf["log_kT_0"],
                "log_kT_min": bf["log_kT_min"],
                "log_r_cool": bf["log_r_cool"],
                "log_r_t":    bf["log_r_t"],
                "a_cool":     bf["a_cool"],
                "a":          bf["a"],
                "b":          bf["b"],
                "c":          bf["c"]}
            log_ne_model_pars = {
                "log_ne_0":  bf["log_ne_0"],
                "log_r_c":   bf["log_r_c"],
                "beta":      bf["beta"],
                "log_r_s":   bf["log_r_s"],
                "eps":       bf["eps"]} 
                # "gamma":     bf["gamma"],
                  

            log_kT_npar = len(log_kT_model_pars)
            log_ne_npar = len(log_ne_model_pars)
            
            # if 2*len(log_x) + 2 == len(log_ne_model_pars) + len(log_kT_model_pars) :
            #     self._logger.warning(f"Number of data points equals number of fitting params + 2 = {2*(len(log_x)+1)}. Will break at this point")
            #     break


            # log_kT_model = self._log_kT_model(log_x, params["log_kT_0"], params["log_kT_min"], params["log_r_cool"],params["log_r_t"], params["a_cool"], params["a"], params["b"], params["c"] )
            # log_ne_model = self._log_ne_model(log_x, params["log_ne_0"], params["log_r_c"],  params["beta"] , params["log_r_s"], params["gamma"], params["eps"])
            
            log_kT_model = self._log_kT_model(log_x, bf["log_kT_0"], bf["log_kT_min"], bf["log_r_cool"],bf["log_r_t"], bf["a_cool"], bf["a"], bf["b"], bf["c"] )
            log_ne_model = self._log_ne_model(log_x, bf["log_ne_0"], bf["log_r_c"],  bf["beta"] , bf["log_r_s"], bf["eps"])
         
            log_kT_resids = abs(log_kT_y - log_kT_model)/log_kT_err
            log_ne_resids = abs(log_ne_y - log_ne_model)/log_ne_err
            log_kT_npar = len(log_kT_model_pars)
            log_ne_npar = len(log_ne_model_pars)
            
            if len(log_x) <= self.min_length :
                self._logger.warning(f"Number of data points equals user specified min length = {self.min_length}. Will break at this point")
                break
                
            if len(log_x) - 1 > log_kT_npar and  len(log_x) - 1 > log_ne_npar:
                log_kT_all_redchisqr = np.sum( log_kT_resids**2 )/ (len(log_x) - log_kT_npar)
                log_ne_all_redchisqr = np.sum( log_ne_resids**2 )/ (len(log_x) - log_ne_npar)   
                log_kT_redchisqr = np.zeros_like(log_x)
                log_ne_redchisqr = np.zeros_like(log_x)
            else:
                self._logger.warning(f"Number of data points - 1 equals min params length = {len(log_x)-1}, so cannot calculate redchisqr in leave-one-out analysis. Will break at this point")
                break

#             os.makedirs(f"./Halo_Sample/PROFILES_{self._fit_data_type}/{self.idx_instr_tag}_{self._fit_data_type}/{self.idx_instr_tag}_{self._fit_data_type}_leaveoneout_loop_{len(original_log_x)-len(log_x)}/", exist_ok = True)
            
#             fig = plt.figure(figsize = (20,10))
#             plt.scatter(log_x, log_kT_y,)
#             plt.xlim(min(0.9*min(original_log_x),1.1*min(original_log_x)), max(0.9*max(original_log_x),1.1*max(original_log_x)))
#             plt.ylim(min(0.9*min(original_log_kT_y),1.1*min(original_log_kT_y)), max(0.9*max(original_log_kT_y),1.1*max(original_log_kT_y)))
#             plt.plot(log_x, log_kT_model, label = "original fit")
#             plt.scatter(log_x, log_kT_model, color = "black", )
#             plt.legend(fontsize = 30)
#             plt.savefig(f"./Halo_Sample/PROFILES_{self._fit_data_type}/{self.idx_instr_tag}_{self._fit_data_type}/{self.idx_instr_tag}_{self._fit_data_type}_leaveoneout_loop_{len(original_log_x)-len(log_x)}/A_{self.idx_instr_tag}_lenx={len(log_x)}_original")
#             plt.cla
#             plt.close()
            
#             fig = plt.figure(figsize = (20,10))
#             plt.scatter(log_x, log_kT_err, label = "log errors")
#             plt.xlim(min(0.9*min(original_log_x),1.1*min(original_log_x)), max(0.9*max(original_log_x),1.1*max(original_log_x)))
#             plt.legend(fontsize = 30)
#             plt.savefig(f"./Halo_Sample/PROFILES_{self._fit_data_type}/{self.idx_instr_tag}_{self._fit_data_type}/{self.idx_instr_tag}_{self._fit_data_type}_leaveoneout_loop_{len(original_log_x)-len(log_x)}/A_{self.idx_instr_tag}_lenx={len(log_x)}_original_errors")
#             plt.cla
#             plt.close()
            
            for i in range(log_x.size):
                ### Perform a fit leaving one out from all current data
                idx = np.arange(0, log_x.size)
                idx = np.delete(idx, i)
                if len(log_x) - len(idx) != 1:
                    print("Something's Wrong!")
                    return
                tmp_minim = minimize(self._joint_log_kT_log_ne_minim, minim.params , args = (log_x[idx],), kws={"log_kT_y": log_kT_y[idx] , "log_ne_y": log_ne_y[idx], "log_kT_yerr":log_kT_err[idx],"log_ne_yerr":log_ne_err[idx]},)
                tmp_bf = tmp_minim.params.valuesdict()

                
                log_kT_model = self._log_kT_model(log_x[idx], tmp_bf["log_kT_0"], tmp_bf["log_kT_min"], tmp_bf["log_r_cool"],tmp_bf["log_r_t"], tmp_bf["a_cool"], tmp_bf["a"], tmp_bf["b"], tmp_bf["c"] )
                log_ne_model = self._log_ne_model(log_x[idx], tmp_bf["log_ne_0"], tmp_bf["log_r_c"],  tmp_bf["beta"] , tmp_bf["log_r_s"],  tmp_bf["eps"])                
                
                log_kT_resids = abs(log_kT_y[idx] - log_kT_model)/log_kT_err[idx]
                log_ne_resids = abs(log_ne_y[idx] - log_ne_model)/log_ne_err[idx]



                log_kT_redchisqr[i] = np.sum( log_kT_resids**2 )/ (len(log_x[idx]) - log_kT_npar)
                log_ne_redchisqr[i] = np.sum( log_ne_resids**2 )/ (len(log_x[idx]) - log_ne_npar)  
                
#                 fig = plt.figure(figsize = (20,10))
#                 plt.scatter(log_x[idx], log_kT_model, color = "black")
#                 plt.scatter(log_x[idx], log_kT_y[idx], label = f"index {i} removed \n redchi = {round(log_kT_redchisqr[i],4)}", color = "red", marker = "x")
#                 plt.xlim(min(0.9*min(original_log_x),1.1*min(original_log_x)), max(0.9*max(original_log_x),1.1*max(original_log_x)))
#                 plt.ylim(min(0.9*min(original_log_kT_y),1.1*min(original_log_kT_y)), max(0.9*max(original_log_kT_y),1.1*max(original_log_kT_y)))
#                 plt.plot(log_x[idx], log_kT_model)
                
#                 plt.legend(fontsize = 30)
#                 plt.savefig(f"./Halo_Sample/PROFILES_{self._fit_data_type}/{self.idx_instr_tag}_{self._fit_data_type}/{self.idx_instr_tag}_{self._fit_data_type}_leaveoneout_loop_{len(original_log_x)-len(log_x)}/B_{self.idx_instr_tag}_lenx={len(log_x)}_idxremoved={i}_fit")
#                 plt.cla
#                 plt.close()
#                 fig = plt.figure(figsize = (20,10))
#                 plt.scatter(log_x[idx], (log_kT_y[idx]-log_kT_model)**2, label = f"index {i} removed \n redchi = {round(log_kT_redchisqr[i],4)}", color = "red")
#                 plt.xlim(min(0.9*min(original_log_x),1.1*min(original_log_x)), max(0.9*max(original_log_x),1.1*max(original_log_x)))
#                 plt.ylim(bottom = 0)
#                 plt.legend(fontsize = 30)
#                 plt.savefig(f"./Halo_Sample/PROFILES_{self._fit_data_type}/{self.idx_instr_tag}_{self._fit_data_type}/{self.idx_instr_tag}_{self._fit_data_type}_leaveoneout_loop_{len(original_log_x)-len(log_x)}/C_{self.idx_instr_tag}_lenx={len(log_x)}_idxremoved={i}_squared_resids_no_err")
#                 plt.cla
#                 plt.close()
#                 fig = plt.figure(figsize = (20,10))
#                 plt.scatter(log_x[idx], ((log_kT_y[idx]-log_kT_model)**2)/log_kT_err[idx]**2, label = f"index {i} removed \n redchi = {round(log_kT_redchisqr[i],4)}", color = "red")
#                 plt.xlim(min(0.9*min(original_log_x),1.1*min(original_log_x)), max(0.9*max(original_log_x),1.1*max(original_log_x)))
#                 plt.ylim(bottom = 0)
#                 plt.legend(fontsize = 30)
#                 plt.savefig(f"./Halo_Sample/PROFILES_{self._fit_data_type}/{self.idx_instr_tag}_{self._fit_data_type}/{self.idx_instr_tag}_{self._fit_data_type}_leaveoneout_loop_{len(original_log_x)-len(log_x)}/D_{self.idx_instr_tag}_lenx={len(log_x)}_idxremoved={i}_squared_resids_over_err")
#                 plt.cla
#                 plt.close()


            log_kT_redchisqr_reduction = -(log_kT_all_redchisqr-log_kT_redchisqr)/log_kT_all_redchisqr
            log_ne_redchisqr_reduction = -(log_ne_all_redchisqr-log_ne_redchisqr)/log_ne_all_redchisqr
            # print("log_kT_redchisqr_reduction",log_kT_redchisqr_reduction)
            # print("log_ne_redchisqr_reduction",log_ne_redchisqr_reduction)
            log_kT_max_idx = np.argmax(-log_kT_redchisqr_reduction)
            log_kT_max_reduct = -log_kT_redchisqr_reduction[log_kT_max_idx]  

            log_ne_max_idx = np.argmax(-log_ne_redchisqr_reduction)
            log_ne_max_reduct = -log_ne_redchisqr_reduction[log_ne_max_idx]   

            if log_kT_max_reduct > kT_redchi_thresh and log_ne_max_reduct > ne_redchi_thresh:
                self._logger.info(f"Joint Fit: The datapoint at index {log_kT_max_idx} increases fractional chi square by {round(log_kT_max_reduct,3)} for log_kT. It will be removed and the profile refit")
                self._logger.info(f"Joint Fit: The datapoint at index {log_ne_max_idx} increases fractional chi square by {round(log_ne_max_reduct,3)} for log_ne. It will be removed and the profile refit")
                idx2 = np.arange(0, log_x.size)
                idx2 = np.delete(idx2, [log_kT_max_idx, log_ne_max_idx])
            elif log_kT_max_reduct > kT_redchi_thresh:
                self._logger.info(f"Joint Fit: The datapoint at index {log_kT_max_idx} increases fractional chi square by {round(log_kT_max_reduct,3)} for log_kT. It will be removed and the profile refit")
                idx2 = np.arange(0, log_x.size)
                idx2 = np.delete(idx2, log_kT_max_idx)
            elif log_ne_max_reduct > ne_redchi_thresh:
                self._logger.info(f"Joint Fit: The datapoint at index {log_ne_max_idx} increases fractional chi square by {round(log_ne_max_reduct,3)} for log_ne. It will be removed and the profile refit")  
                idx2 = np.arange(0, log_x.size)
                idx2 = np.delete(idx2, log_ne_max_idx)
            else:
                self._logger.info(f"Joint Fit: The maximum fractional reduction in chi square achieved via leaving a point out is {round(log_kT_max_reduct,3)} (index {log_kT_max_idx}) for log_kT and {round(log_ne_max_reduct,3)} (index {log_ne_max_idx}) for log_ne. This is below threshold so we accept the fit")
                break

            log_x = log_x[idx2]
            log_kT_y = log_kT_y[idx2] 
            log_ne_y = log_ne_y[idx2] 
            log_kT_err = log_kT_err[idx2]
            log_ne_err = log_ne_err[idx2]
        included_idxs = np.array([i for i in range(0,len(original_log_x)) if original_log_x[i] in log_x])
        self._logger.info(f"len(log x) = {len(log_x)}")
        self._logger.info(f"log kT npar = {log_kT_npar}")
        self._logger.info(f"log ne npar = {log_ne_npar}")
        self._logger.info(f"log kT resids = {log_kT_resids}")
        self._logger.info(f"log ne resids = {log_ne_resids}")  
        try:
            self._logger.info(f"Reduced chi-squared of final fit: kT_chi = {round(log_kT_all_redchisqr,3)}, ne_chi = {round(log_ne_all_redchisqr,3)}")
        except:
            pass
        return minim, log_kT_model_pars, log_ne_model_pars, log_x, log_kT_y, log_ne_y, log_kT_err, log_ne_err, included_idxs
    
    
    def kT_model(self, r, log_kT_0, log_kT_min, log_r_cool,log_r_t, a_cool, a, b, c):
        kT_0 = 10**log_kT_0
        r_cool = 10**log_r_cool
        r_t = 10**log_r_t
        kT_min =10**log_kT_min
        
        x = (r/r_cool)**a_cool
        
        t_r = ((r/r_t)**-a)  /  (( 1 + (1/r_t)**b)**(c/b))
        
        t_cool = (x + (kT_min/kT_0))   /   (x+1)
        
        kT = kT_0 *  t_r * t_cool

        return kT     
    
    def _log_kT_model(self, log_r, log_kT_0, log_kT_min, log_r_cool,log_r_t, a_cool, a, b, c):
        r = 10**(log_r)            
        return np.log10(self.kT_model(r, log_kT_0, log_kT_min, log_r_cool,log_r_t, a_cool, a, b, c))     
    
    def ne_model(self,  r, log_ne_0, log_r_c,  beta , log_r_s,  eps):
        gamma = 3
        ne_0 = 10**log_ne_0
        r_c = 10**log_r_c
        r_s = 10**log_r_s   
        p1 = ( 1 + (r/r_c)**2 )**(-3*beta)
        p1 /= (r/r_c)
        p2 = 1/(1 + (r/r_s)**gamma)**(eps/gamma)
        # try:
        #     print(f"{np.count_nonzero(np.isnan(np.sqrt( p1*p2 )))} are nan")
        #     pass
        # except:
        #     print("p1 times p2 negative")
        return ne_0 * np.sqrt( p1*p2 )    
    
    def _log_ne_model(self,  log_r, log_ne_0, log_r_c,  beta , log_r_s,  eps):
        r = 10**(log_r)
        return np.log10(self.ne_model(r, log_ne_0, log_r_c,  beta , log_r_s,  eps))
    
    
    
    
    
    
    
    def ODR_fit_Xray_kT_ne(self, kT_redchi_thresh = 0.2, ne_redchi_thresh = 0.2, initial_values = {}, kT_smoothness_radius_limit = None):
        
        self._logger.info(f"Currently ODR-fitting for halo {self.halo_idx}")
        self._fit_data_type = "Xray_ODR"
        self._ODR_radius_limit = kT_smoothness_radius_limit
        
        def ODR_log_kT_model(B,x):   
            log_kT_0, log_kT_min, log_r_cool,log_r_t, a_cool, a, b, c = np.array(B)
            r = 10**(x)   
            if log_kT_0 > 2 or log_kT_0 < -2:
                return 1e50 * np.ones(len(x))
            if log_kT_min > 2 or log_kT_min < -2:
                return 1e50 * np.ones(len(x))
            if log_r_cool > np.log10(3000) or log_r_cool < np.log10(200):
                return 1e50 * np.ones(len(x))
            if log_r_t > np.log10(5000) or log_r_t < np.log10(500):
                return 1e50 * np.ones(len(x))
            if a_cool > 20 or a_cool < 1e-1:
                return 1e50 * np.ones(len(x))
            if a > 10 or a < 1e-2:
                return 1e50 * np.ones(len(x))
            if b > 20 or b < 1e-1:
                return 1e50 * np.ones(len(x))
            if c > 10 or c < 1e-2:
                return 1e50 * np.ones(len(x))
            
            log_kT_model_fine = self._log_kT_model(np.log10(self.r_values_fine.value), log_kT_0, log_kT_min, log_r_cool,log_r_t, a_cool, a, b, c )
            if self._check_kT_gradient(self._ODR_radius_limit, log_kT_model_fine) != 1:
                return 1e50 * np.ones(len(x))
            
            return np.log10(self.kT_model(r, log_kT_0, log_kT_min, log_r_cool,log_r_t, a_cool, a, b, c))     

        def ODR_log_ne_model(B,x):    
            log_ne_0, log_r_c,  beta , log_r_s,  eps = np.array(B)
            r = 10**(x)
            if beta > 2 or beta < 0.5:
                return 1e50 * np.ones(len(x))
            if eps > 5 or eps < 0:
                return 1e50 * np.ones(len(x))
            if log_r_s > np.log10(4000) or log_r_s < np.log10(200):
                return 1e50 * np.ones(len(x))
            if log_r_c > np.log10(800) or log_r_c < np.log10(5):
                return 1e50 * np.ones(len(x))
            if log_ne_0 > -2 or log_ne_0 < -4:
                return 1e50 * np.ones(len(x))
            return np.log10(self.ne_model(r, log_ne_0, log_r_c,  beta , log_r_s, eps))

        kT_model = scipy.odr.Model(ODR_log_kT_model)
        ne_model = scipy.odr.Model(ODR_log_ne_model)
              
        log_kT_y = np.log10(self.kT_y.value)
        log_ne_y = np.log10(self.ne_y.value)
        log_x = np.log10(self.r.value)
        self._logger.info("Using max +/- error as error for ODR X-ray ne and kT fits")

        kT_err = np.array( [max(abs(x)) for x in self.kT_yerr.value] )
        ne_err = np.array( [max(abs(x)) for x in self.ne_yerr.value] )
        x_err =  np.array( [max(abs(x)) for x in self.r_err.value] )
        

        
        ### Turn the errors in true values to errors in the logarithm, since we perform the minimisation in log space
        # log_kT_err = np.array( [  [kT_err[i][0]/(float(self.kT_y.value[i])*np.log(10)),kT_err[i][1]/(float(self.kT_y.value[i])*np.log(10))] for i in range(len(self.kT_y.value))])
        # log_ne_err = np.array( [  [ne_err[i][0]/(float(self.ne_y.value[i])*np.log(10)),ne_err[i][1]/(float(self.ne_y.value[i])*np.log(10))] for i in range(len(self.ne_y.value))])
        # log_x_err  = np.array( [  [x_err[i][0]/(float(self.r.value[i])*np.log(10)),x_err[i][1]/(float(self.r.value[i])*np.log(10))] for i in range(len(self.r.value))])
        log_kT_err = np.array( [  kT_err[i]/(float(self.kT_y.value[i])*np.log(10)) for i in range(len(self.kT_y.value))])
        log_ne_err = np.array( [  ne_err[i]/(float(self.ne_y.value[i])*np.log(10)) for i in range(len(self.ne_y.value))])
        log_x_err  = np.array( [  x_err[i] /(float(self.r.value[i])*np.log(10)) for i in range(len(self.r.value))])
        self._logger.info("Using non-outlier idxs from joint fit for ODR...")
        
        log_kT_y = log_kT_y[self.Xray_non_outlier_idxs]
        log_ne_y = log_ne_y[self.Xray_non_outlier_idxs]
        log_x  = log_x[self.Xray_non_outlier_idxs]
        
        log_kT_err = log_kT_err[self.Xray_non_outlier_idxs]
        log_ne_err = log_ne_err[self.Xray_non_outlier_idxs]
        log_x_err  = log_x_err[self.Xray_non_outlier_idxs]
        
        
        print("log_kT_err", log_kT_err)
        print("\n x_err", x_err)
        print("\n log_x_err", log_x_err)
        log_kT_data = scipy.odr.Data(x=log_x, y=log_kT_y, wd=1./np.power(log_x_err,2), we=1./np.power(log_kT_err,2))
        log_ne_data = scipy.odr.Data(x=log_x, y=log_ne_y, wd=1./np.power(log_x_err,2), we=1./np.power(log_ne_err,2))
        
        kT_param_keys = ["log_kT_0","log_kT_min","log_r_cool","log_r_t","a_cool", "a","b","c" ]
        ne_param_keys = ["log_ne_0", "log_r_c", "beta", "log_r_s", "eps"]
        
        # kT_odr = scipy.odr.ODR(log_kT_data, kT_model, beta0=[0,np.log10(0.3), np.log10(1000), np.log10(1000), 5,0.3,5, 1.3], maxit = 5000)
        # ne_odr = scipy.odr.ODR(log_ne_data, ne_model, beta0=[-2, np.log10(150), 0.5, np.log10(700), 1], maxit = 5000)
        
        kT_odr = scipy.odr.ODR(log_kT_data, kT_model, beta0=[initial_values.get(x, self.Xray_kT_model_best_fit_pars[x]) for x in kT_param_keys], maxit = 1000)
        ne_odr = scipy.odr.ODR(log_ne_data, ne_model, beta0=[initial_values.get(x, self.Xray_ne_model_best_fit_pars[x]) for x in ne_param_keys], maxit = 1000) 
        
        self._kT_odr_out = kT_odr.run()
        self._ne_odr_out = ne_odr.run()
        
        


        self.Xray_ODR_kT_model_best_fit_pars = dict(zip(kT_param_keys ,self._kT_odr_out.beta))
        self.Xray_ODR_ne_model_best_fit_pars = dict(zip(ne_param_keys ,self._ne_odr_out.beta))
        # self._kT_odr_out.pprint()
        # self._ne_odr_out.pprint()
        self.calculate_Xray_ODR_derived_mass_profile()       
        with open('./halo_fitting_record.out','a') as f:
            f.write(f"\n {self.halo_idx}: successfully ODR fit Xray profiles" )        
        
        
    
    
    
    def calculate_Xray_mgas(self,):    
        r_in_frac = 0.1
        r_out_frac = 1
        
        def _mgas_integrand(r, ne_model_best_fit_pars):
            ne_ml_yval = float(self.ne_model(r, **ne_model_best_fit_pars))*u.cm**-3
            num_integrand = (ne_ml_yval * ((r*u.kpc)**2)).to("kpc**-1").value
            return num_integrand

        r_in = r_in_frac  *  self.Xray_HSE_R500.to('kpc').value
        r_out = r_out_frac *  self.Xray_HSE_R500.to('kpc').value

        mu_e = 1.155
        mgas = 4*math.pi*mu_e*m_p* scipy.integrate.quad(_mgas_integrand, r_in, r_out, args = (self.Xray_ne_model_best_fit_pars,) )[0]
        self.Xray_mgas_in_HSE_R500 = mgas
        self.Xray_Y = (self.weighted_kT_Xray * self.Xray_mgas_in_HSE_R500).to("keV * solMass") 

        if self.do_MCMC:

            def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
                HSE_log_r = np.log10(HSE_r.value)
                log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
                log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
                mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
                term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
                term2 = log_ne_dash 
                term3 = log_kT_dash 
                mass = (term1 *(term2 + term3)).to("solMass")
                mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
                return mass, mean_rho

            mgas_models = []
            ### Do an effective outer product of the first N samples drawn from each of the posterior distributions of kT and ne
            for i in range(self._mcmc_mass_samples):
                    sample_kT_pars =  self.Xray_kT_thetas.to_dict(orient = 'records')[i]
                # for j in range(self._mcmc_mass_samples):
                    # sample_ne_pars =  self.Xray_ne_thetas.to_dict(orient = 'records')[j]
                    sample_ne_pars =  self.Xray_ne_thetas.to_dict(orient = 'records')[i]
                    log_r = np.log10(self.r_values_fine.value)
                    log_kT = np.log10(self.kT_model(self.r_values_fine.value, **sample_kT_pars))
                    log_ne = np.log10(self.ne_model(self.r_values_fine.value, **sample_ne_pars))        
                    _, sample_mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    sample_HSE_R500 = self.r_values_fine[np.abs(  (sample_mean_rho - 500*self.rho_crit).value).argmin()]                  
                    sample_r_in = r_in_frac * sample_HSE_R500.to('kpc').value
                    sample_r_out = r_out_frac *sample_HSE_R500.to('kpc').value

                    mgas_model = 4*math.pi*mu_e*m_p* scipy.integrate.quad(_mgas_integrand, sample_r_in, sample_r_out, args = (sample_ne_pars,) )[0]
                    mgas_models.append(mgas_model.to("Msun").value)   

            self.Xray_mgas_in_HSE_R500_spread = (abs(np.percentile(mgas_models,(16,84), axis=0)*u.Msun - self.Xray_mgas_in_HSE_R500)).to("solMass")
            self.Xray_Y_spread = (np.sqrt(np.array(self.Xray_mgas_in_HSE_R500_spread.to("Msun"))**2 + np.array(self.weighted_kT_Xray_spread.to("keV"))**2) * u.keV * u.Msun).to("keV * solMass")    

        
    def calculate_LW_mgas(self,):    
        r_in_frac = 0.1
        r_out_frac = 1
        
        def _mgas_integrand(r, ne_model_best_fit_pars):
            ne_ml_yval = float(self.ne_model(r, **ne_model_best_fit_pars))*u.cm**-3
            num_integrand = (ne_ml_yval * ((r*u.kpc)**2)).to("kpc**-1").value
            return num_integrand

        r_in = r_in_frac  *  self.LW_HSE_R500.to('kpc').value
        r_out = r_out_frac *  self.LW_HSE_R500.to('kpc').value

        mu_e = 1.155
        mgas = 4*math.pi*mu_e*m_p* scipy.integrate.quad(_mgas_integrand, r_in, r_out, args = (self.LW_ne_model_best_fit_pars,) )[0]
        self.LW_mgas_in_HSE_R500 = mgas
        self.LW_Y = (self.weighted_kT_LW * self.LW_mgas_in_HSE_R500).to("keV * solMass") 

        if self.do_MCMC:

            def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
                HSE_log_r = np.log10(HSE_r.value)
                log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
                log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
                mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
                term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
                term2 = log_ne_dash 
                term3 = log_kT_dash 
                mass = (term1 *(term2 + term3)).to("solMass")
                mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
                return mass, mean_rho

            mgas_models = []
            ### Do an effective outer product of the first N samples drawn from each of the posterior distributions of kT and ne
            for i in range(self._mcmc_mass_samples):
                    sample_kT_pars =  self.LW_kT_thetas.to_dict(orient = 'records')[i]
                # for j in range(self._mcmc_mass_samples):
                    # sample_ne_pars =  self.Xray_ne_thetas.to_dict(orient = 'records')[j]
                    sample_ne_pars =  self.LW_ne_thetas.to_dict(orient = 'records')[i]
                    log_r = np.log10(self.r_values_fine.value)
                    log_kT = np.log10(self.kT_model(self.r_values_fine.value, **sample_kT_pars))
                    log_ne = np.log10(self.ne_model(self.r_values_fine.value, **sample_ne_pars))        
                    _, sample_mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    sample_HSE_R500 = self.r_values_fine[np.abs(  (sample_mean_rho - 500*self.rho_crit).value).argmin()]                  
                    sample_r_in = r_in_frac * sample_HSE_R500.to('kpc').value
                    sample_r_out = r_out_frac *sample_HSE_R500.to('kpc').value

                    mgas_model = 4*math.pi*mu_e*m_p* scipy.integrate.quad(_mgas_integrand, sample_r_in, sample_r_out, args = (sample_ne_pars,) )[0]
                    mgas_models.append(mgas_model.to("Msun").value)   

            self.LW_mgas_in_HSE_R500_spread = (abs(np.percentile(mgas_models,(16,84), axis=0)*u.Msun - self.LW_mgas_in_HSE_R500)).to("solMass")
            self.LW_Y_spread = (np.sqrt(np.array(self.LW_mgas_in_HSE_R500_spread.to("Msun"))**2 + np.array(self.weighted_kT_LW_spread.to("keV"))**2) * u.keV * u.Msun).to("keV * solMass")    

    def calculate_EW_mgas(self,):    
        r_in_frac = 0.1
        r_out_frac = 1
        
        def _mgas_integrand(r, ne_model_best_fit_pars):
            ne_ml_yval = float(self.ne_model(r, **ne_model_best_fit_pars))*u.cm**-3
            num_integrand = (ne_ml_yval * ((r*u.kpc)**2)).to("kpc**-1").value
            return num_integrand

        r_in = r_in_frac  *  self.EW_HSE_R500.to('kpc').value
        r_out = r_out_frac *  self.EW_HSE_R500.to('kpc').value

        mu_e = 1.155
        mgas = 4*math.pi*mu_e*m_p* scipy.integrate.quad(_mgas_integrand, r_in, r_out, args = (self.EW_ne_model_best_fit_pars,) )[0]
        self.EW_mgas_in_HSE_R500 = mgas
        self.EW_Y = (self.weighted_kT_EW * self.EW_mgas_in_HSE_R500).to("keV * solMass") 

        if self.do_MCMC:

            def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
                HSE_log_r = np.log10(HSE_r.value)
                log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
                log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
                mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
                term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
                term2 = log_ne_dash 
                term3 = log_kT_dash 
                mass = (term1 *(term2 + term3)).to("solMass")
                mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
                return mass, mean_rho

            mgas_models = []
            ### Do an effective outer product of the first N samples drawn from each of the posterior distributions of kT and ne
            for i in range(self._mcmc_mass_samples):
                    sample_kT_pars =  self.EW_kT_thetas.to_dict(orient = 'records')[i]
                # for j in range(self._mcmc_mass_samples):
                    # sample_ne_pars =  self.Xray_ne_thetas.to_dict(orient = 'records')[j]
                    sample_ne_pars =  self.EW_ne_thetas.to_dict(orient = 'records')[i]
                    log_r = np.log10(self.r_values_fine.value)
                    log_kT = np.log10(self.kT_model(self.r_values_fine.value, **sample_kT_pars))
                    log_ne = np.log10(self.ne_model(self.r_values_fine.value, **sample_ne_pars))        
                    _, sample_mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    sample_HSE_R500 = self.r_values_fine[np.abs(  (sample_mean_rho - 500*self.rho_crit).value).argmin()]                  
                    sample_r_in = r_in_frac * sample_HSE_R500.to('kpc').value
                    sample_r_out = r_out_frac *sample_HSE_R500.to('kpc').value

                    mgas_model = 4*math.pi*mu_e*m_p* scipy.integrate.quad(_mgas_integrand, sample_r_in, sample_r_out, args = (sample_ne_pars,) )[0]
                    mgas_models.append(mgas_model.to("Msun").value)   

            self.EW_mgas_in_HSE_R500_spread = (abs(np.percentile(mgas_models,(16,84), axis=0)*u.Msun - self.EW_mgas_in_HSE_R500)).to("solMass")
            self.EW_Y_spread = (np.sqrt(np.array(self.EW_mgas_in_HSE_R500_spread.to("Msun"))**2 + np.array(self.weighted_kT_EW_spread.to("keV"))**2) * u.keV * u.Msun).to("keV * solMass")    
            
            
            
    def calculate_MW_mgas(self,):    
        r_in_frac = 0.1
        r_out_frac = 1
        
        def _mgas_integrand(r, ne_model_best_fit_pars):
            ne_ml_yval = float(self.ne_model(r, **ne_model_best_fit_pars))*u.cm**-3
            num_integrand = (ne_ml_yval * ((r*u.kpc)**2)).to("kpc**-1").value
            return num_integrand

        r_in = r_in_frac  *  self.MW_HSE_R500.to('kpc').value
        r_out = r_out_frac *  self.MW_HSE_R500.to('kpc').value

        mu_e = 1.155
        mgas = 4*math.pi*mu_e*m_p* scipy.integrate.quad(_mgas_integrand, r_in, r_out, args = (self.MW_ne_model_best_fit_pars,) )[0]
        self.MW_mgas_in_HSE_R500 = mgas
        self.MW_Y = (self.weighted_kT_MW * self.MW_mgas_in_HSE_R500).to("keV * solMass") 

        if self.do_MCMC:

            def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
                HSE_log_r = np.log10(HSE_r.value)
                log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
                log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
                mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
                term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
                term2 = log_ne_dash 
                term3 = log_kT_dash 
                mass = (term1 *(term2 + term3)).to("solMass")
                mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
                return mass, mean_rho

            mgas_models = []
            ### Do an effective outer product of the first N samples drawn from each of the posterior distributions of kT and ne
            for i in range(self._mcmc_mass_samples):
                    sample_kT_pars =  self.MW_kT_thetas.to_dict(orient = 'records')[i]
                # for j in range(self._mcmc_mass_samples):
                    # sample_ne_pars =  self.Xray_ne_thetas.to_dict(orient = 'records')[j]
                    sample_ne_pars =  self.MW_ne_thetas.to_dict(orient = 'records')[i]
                    log_r = np.log10(self.r_values_fine.value)
                    log_kT = np.log10(self.kT_model(self.r_values_fine.value, **sample_kT_pars))
                    log_ne = np.log10(self.ne_model(self.r_values_fine.value, **sample_ne_pars))        
                    _, sample_mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    sample_HSE_R500 = self.r_values_fine[np.abs(  (sample_mean_rho - 500*self.rho_crit).value).argmin()]                  
                    sample_r_in = r_in_frac * sample_HSE_R500.to('kpc').value
                    sample_r_out = r_out_frac *sample_HSE_R500.to('kpc').value

                    mgas_model = 4*math.pi*mu_e*m_p* scipy.integrate.quad(_mgas_integrand, sample_r_in, sample_r_out, args = (sample_ne_pars,) )[0]
                    mgas_models.append(mgas_model.to("Msun").value)   

            self.MW_mgas_in_HSE_R500_spread = (abs(np.percentile(mgas_models,(16,84), axis=0)*u.Msun - self.MW_mgas_in_HSE_R500)).to("solMass")
            self.MW_Y_spread = (np.sqrt(np.array(self.MW_mgas_in_HSE_R500_spread.to("Msun"))**2 + np.array(self.weighted_kT_MW_spread.to("keV"))**2) * u.keV * u.Msun).to("keV * solMass")                
            
        
    def calculate_weighted_kT_Xray(self):
        r_in = self.inner_R500_frac*self.Xray_HSE_R500.to('kpc').value
        r_out = 1*self.Xray_HSE_R500.to('kpc').value

        
        def _kT_numerator_integrand(r, Xray_kT_model_best_fit_pars, Xray_ne_model_best_fit_pars):
            kT_Xray_ml_yval = float(self.kT_model(r, **Xray_kT_model_best_fit_pars))
            ne_Xray_ml_yval = float(self.ne_model(r, **Xray_ne_model_best_fit_pars))
            num_integrand = kT_Xray_ml_yval * (ne_Xray_ml_yval**2) * (r**2)
            return num_integrand
        def _kT_denom_integrand(r, Xray_ne_model_best_fit_pars):
            ne_Xray_ml_yval = float(self.ne_model(r, **Xray_ne_model_best_fit_pars))
            den_integrand = (ne_Xray_ml_yval**2) * (r**2)
            return den_integrand
        
        weighted_kT = scipy.integrate.quad(_kT_numerator_integrand, r_in, r_out, args = (self.Xray_kT_model_best_fit_pars, self.Xray_ne_model_best_fit_pars) )
        normalisation = scipy.integrate.quad(_kT_denom_integrand, r_in, r_out, args = (self.Xray_ne_model_best_fit_pars,) )
        self.weighted_kT_Xray = weighted_kT[0]/normalisation[0]
        self.weighted_kT_Xray *= u.keV 

        if self.do_MCMC:

            def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
                HSE_log_r = np.log10(HSE_r.value)
                log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
                log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
                mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
                term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
                term2 = log_ne_dash 
                term3 = log_kT_dash 
                mass = (term1 *(term2 + term3)).to("solMass")
                mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
                return mass, mean_rho

            weighted_kT_Xray_models = []
            kT_Xray_at_R500_models = []
            ### Do an effective outer product of the first N samples drawn from each of the posterior distributions of kT and ne
            for i in range(self._mcmc_mass_samples):
                    sample_kT_pars =  self.Xray_kT_thetas.to_dict(orient = 'records')[i]
                # for j in range(self._mcmc_mass_samples):
                    # sample_ne_pars =  self.Xray_ne_thetas.to_dict(orient = 'records')[j]
                    sample_ne_pars =  self.Xray_ne_thetas.to_dict(orient = 'records')[i]
                    log_r = np.log10(self.r_values_fine.value)
                    log_kT = np.log10(self.kT_model(self.r_values_fine.value, **sample_kT_pars))
                    log_ne = np.log10(self.ne_model(self.r_values_fine.value, **sample_ne_pars))        
                    _, sample_mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    sample_HSE_R500 = self.r_values_fine[np.abs(  (sample_mean_rho - 500*self.rho_crit).value).argmin()]                  
                    sample_r_in = self.inner_R500_frac*sample_HSE_R500.to('kpc').value
                    sample_r_out = 1*sample_HSE_R500.to('kpc').value
                    weighted_kT = scipy.integrate.quad(_kT_numerator_integrand, sample_r_in, sample_r_out, args = (sample_kT_pars, sample_ne_pars) )
                    normalisation = scipy.integrate.quad(_kT_denom_integrand, sample_r_in, sample_r_out, args = (sample_ne_pars,) )
                    weighted_kT_Xray = weighted_kT[0]/normalisation[0]
                    weighted_kT_Xray_models.append(weighted_kT_Xray)
                    
                    kT_Xray_at_R500 = self.kT_model(sample_HSE_R500.value, **sample_kT_pars)
                    if round(kT_Xray_at_R500,5) != round(10**log_kT[np.abs( (sample_mean_rho - 500*self.rho_crit).value).argmin()],5):
                        self._logger.warning("Discrepancy in kT value at R500!")
                        self._logger.warning(round(kT_Xray_at_R500,5))
                        self._logger.warning(round(10**log_kT[np.abs( (sample_mean_rho - 500*self.rho_crit).value).argmin()],5))
                    kT_Xray_at_R500_models.append(kT_Xray_at_R500)
                    
            # self.weighted_kT_Xray_spread = (np.std(weighted_kT_Xray_models,axis=0) * u.keV).to("keV")        
            self.weighted_kT_Xray_spread = (abs(np.percentile(weighted_kT_Xray_models,(16,84), axis=0)*u.keV - self.weighted_kT_Xray)).to("keV")
            self.kT_Xray_at_Xray_HSE_R500  = (self.kT_model(self.Xray_HSE_R500.to('kpc').value, **self.Xray_kT_model_best_fit_pars) * u.keV).to("keV")
            self.kT_Xray_at_Xray_HSE_R500_spread = (abs(np.percentile(kT_Xray_at_R500_models,(16,84), axis=0)*u.keV - self.kT_Xray_at_Xray_HSE_R500 )).to("keV")
            
            
            
            
            
            
            
    def calculate_weighted_kT_MW(self):
        r_in = self.inner_R500_frac*self.MW_HSE_R500.to('kpc').value
        r_out = 1*self.MW_HSE_R500.to('kpc').value

        
        def _kT_numerator_integrand(r, MW_kT_model_best_fit_pars, MW_ne_model_best_fit_pars):
            kT_MW_ml_yval = float(self.kT_model(r, **MW_kT_model_best_fit_pars))
            ne_MW_ml_yval = float(self.ne_model(r, **MW_ne_model_best_fit_pars))
            num_integrand = kT_MW_ml_yval * (ne_MW_ml_yval**2) * (r**2)
            return num_integrand
        def _kT_denom_integrand(r, MW_ne_model_best_fit_pars):
            ne_MW_ml_yval = float(self.ne_model(r, **MW_ne_model_best_fit_pars))
            den_integrand = (ne_MW_ml_yval**2) * (r**2)
            return den_integrand
        
        weighted_kT = scipy.integrate.quad(_kT_numerator_integrand, r_in, r_out, args = (self.MW_kT_model_best_fit_pars, self.MW_ne_model_best_fit_pars) )
        normalisation = scipy.integrate.quad(_kT_denom_integrand, r_in, r_out, args = (self.MW_ne_model_best_fit_pars,) )
        self.weighted_kT_MW = weighted_kT[0]/normalisation[0]
        self.weighted_kT_MW *= u.keV

        if self.do_MCMC:

            def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
                HSE_log_r = np.log10(HSE_r.value)
                log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
                log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
                mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
                term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
                term2 = log_ne_dash 
                term3 = log_kT_dash 
                mass = (term1 *(term2 + term3)).to("solMass")
                mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
                return mass, mean_rho

            weighted_kT_MW_models = []
            kT_MW_at_R500_models = []
            ### Do an effective outer product of the first N samples drawn from each of the posterior distributions of kT and ne
            for i in range(self._mcmc_mass_samples):
                    sample_kT_pars =  self.MW_kT_thetas.to_dict(orient = 'records')[i]
                # for j in range(self._mcmc_mass_samples):
                    # sample_ne_pars =  self.Xray_ne_thetas.to_dict(orient = 'records')[j]
                    sample_ne_pars =  self.MW_ne_thetas.to_dict(orient = 'records')[i]
                    log_r = np.log10(self.r_values_fine.value)
                    log_kT = np.log10(self.kT_model(self.r_values_fine.value, **sample_kT_pars))
                    log_ne = np.log10(self.ne_model(self.r_values_fine.value, **sample_ne_pars))        
                    _, sample_mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    sample_HSE_R500 = self.r_values_fine[np.abs(  (sample_mean_rho - 500*self.rho_crit).value).argmin()]                  
                    sample_r_in = self.inner_R500_frac*sample_HSE_R500.to('kpc').value
                    sample_r_out = 1*sample_HSE_R500.to('kpc').value
                    weighted_kT = scipy.integrate.quad(_kT_numerator_integrand, sample_r_in, sample_r_out, args = (sample_kT_pars, sample_ne_pars) )
                    normalisation = scipy.integrate.quad(_kT_denom_integrand, sample_r_in, sample_r_out, args = (sample_ne_pars,) )
                    weighted_kT_MW = weighted_kT[0]/normalisation[0]
                    weighted_kT_MW_models.append(weighted_kT_MW)
                    
                    kT_MW_at_R500 = self.kT_model(sample_HSE_R500.value, **sample_kT_pars)
                    if round(kT_MW_at_R500,5) != round(10**log_kT[np.abs( (sample_mean_rho - 500*self.rho_crit).value).argmin()],5):
                        self._logger.warning("Discrepancy in kT value at R500!")
                        self._logger.warning(round(kT_MW_at_R500,5))
                        self._logger.warning(round(10**log_kT[np.abs( (sample_mean_rho - 500*self.rho_crit).value).argmin()],5))
                    kT_MW_at_R500_models.append(kT_MW_at_R500)
                    
            # self.weighted_kT_MW_spread = (np.std(weighted_kT_MW_models,axis=0) * u.keV).to("keV")        
            self.weighted_kT_MW_spread = (abs(np.percentile(weighted_kT_MW_models,(16,84), axis=0)*u.keV - self.weighted_kT_MW)).to("keV")
            self.kT_MW_at_MW_HSE_R500 = (self.kT_model(self.MW_HSE_R500.to('kpc').value, **self.MW_kT_model_best_fit_pars) * u.keV).to("keV")
            self.kT_MW_at_MW_HSE_R500_spread = (abs(np.percentile(kT_MW_at_R500_models,(16,84), axis=0)*u.keV - self.kT_MW_at_MW_HSE_R500)).to("keV")
            
            
            
            
            
            
    def calculate_weighted_kT_EW(self):
        r_in = self.inner_R500_frac*self.EW_HSE_R500.to('kpc').value
        r_out = 1*self.EW_HSE_R500.to('kpc').value

        
        def _kT_numerator_integrand(r, EW_kT_model_best_fit_pars, EW_ne_model_best_fit_pars):
            kT_EW_ml_yval = float(self.kT_model(r, **EW_kT_model_best_fit_pars))
            ne_EW_ml_yval = float(self.ne_model(r, **EW_ne_model_best_fit_pars))
            num_integrand = kT_EW_ml_yval * (ne_EW_ml_yval**2) * (r**2)
            return num_integrand
        def _kT_denom_integrand(r, EW_ne_model_best_fit_pars):
            ne_EW_ml_yval = float(self.ne_model(r, **EW_ne_model_best_fit_pars))
            den_integrand = (ne_EW_ml_yval**2) * (r**2)
            return den_integrand
        
        weighted_kT = scipy.integrate.quad(_kT_numerator_integrand, r_in, r_out, args = (self.EW_kT_model_best_fit_pars, self.EW_ne_model_best_fit_pars) )
        normalisation = scipy.integrate.quad(_kT_denom_integrand, r_in, r_out, args = (self.EW_ne_model_best_fit_pars,) )
        self.weighted_kT_EW = weighted_kT[0]/normalisation[0]
        self.weighted_kT_EW *= u.keV

        if self.do_MCMC:

            def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
                HSE_log_r = np.log10(HSE_r.value)
                log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
                log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
                mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
                term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
                term2 = log_ne_dash 
                term3 = log_kT_dash 
                mass = (term1 *(term2 + term3)).to("solMass")
                mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
                return mass, mean_rho

            weighted_kT_EW_models = []
            kT_EW_at_R500_models = []
            ### Do an effective outer product of the first N samples drawn from each of the posterior distributions of kT and ne
            for i in range(self._mcmc_mass_samples):
                    sample_kT_pars =  self.EW_kT_thetas.to_dict(orient = 'records')[i]
                # for j in range(self._mcmc_mass_samples):
                    # sample_ne_pars =  self.Xray_ne_thetas.to_dict(orient = 'records')[j]
                    sample_ne_pars =  self.EW_ne_thetas.to_dict(orient = 'records')[i]
                    log_r = np.log10(self.r_values_fine.value)
                    log_kT = np.log10(self.kT_model(self.r_values_fine.value, **sample_kT_pars))
                    log_ne = np.log10(self.ne_model(self.r_values_fine.value, **sample_ne_pars))        
                    _, sample_mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    sample_HSE_R500 = self.r_values_fine[np.abs(  (sample_mean_rho - 500*self.rho_crit).value).argmin()]                  
                    sample_r_in = self.inner_R500_frac*sample_HSE_R500.to('kpc').value
                    sample_r_out = 1*sample_HSE_R500.to('kpc').value
                    weighted_kT = scipy.integrate.quad(_kT_numerator_integrand, sample_r_in, sample_r_out, args = (sample_kT_pars, sample_ne_pars) )
                    normalisation = scipy.integrate.quad(_kT_denom_integrand, sample_r_in, sample_r_out, args = (sample_ne_pars,) )
                    weighted_kT_EW = weighted_kT[0]/normalisation[0]
                    weighted_kT_EW_models.append(weighted_kT_EW)
                    
                    kT_EW_at_R500 = self.kT_model(sample_HSE_R500.value, **sample_kT_pars)
                    if round(kT_EW_at_R500,5) != round(10**log_kT[np.abs( (sample_mean_rho - 500*self.rho_crit).value).argmin()],5):
                        self._logger.warning("Discrepancy in kT value at R500!")
                        self._logger.warning(round(kT_EW_at_R500,5))
                        self._logger.warning(round(10**log_kT[np.abs( (sample_mean_rho - 500*self.rho_crit).value).argmin()],5))
                    kT_EW_at_R500_models.append(kT_EW_at_R500)
                    
            # self.weighted_kT_EW_spread = (np.std(weighted_kT_EW_models,axis=0) * u.keV).to("keV")        
            self.weighted_kT_EW_spread = (abs(np.percentile(weighted_kT_EW_models,(16,84), axis=0)*u.keV - self.weighted_kT_EW)).to("keV")
            self.kT_EW_at_EW_HSE_R500 = (self.kT_model(self.EW_HSE_R500.to('kpc').value, **self.EW_kT_model_best_fit_pars) * u.keV).to("keV")
            self.kT_EW_at_EW_HSE_R500_spread = (abs(np.percentile(kT_EW_at_R500_models,(16,84), axis=0)*u.keV - self.kT_EW_at_EW_HSE_R500)).to("keV")
            
    def calculate_weighted_kT_LW(self):
        r_in = self.inner_R500_frac*self.LW_HSE_R500.to('kpc').value
        r_out = 1*self.LW_HSE_R500.to('kpc').value

        
        def _kT_numerator_integrand(r, LW_kT_model_best_fit_pars, LW_ne_model_best_fit_pars):
            kT_LW_ml_yval = float(self.kT_model(r, **LW_kT_model_best_fit_pars))
            ne_LW_ml_yval = float(self.ne_model(r, **LW_ne_model_best_fit_pars))
            num_integrand = kT_LW_ml_yval * (ne_LW_ml_yval**2) * (r**2)
            return num_integrand
        def _kT_denom_integrand(r, LW_ne_model_best_fit_pars):
            ne_LW_ml_yval = float(self.ne_model(r, **LW_ne_model_best_fit_pars))
            den_integrand = (ne_LW_ml_yval**2) * (r**2)
            return den_integrand
        
        weighted_kT = scipy.integrate.quad(_kT_numerator_integrand, r_in, r_out, args = (self.LW_kT_model_best_fit_pars, self.LW_ne_model_best_fit_pars) )
        normalisation = scipy.integrate.quad(_kT_denom_integrand, r_in, r_out, args = (self.LW_ne_model_best_fit_pars,) )
        self.weighted_kT_LW = weighted_kT[0]/normalisation[0]
        self.weighted_kT_LW *= u.keV

        if self.do_MCMC:

            def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
                HSE_log_r = np.log10(HSE_r.value)
                log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
                log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
                mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
                term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
                term2 = log_ne_dash 
                term3 = log_kT_dash 
                mass = (term1 *(term2 + term3)).to("solMass")
                mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
                return mass, mean_rho

            weighted_kT_LW_models = []
            kT_LW_at_R500_models = []
            ### Do an effective outer product of the first N samples drawn from each of the posterior distributions of kT and ne
            for i in range(self._mcmc_mass_samples):
                    sample_kT_pars =  self.LW_kT_thetas.to_dict(orient = 'records')[i]
                # for j in range(self._mcmc_mass_samples):
                    # sample_ne_pars =  self.Xray_ne_thetas.to_dict(orient = 'records')[j]
                    sample_ne_pars =  self.LW_ne_thetas.to_dict(orient = 'records')[i]
                    log_r = np.log10(self.r_values_fine.value)
                    log_kT = np.log10(self.kT_model(self.r_values_fine.value, **sample_kT_pars))
                    log_ne = np.log10(self.ne_model(self.r_values_fine.value, **sample_ne_pars))        
                    _, sample_mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    sample_HSE_R500 = self.r_values_fine[np.abs(  (sample_mean_rho - 500*self.rho_crit).value).argmin()]                  
                    sample_r_in = self.inner_R500_frac*sample_HSE_R500.to('kpc').value
                    sample_r_out = 1*sample_HSE_R500.to('kpc').value
                    weighted_kT = scipy.integrate.quad(_kT_numerator_integrand, sample_r_in, sample_r_out, args = (sample_kT_pars, sample_ne_pars) )
                    normalisation = scipy.integrate.quad(_kT_denom_integrand, sample_r_in, sample_r_out, args = (sample_ne_pars,) )
                    weighted_kT_LW = weighted_kT[0]/normalisation[0]
                    weighted_kT_LW_models.append(weighted_kT_LW)
                    
                    kT_LW_at_R500 = self.kT_model(sample_HSE_R500.value, **sample_kT_pars)
                    if round(kT_LW_at_R500,5) != round(10**log_kT[np.abs( (sample_mean_rho - 500*self.rho_crit).value).argmin()],5):
                        self._logger.warning("Discrepancy in kT value at R500!")
                        self._logger.warning(round(kT_LW_at_R500,5))
                        self._logger.warning(round(10**log_kT[np.abs( (sample_mean_rho - 500*self.rho_crit).value).argmin()],5))
                    kT_LW_at_R500_models.append(kT_LW_at_R500)
                    
            # self.weighted_kT_LW_spread = (np.std(weighted_kT_LW_models,axis=0) * u.keV).to("keV")        
            self.weighted_kT_LW_spread = (abs(np.percentile(weighted_kT_LW_models,(16,84), axis=0)*u.keV - self.weighted_kT_LW)).to("keV")
            self.kT_LW_at_LW_HSE_R500 = (self.kT_model(self.LW_HSE_R500.to('kpc').value, **self.LW_kT_model_best_fit_pars) * u.keV).to("keV")
            self.kT_LW_at_LW_HSE_R500_spread = (abs(np.percentile(kT_LW_at_R500_models,(16,84), axis=0)*u.keV - self.kT_LW_at_LW_HSE_R500)).to("keV")
            
            
        
    def calculate_weighted_S_Xray(self):
        r_in = self.inner_R500_frac*self.Xray_HSE_R500.to('kpc').value
        r_out = 1*self.Xray_HSE_R500.to('kpc').value

        
        def _S_numerator_integrand(r, Xray_kT_model_best_fit_pars, Xray_ne_model_best_fit_pars):
            kT_Xray_ml_yval = float(self.kT_model(r, **Xray_kT_model_best_fit_pars))
            ne_Xray_ml_yval = float(self.ne_model(r, **Xray_ne_model_best_fit_pars))
            entropy =  kT_Xray_ml_yval/(ne_Xray_ml_yval**(2/3))
            num_integrand = entropy * (ne_Xray_ml_yval**2) * (r**2)
            return num_integrand
        def _S_denom_integrand(r, Xray_ne_model_best_fit_pars):
            ne_Xray_ml_yval = float(self.ne_model(r, **Xray_ne_model_best_fit_pars))
            den_integrand = (ne_Xray_ml_yval**2) * (r**2)
            return den_integrand
        
        weighted_S = scipy.integrate.quad(_S_numerator_integrand, r_in, r_out, args = (self.Xray_kT_model_best_fit_pars, self.Xray_ne_model_best_fit_pars) )
        normalisation = scipy.integrate.quad(_S_denom_integrand, r_in, r_out, args = (self.Xray_ne_model_best_fit_pars,) )
        self.weighted_S_Xray = weighted_S[0]/normalisation[0]
        self.weighted_S_Xray *= (u.keV*u.cm**2)

        if self.do_MCMC:

            def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
                HSE_log_r = np.log10(HSE_r.value)
                log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
                log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
                mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
                term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
                term2 = log_ne_dash 
                term3 = log_kT_dash 
                mass = (term1 *(term2 + term3)).to("solMass")
                mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
                return mass, mean_rho
            weighted_S_Xray_models = []
            S_Xray_at_R500_models = []
            ### Do an effective outer product of the first N samples drawn from each of the posterior distributions of kT and ne
            for i in range(self._mcmc_mass_samples):
                    sample_kT_pars =  self.Xray_kT_thetas.to_dict(orient = 'records')[i]
                # for j in range(self._mcmc_mass_samples):
                    # sample_ne_pars =  self.Xray_ne_thetas.to_dict(orient = 'records')[j]
                    sample_ne_pars =  self.Xray_ne_thetas.to_dict(orient = 'records')[i]
                    log_r = np.log10(self.r_values_fine.value)
                    log_kT = np.log10(self.kT_model(self.r_values_fine.value, **sample_kT_pars))
                    log_ne = np.log10(self.ne_model(self.r_values_fine.value, **sample_ne_pars))        
                    _, sample_mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    sample_HSE_R500 = self.r_values_fine[np.abs(  (sample_mean_rho - 500*self.rho_crit).value).argmin()]                  
                    sample_r_in = self.inner_R500_frac*sample_HSE_R500.to('kpc').value
                    sample_r_out = 1*sample_HSE_R500.to('kpc').value
                    weighted_S = scipy.integrate.quad(_S_numerator_integrand, sample_r_in, sample_r_out, args = (sample_kT_pars, sample_ne_pars) )
                    normalisation = scipy.integrate.quad(_S_denom_integrand, sample_r_in, sample_r_out, args = (sample_ne_pars,) )
                    weighted_S_Xray = weighted_S[0]/normalisation[0]
                    weighted_S_Xray_models.append(weighted_S_Xray)     
                    kT_Xray_at_R500 = self.kT_model(sample_HSE_R500.value, **sample_kT_pars)
                    ne_Xray_at_R500 = self.ne_model(sample_HSE_R500.value, **sample_ne_pars)
                    S_Xray_at_R500 = kT_Xray_at_R500/(ne_Xray_at_R500**(2/3))
                    S_Xray_at_R500_models.append(S_Xray_at_R500)
                    
            self.weighted_S_Xray_spread = (abs(np.percentile(weighted_S_Xray_models,(16,84), axis=0)*(u.keV*u.cm**2) - self.weighted_S_Xray)).to("keV*cm**2")
            self.S_Xray_at_Xray_HSE_R500 = ((self.kT_model(self.Xray_HSE_R500.to('kpc').value, **self.Xray_kT_model_best_fit_pars)/(self.ne_model(self.Xray_HSE_R500.to('kpc').value, **self.Xray_ne_model_best_fit_pars)**(2/3)))*(u.keV*u.cm**2)).to("keV*cm**2")
            self.S_Xray_at_Xray_HSE_R500_spread = (abs(np.percentile(S_Xray_at_R500_models,(16,84), axis=0)*(u.keV*u.cm**2) - self.S_Xray_at_Xray_HSE_R500)).to("keV*cm**2")

    def calculate_weighted_S_LW(self):
        r_in = self.inner_R500_frac*self.LW_HSE_R500.to('kpc').value
        r_out = 1*self.LW_HSE_R500.to('kpc').value

        
        def _S_numerator_integrand(r, LW_kT_model_best_fit_pars, LW_ne_model_best_fit_pars):
            kT_LW_ml_yval = float(self.kT_model(r, **LW_kT_model_best_fit_pars))
            ne_LW_ml_yval = float(self.ne_model(r, **LW_ne_model_best_fit_pars))
            entropy =  kT_LW_ml_yval/(ne_LW_ml_yval**(2/3))
            num_integrand = entropy * (ne_LW_ml_yval**2) * (r**2)
            return num_integrand
        def _S_denom_integrand(r, LW_ne_model_best_fit_pars):
            ne_LW_ml_yval = float(self.ne_model(r, **LW_ne_model_best_fit_pars))
            den_integrand = (ne_LW_ml_yval**2) * (r**2)
            return den_integrand
        
        weighted_S = scipy.integrate.quad(_S_numerator_integrand, r_in, r_out, args = (self.LW_kT_model_best_fit_pars, self.LW_ne_model_best_fit_pars) )
        normalisation = scipy.integrate.quad(_S_denom_integrand, r_in, r_out, args = (self.LW_ne_model_best_fit_pars,) )
        self.weighted_S_LW = weighted_S[0]/normalisation[0]
        self.weighted_S_LW *= (u.keV*u.cm**2)

        if self.do_MCMC:

            def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
                HSE_log_r = np.log10(HSE_r.value)
                log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
                log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
                mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
                term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
                term2 = log_ne_dash 
                term3 = log_kT_dash 
                mass = (term1 *(term2 + term3)).to("solMass")
                mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
                return mass, mean_rho
            weighted_S_LW_models = []
            S_LW_at_R500_models = []
            ### Do an effective outer product of the first N samples drawn from each of the posterior distributions of kT and ne
            for i in range(self._mcmc_mass_samples):
                    sample_kT_pars =  self.LW_kT_thetas.to_dict(orient = 'records')[i]
                # for j in range(self._mcmc_mass_samples):
                    # sample_ne_pars =  self.Xray_ne_thetas.to_dict(orient = 'records')[j]
                    sample_ne_pars =  self.LW_ne_thetas.to_dict(orient = 'records')[i]
                    log_r = np.log10(self.r_values_fine.value)
                    log_kT = np.log10(self.kT_model(self.r_values_fine.value, **sample_kT_pars))
                    log_ne = np.log10(self.ne_model(self.r_values_fine.value, **sample_ne_pars))        
                    _, sample_mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    sample_HSE_R500 = self.r_values_fine[np.abs(  (sample_mean_rho - 500*self.rho_crit).value).argmin()]                  
                    sample_r_in = self.inner_R500_frac*sample_HSE_R500.to('kpc').value
                    sample_r_out = 1*sample_HSE_R500.to('kpc').value
                    weighted_S = scipy.integrate.quad(_S_numerator_integrand, sample_r_in, sample_r_out, args = (sample_kT_pars, sample_ne_pars) )
                    normalisation = scipy.integrate.quad(_S_denom_integrand, sample_r_in, sample_r_out, args = (sample_ne_pars,) )
                    weighted_S_LW = weighted_S[0]/normalisation[0]
                    weighted_S_LW_models.append(weighted_S_LW)     
                    kT_LW_at_R500 = self.kT_model(sample_HSE_R500.value, **sample_kT_pars)
                    ne_LW_at_R500 = self.ne_model(sample_HSE_R500.value, **sample_ne_pars)
                    S_LW_at_R500 = kT_LW_at_R500/(ne_LW_at_R500**(2/3))
                    S_LW_at_R500_models.append(S_LW_at_R500)
                    
            self.weighted_S_LW_spread = (abs(np.percentile(weighted_S_LW_models,(16,84), axis=0)*(u.keV*u.cm**2) - self.weighted_S_LW)).to("keV*cm**2")
            self.S_LW_at_LW_HSE_R500 = ((self.kT_model(self.LW_HSE_R500.to('kpc').value, **self.LW_kT_model_best_fit_pars)/(self.ne_model(self.LW_HSE_R500.to('kpc').value, **self.LW_ne_model_best_fit_pars)**(2/3)))*(u.keV*u.cm**2)).to("keV*cm**2")
            self.S_LW_at_LW_HSE_R500_spread = (abs(np.percentile(S_LW_at_R500_models,(16,84), axis=0)*(u.keV*u.cm**2) - self.S_LW_at_LW_HSE_R500)).to("keV*cm**2")
    
    
    
    def calculate_weighted_S_EW(self):
        r_in = self.inner_R500_frac*self.EW_HSE_R500.to('kpc').value
        r_out = 1*self.EW_HSE_R500.to('kpc').value

        
        def _S_numerator_integrand(r, EW_kT_model_best_fit_pars, EW_ne_model_best_fit_pars):
            kT_EW_ml_yval = float(self.kT_model(r, **EW_kT_model_best_fit_pars))
            ne_EW_ml_yval = float(self.ne_model(r, **EW_ne_model_best_fit_pars))
            entropy =  kT_EW_ml_yval/(ne_EW_ml_yval**(2/3))
            num_integrand = entropy * (ne_EW_ml_yval**2) * (r**2)
            return num_integrand
        def _S_denom_integrand(r, EW_ne_model_best_fit_pars):
            ne_EW_ml_yval = float(self.ne_model(r, **EW_ne_model_best_fit_pars))
            den_integrand = (ne_EW_ml_yval**2) * (r**2)
            return den_integrand
        
        weighted_S = scipy.integrate.quad(_S_numerator_integrand, r_in, r_out, args = (self.EW_kT_model_best_fit_pars, self.EW_ne_model_best_fit_pars) )
        normalisation = scipy.integrate.quad(_S_denom_integrand, r_in, r_out, args = (self.EW_ne_model_best_fit_pars,) )
        self.weighted_S_EW = weighted_S[0]/normalisation[0]
        self.weighted_S_EW *= (u.keV*u.cm**2)

        if self.do_MCMC:

            def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
                HSE_log_r = np.log10(HSE_r.value)
                log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
                log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
                mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
                term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
                term2 = log_ne_dash 
                term3 = log_kT_dash 
                mass = (term1 *(term2 + term3)).to("solMass")
                mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
                return mass, mean_rho
            weighted_S_EW_models = []
            S_EW_at_R500_models = []
            ### Do an effective outer product of the first N samples drawn from each of the posterior distributions of kT and ne
            for i in range(self._mcmc_mass_samples):
                    sample_kT_pars =  self.EW_kT_thetas.to_dict(orient = 'records')[i]
                # for j in range(self._mcmc_mass_samples):
                    # sample_ne_pars =  self.Xray_ne_thetas.to_dict(orient = 'records')[j]
                    sample_ne_pars =  self.EW_ne_thetas.to_dict(orient = 'records')[i]
                    log_r = np.log10(self.r_values_fine.value)
                    log_kT = np.log10(self.kT_model(self.r_values_fine.value, **sample_kT_pars))
                    log_ne = np.log10(self.ne_model(self.r_values_fine.value, **sample_ne_pars))        
                    _, sample_mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    sample_HSE_R500 = self.r_values_fine[np.abs(  (sample_mean_rho - 500*self.rho_crit).value).argmin()]                  
                    sample_r_in = self.inner_R500_frac*sample_HSE_R500.to('kpc').value
                    sample_r_out = 1*sample_HSE_R500.to('kpc').value
                    weighted_S = scipy.integrate.quad(_S_numerator_integrand, sample_r_in, sample_r_out, args = (sample_kT_pars, sample_ne_pars) )
                    normalisation = scipy.integrate.quad(_S_denom_integrand, sample_r_in, sample_r_out, args = (sample_ne_pars,) )
                    weighted_S_EW = weighted_S[0]/normalisation[0]
                    weighted_S_EW_models.append(weighted_S_EW)     
                    kT_EW_at_R500 = self.kT_model(sample_HSE_R500.value, **sample_kT_pars)
                    ne_EW_at_R500 = self.ne_model(sample_HSE_R500.value, **sample_ne_pars)
                    S_EW_at_R500 = kT_EW_at_R500/(ne_EW_at_R500**(2/3))
                    S_EW_at_R500_models.append(S_EW_at_R500)
                    
            self.weighted_S_EW_spread = (abs(np.percentile(weighted_S_EW_models,(16,84), axis=0)*(u.keV*u.cm**2) - self.weighted_S_EW)).to("keV*cm**2")
            self.S_EW_at_EW_HSE_R500 = ((self.kT_model(self.EW_HSE_R500.to('kpc').value, **self.EW_kT_model_best_fit_pars)/(self.ne_model(self.EW_HSE_R500.to('kpc').value, **self.EW_ne_model_best_fit_pars)**(2/3)))*(u.keV*u.cm**2)).to("keV*cm**2")
            self.S_EW_at_EW_HSE_R500_spread = (abs(np.percentile(S_EW_at_R500_models,(16,84), axis=0)*(u.keV*u.cm**2) - self.S_EW_at_EW_HSE_R500)).to("keV*cm**2")
            
            
            
    def calculate_weighted_S_MW(self):
        r_in = self.inner_R500_frac*self.MW_HSE_R500.to('kpc').value
        r_out = 1*self.MW_HSE_R500.to('kpc').value

        
        def _S_numerator_integrand(r, MW_kT_model_best_fit_pars, MW_ne_model_best_fit_pars):
            kT_MW_ml_yval = float(self.kT_model(r, **MW_kT_model_best_fit_pars))
            ne_MW_ml_yval = float(self.ne_model(r, **MW_ne_model_best_fit_pars))
            entropy =  kT_MW_ml_yval/(ne_MW_ml_yval**(2/3))
            num_integrand = entropy * (ne_MW_ml_yval**2) * (r**2)
            return num_integrand
        def _S_denom_integrand(r, MW_ne_model_best_fit_pars):
            ne_MW_ml_yval = float(self.ne_model(r, **MW_ne_model_best_fit_pars))
            den_integrand = (ne_MW_ml_yval**2) * (r**2)
            return den_integrand
        
        weighted_S = scipy.integrate.quad(_S_numerator_integrand, r_in, r_out, args = (self.MW_kT_model_best_fit_pars, self.MW_ne_model_best_fit_pars) )
        normalisation = scipy.integrate.quad(_S_denom_integrand, r_in, r_out, args = (self.MW_ne_model_best_fit_pars,) )
        self.weighted_S_MW = weighted_S[0]/normalisation[0]
        self.weighted_S_MW *= (u.keV*u.cm**2)

        if self.do_MCMC:

            def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
                HSE_log_r = np.log10(HSE_r.value)
                log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
                log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
                mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
                term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
                term2 = log_ne_dash 
                term3 = log_kT_dash 
                mass = (term1 *(term2 + term3)).to("solMass")
                mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
                return mass, mean_rho
            weighted_S_MW_models = []
            S_MW_at_R500_models = []
            ### Do an effective outer product of the first N samples drawn from each of the posterior distributions of kT and ne
            for i in range(self._mcmc_mass_samples):
                    sample_kT_pars =  self.MW_kT_thetas.to_dict(orient = 'records')[i]
                # for j in range(self._mcmc_mass_samples):
                    # sample_ne_pars =  self.Xray_ne_thetas.to_dict(orient = 'records')[j]
                    sample_ne_pars =  self.MW_ne_thetas.to_dict(orient = 'records')[i]
                    log_r = np.log10(self.r_values_fine.value)
                    log_kT = np.log10(self.kT_model(self.r_values_fine.value, **sample_kT_pars))
                    log_ne = np.log10(self.ne_model(self.r_values_fine.value, **sample_ne_pars))        
                    _, sample_mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    sample_HSE_R500 = self.r_values_fine[np.abs(  (sample_mean_rho - 500*self.rho_crit).value).argmin()]                  
                    sample_r_in = self.inner_R500_frac*sample_HSE_R500.to('kpc').value
                    sample_r_out = 1*sample_HSE_R500.to('kpc').value
                    weighted_S = scipy.integrate.quad(_S_numerator_integrand, sample_r_in, sample_r_out, args = (sample_kT_pars, sample_ne_pars) )
                    normalisation = scipy.integrate.quad(_S_denom_integrand, sample_r_in, sample_r_out, args = (sample_ne_pars,) )
                    weighted_S_MW = weighted_S[0]/normalisation[0]
                    weighted_S_MW_models.append(weighted_S_MW)     
                    kT_MW_at_R500 = self.kT_model(sample_HSE_R500.value, **sample_kT_pars)
                    ne_MW_at_R500 = self.ne_model(sample_HSE_R500.value, **sample_ne_pars)
                    S_MW_at_R500 = kT_MW_at_R500/(ne_MW_at_R500**(2/3))
                    S_MW_at_R500_models.append(S_MW_at_R500)
                    
            self.weighted_S_MW_spread = (abs(np.percentile(weighted_S_MW_models,(16,84), axis=0)*(u.keV*u.cm**2) - self.weighted_S_MW)).to("keV*cm**2")
            self.S_MW_at_MW_HSE_R500 = ((self.kT_model(self.MW_HSE_R500.to('kpc').value, **self.MW_kT_model_best_fit_pars)/(self.ne_model(self.MW_HSE_R500.to('kpc').value, **self.MW_ne_model_best_fit_pars)**(2/3)))*(u.keV*u.cm**2)).to("keV*cm**2")
            self.S_MW_at_MW_HSE_R500_spread = (abs(np.percentile(S_MW_at_R500_models,(16,84), axis=0)*(u.keV*u.cm**2) - self.S_MW_at_MW_HSE_R500)).to("keV*cm**2")            
            
            
            
            
            
    def calculate_Xray_derived_mass_profile(self):      
        
        def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
            HSE_log_r = np.log10(HSE_r.value)
            log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
            log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
            mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
            term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
            term2 = log_ne_dash 
            term3 = log_kT_dash 
            mass = (term1 *(term2 + term3)).to("solMass")
            mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
            return mass, mean_rho

        log_r = np.log10(self.r_values_fine.value)
        log_kT = np.log10(self.kT_model(self.r_values_fine.value, **self.Xray_kT_model_best_fit_pars))
        log_ne = np.log10(self.ne_model(self.r_values_fine.value, **self.Xray_ne_model_best_fit_pars))        
        mass, mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
        self.Xray_HSE_total_mass_profile = np.copy(mass)
        dx = np.diff(mass)
        # print("dx:", dx)
        monotonic = np.all(dx[1:] > 0)
        if not monotonic:
            self._logger.warning("Mass profile not monotonic!!")
        self.Xray_HSE_R500 = self.r_values_fine[np.abs(  (mean_rho - 500*self.rho_crit).value).argmin()]
        self.Xray_HSE_R200 = self.r_values_fine[np.abs(  (mean_rho - 200*self.rho_crit).value).argmin()]
        self.Xray_HSE_M500 = mass[np.abs(  (mean_rho - 500*self.rho_crit).value).argmin()]
        self.Xray_HSE_M200 = mass[np.abs(  (mean_rho - 200*self.rho_crit).value).argmin()]
        
        if self.do_MCMC:
            mass_models = []
            M500_models = []
            R500_models = []

            for i in range(self._mcmc_mass_samples):
                    log_kT = np.log10(self.Xray_kT_models[i])
                # for j in range(self._mcmc_mass_samples):
                    log_ne = np.log10(self.Xray_ne_models[i])
                    mass_mod, rho_mod = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    mass_models.append(np.array([x.value for x in mass_mod]))
                    mean_rho = mass_mod / ( (4/3) * math.pi * np.power(self.r_values_fine,3) )
                    M500_model = mass[np.abs(  (mean_rho - 500*self.rho_crit).value).argmin()]
                    M500_models.append(M500_model.value)
                    R500_model = self.r_values_fine[np.abs(  (mean_rho - 500*self.rho_crit).value).argmin()]
                    R500_models.append(R500_model.value)

            self.Xray_HSE_mass_spread = abs((np.percentile(mass_models,(16,84), axis=0) * u.Msun   -  self.Xray_HSE_total_mass_profile)).to("solMass") 
            self.Xray_HSE_M500_spread =abs((np.percentile(M500_models,(16,84), axis=0) * u.Msun).to("solMass") - self.Xray_HSE_M500)
            self.Xray_HSE_R500_spread =abs((np.percentile(R500_models,(16,84), axis=0) * u.kpc).to("kpc") - self.Xray_HSE_R500)
            
        
    def calculate_EW_derived_mass_profile(self):
        def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
            HSE_log_r = np.log10(HSE_r.value)
            log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
            log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
            mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
            term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
            term2 = log_ne_dash 
            term3 = log_kT_dash 
            mass = (term1 *(term2 + term3)).to("solMass")
            mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
            return mass, mean_rho

        log_r = np.log10(self.r_values_fine.value)
        log_kT = np.log10(self.kT_model(self.r_values_fine.value, **self.EW_kT_model_best_fit_pars))
        log_ne = np.log10(self.ne_model(self.r_values_fine.value, **self.EW_ne_model_best_fit_pars))        
        mass, mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
        self.EW_HSE_total_mass_profile = np.copy(mass)
        self.EW_HSE_R500 = self.r_values_fine[np.abs(  (mean_rho - 500*self.rho_crit).value).argmin()]
        self.EW_HSE_R200 = self.r_values_fine[np.abs(  (mean_rho - 200*self.rho_crit).value).argmin()]
        self.EW_HSE_M500 = mass[np.abs(  (mean_rho - 500*self.rho_crit).value).argmin()]
        self.EW_HSE_M200 = mass[np.abs(  (mean_rho - 200*self.rho_crit).value).argmin()]   
        
        if self.do_MCMC:
            mass_models = []
            M500_models = []
            for i in range(self._mcmc_mass_samples):
                    log_kT = np.log10(self.EW_kT_models[i])
                # for j in range(self._mcmc_mass_samples):
                    log_ne = np.log10(self.EW_ne_models[i])
                    mass_mod, rho_mod = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    mass_models.append(np.array([x.value for x in mass_mod]))
                    mean_rho = mass_mod / ( (4/3) * math.pi * np.power(self.r_values_fine,3) )
                    M500_model = mass[np.abs(  (mean_rho - 500*self.rho_crit).value).argmin()]
                    M500_models.append(M500_model.value)

            self.EW_HSE_mass_spread = (np.percentile(mass_models,(16,84), axis=0) * u.Msun).to("solMass")
            self.EW_HSE_M500_spread = abs((np.percentile(M500_models,(16,84), axis=0) * u.Msun).to("solMass") - self.EW_HSE_M500)

    
        
    def calculate_MW_derived_mass_profile(self):
        def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
            HSE_log_r = np.log10(HSE_r.value)
            log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
            log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
            mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
            term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
            term2 = log_ne_dash 
            term3 = log_kT_dash 
            mass = (term1 *(term2 + term3)).to("solMass")
            mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
            return mass, mean_rho

        log_r = np.log10(self.r_values_fine.value)
        log_kT = np.log10(self.kT_model(self.r_values_fine.value, **self.MW_kT_model_best_fit_pars))
        log_ne = np.log10(self.ne_model(self.r_values_fine.value, **self.MW_ne_model_best_fit_pars))        
        mass, mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
        self.MW_HSE_total_mass_profile = np.copy(mass)
        self.MW_HSE_R500 = self.r_values_fine[np.abs(  (mean_rho - 500*self.rho_crit).value).argmin()]
        self.MW_HSE_R200 = self.r_values_fine[np.abs(  (mean_rho - 200*self.rho_crit).value).argmin()]
        self.MW_HSE_M500 = mass[np.abs(  (mean_rho - 500*self.rho_crit).value).argmin()]
        self.MW_HSE_M200 = mass[np.abs(  (mean_rho - 200*self.rho_crit).value).argmin()]   
        
        if self.do_MCMC:
            mass_models = []
            M500_models = []
            for i in range(self._mcmc_mass_samples):
                    log_kT = np.log10(self.MW_kT_models[i])
                # for j in range(self._mcmc_mass_samples):
                    log_ne = np.log10(self.MW_ne_models[i])
                    mass_mod, rho_mod = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    mass_models.append(np.array([x.value for x in mass_mod]))
                    mean_rho = mass_mod / ( (4/3) * math.pi * np.power(self.r_values_fine,3) )
                    M500_model = mass[np.abs(  (mean_rho - 500*self.rho_crit).value).argmin()]
                    M500_models.append(M500_model.value)

            self.MW_HSE_mass_spread = (np.percentile(mass_models,(16,84), axis=0) * u.Msun).to("solMass")
            self.MW_HSE_M500_spread = abs((np.percentile(M500_models,(16,84), axis=0) * u.Msun).to("solMass") - self.MW_HSE_M500)
  

    def calculate_LW_derived_mass_profile(self):
        def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
            HSE_log_r = np.log10(HSE_r.value)
            log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
            log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
            mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
            term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
            term2 = log_ne_dash 
            term3 = log_kT_dash 
            mass = (term1 *(term2 + term3)).to("solMass")
            mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
            return mass, mean_rho

        log_r = np.log10(self.r_values_fine.value)
        log_kT = np.log10(self.kT_model(self.r_values_fine.value, **self.LW_kT_model_best_fit_pars))
        log_ne = np.log10(self.ne_model(self.r_values_fine.value, **self.LW_ne_model_best_fit_pars))        
        mass, mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
        self.LW_HSE_total_mass_profile = np.copy(mass)
        self.LW_HSE_R500 = self.r_values_fine[np.abs(  (mean_rho - 500*self.rho_crit).value).argmin()]
        self.LW_HSE_R200 = self.r_values_fine[np.abs(  (mean_rho - 200*self.rho_crit).value).argmin()]
        self.LW_HSE_M500 = mass[np.abs(  (mean_rho - 500*self.rho_crit).value).argmin()]
        self.LW_HSE_M200 = mass[np.abs(  (mean_rho - 200*self.rho_crit).value).argmin()]   
        
        if self.do_MCMC:
            mass_models = []
            M500_models = []

            for i in range(self._mcmc_mass_samples):
                    log_kT = np.log10(self.LW_kT_models[i])
                # for j in range(self._mcmc_mass_samples):
                    log_ne = np.log10(self.LW_ne_models[i])
                    mass_mod, rho_mod = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
                    mass_models.append(np.array([x.value for x in mass_mod]))
                    mean_rho = mass_mod / ( (4/3) * math.pi * np.power(self.r_values_fine,3) )
                    M500_model = mass[np.abs(  (mean_rho - 500*self.rho_crit).value).argmin()]
                    M500_models.append(M500_model.value)

            self.LW_HSE_mass_spread = (np.percentile(mass_models,(16,84), axis=0) * u.Msun).to("solMass")
            self.LW_HSE_M500_spread = abs((np.percentile(M500_models,(16,84), axis=0) * u.Msun).to("solMass") - self.LW_HSE_M500)

    def calculate_Xray_ODR_derived_mass_profile(self,):

        def _HSE_formula(HSE_log_kT,HSE_log_ne,  HSE_r):
            HSE_log_r = np.log10(HSE_r.value)
            log_kT_dash = np.gradient(HSE_log_kT, HSE_log_r)
            log_ne_dash = np.gradient(HSE_log_ne, HSE_log_r)
            mu = 0.59 #0.59 #https://arxiv.org/pdf/2001.11508.pdf
            term1 = -( HSE_r  * (10**(HSE_log_kT) * u.keV)/(G * m_p * mu))
            term2 = log_ne_dash 
            term3 = log_kT_dash 
            mass = (term1 *(term2 + term3)).to("solMass")
            mean_rho = mass/ ( (4/3) * math.pi * np.power(HSE_r,3) )
            return mass, mean_rho

        log_r = np.log10(self.r_values_fine.value)
        log_kT = np.log10(self.kT_model(self.r_values_fine.value, **self.Xray_ODR_kT_model_best_fit_pars))
        log_ne = np.log10(self.ne_model(self.r_values_fine.value, **self.Xray_ODR_ne_model_best_fit_pars))        
        mass, mean_rho = _HSE_formula(log_kT, log_ne,  self.r_values_fine)
        self.Xray_ODR_HSE_total_mass_profile = np.copy(mass)
        dx = np.diff(mass)
        # print("dx:", dx)
        monotonic = np.all(dx[1:] > 0)
        if not monotonic:
            self._logger.warning("Mass profile not monotonic!!")
        self.Xray_ODR_HSE_R500 = self.r_values_fine[np.abs(  (mean_rho - 500*self.rho_crit).value).argmin()]
        self.Xray_ODR_HSE_R200 = self.r_values_fine[np.abs(  (mean_rho - 200*self.rho_crit).value).argmin()]
        self.Xray_ODR_HSE_M500 = mass[np.abs(  (mean_rho - 500*self.rho_crit).value).argmin()]
        self.Xray_ODR_HSE_M200 = mass[np.abs(  (mean_rho - 200*self.rho_crit).value).argmin()]               
            
        
    def calculate_MW_derived_entropy(self):  ### This is for profiles, NOT a weighted value
        kT = self.kT_model(self.r_values_fine.value, **self.MW_kT_model_best_fit_pars) * u.keV
        ne = self.ne_model(self.r_values_fine.value, **self.MW_ne_model_best_fit_pars) * u.cm**-3      
        self.MW_entropy_profile = kT/(ne**(2/3))
        if self.do_MCMC:
            k_models = []
            for i in range(self._mcmc_mass_samples):
                    kT = self.MW_kT_models[i] * u.keV
                # for j in range(self._mcmc_mass_samples):
                    ne = self.MW_ne_models[i] * u.cm**-3
                    MW_entropy_profile = kT/(ne**(2/3))
                    k_models.append(MW_entropy_profile)
            # self.MW_entropy_spread = np.std(k_models,axis=0) * k_models[0].unit
            self.MW_entropy_spread = (abs(np.percentile(k_models,(16,84), axis=0)*k_models[0].unit) - self.MW_entropy_profile).to("keV*cm**2")
        

    def calculate_Xray_derived_entropy(self):  ### This is for profiles, NOT a weighted value
        kT = self.kT_model(self.r_values_fine.value, **self.Xray_kT_model_best_fit_pars) * u.keV
        ne = self.ne_model(self.r_values_fine.value, **self.Xray_ne_model_best_fit_pars) * u.cm**-3      
        self.Xray_entropy_profile = kT/(ne**(2/3))
        if self.do_MCMC:
            k_models = []
            for i in range(self._mcmc_mass_samples):
                    kT = self.Xray_kT_models[i] * u.keV
                # for j in range(self._mcmc_mass_samples):
                    ne = self.Xray_ne_models[i] * u.cm**-3
                    Xray_entropy_profile = kT/(ne**(2/3))
                    k_models.append(Xray_entropy_profile)
            # self.Xray_entropy_spread = np.std(k_models,axis=0) * k_models[0].unit
            self.Xray_entropy_spread = (abs(np.percentile(k_models,(16,84), axis=0)*k_models[0].unit) - self.Xray_entropy_profile).to("keV*cm**2")

            
            
            
            
            
    def calculate_EW_derived_entropy(self):  ### This is for profiles, NOT a weighted value
        kT = self.kT_model(self.r_values_fine.value, **self.EW_kT_model_best_fit_pars) * u.keV
        ne = self.ne_model(self.r_values_fine.value, **self.EW_ne_model_best_fit_pars) * u.cm**-3      
        self.EW_entropy_profile = kT/(ne**(2/3))
        if self.do_MCMC:
            k_models = []
            for i in range(self._mcmc_mass_samples):
                    kT = self.EW_kT_models[i] * u.keV
                # for j in range(self._mcmc_mass_samples):
                    ne = self.EW_ne_models[i] * u.cm**-3
                    EW_entropy_profile = kT/(ne**(2/3))
                    k_models.append(EW_entropy_profile)
            # self.EW_entropy_spread = np.std(k_models,axis=0) * k_models[0].unit
            self.EW_entropy_spread = (abs(np.percentile(k_models,(16,84), axis=0)*k_models[0].unit) - self.EW_entropy_profile).to("keV*cm**2")
        
    def calculate_LW_derived_entropy(self):  ### This is for profiles, NOT a weighted value
        kT = self.kT_model(self.r_values_fine.value, **self.LW_kT_model_best_fit_pars) * u.keV
        ne = self.ne_model(self.r_values_fine.value, **self.LW_ne_model_best_fit_pars) * u.cm**-3      
        self.LW_entropy_profile = kT/(ne**(2/3))
        if self.do_MCMC:
            k_models = []
            for i in range(self._mcmc_mass_samples):
                    kT = self.LW_kT_models[i] * u.keV
                # for j in range(self._mcmc_mass_samples):
                    ne = self.LW_ne_models[i] * u.cm**-3
                    LW_entropy_profile = kT/(ne**(2/3))
                    k_models.append(LW_entropy_profile)
            # self.LW_entropy_spread = np.std(k_models,axis=0) * k_models[0].unit
            self.LW_entropy_spread = (abs(np.percentile(k_models,(16,84), axis=0)*k_models[0].unit) - self.LW_entropy_profile).to("keV*cm**2")
    
        
    def calculate_MW_derived_pressure(self):  
        kT = self.kT_model(self.r_values_fine.value, **self.MW_kT_model_best_fit_pars) * u.keV
        ne = self.ne_model(self.r_values_fine.value, **self.MW_ne_model_best_fit_pars) * u.cm**-3      
        self.MW_pressure_profile = kT*ne           

    def calculate_EW_derived_pressure(self):  
        kT = self.kT_model(self.r_values_fine.value, **self.EW_kT_model_best_fit_pars) * u.keV
        ne = self.ne_model(self.r_values_fine.value, **self.EW_ne_model_best_fit_pars) * u.cm**-3      
        self.EW_pressure_profile = kT*ne   
            
    def calculate_LW_derived_pressure(self):  
        kT = self.kT_model(self.r_values_fine.value, **self.LW_kT_model_best_fit_pars) * u.keV
        ne = self.ne_model(self.r_values_fine.value, **self.LW_ne_model_best_fit_pars) * u.cm**-3      
        self.LW_pressure_profile = kT*ne   
        

    def calculate_Xray_derived_pressure(self):  
        kT = self.kT_model(self.r_values_fine.value, **self.Xray_kT_model_best_fit_pars) * u.keV
        ne = self.ne_model(self.r_values_fine.value, **self.Xray_ne_model_best_fit_pars) * u.cm**-3      
        self.Xray_pressure_profile = kT*ne
            
            
    @staticmethod
    def set_plotstyle():
        plt.rcParams.update({ "text.usetex": False})
        
        # print("font.monospace", plt.rcParams["font.monospace"])
        font = {'family' : 'monospace',
                'weight' : 'normal',
                'size'   : 45}
        plt.rc("font", **font)
        # plt.rc("text",usetex = True )
        plt.rc("axes", linewidth=2)
        plt.rc("xtick.major", width=3, size=20)
        plt.rc("xtick.minor", width=2, size=10)
        plt.rc("ytick.major", width=3, size=20)
        plt.rc("ytick.minor", width=2, size=10)
        
        plt.rcParams["xtick.labelsize"] = 35
        plt.rcParams["ytick.labelsize"] = 35
        plt.rcParams["legend.fontsize"] =  30
        plt.rcParams["legend.framealpha"] =  0.2
        # print("font.monospace", plt.rcParams["font.monospace"])
        # plt.rcParams["font.monospace"] = "Courier"
        
        
        
  
 

    def check_near_host(self, halo_position_list, halo_R500_list, halo_M500_list, N = 5):
        self.position = halo_position_list[self.halo_idx]
        if halo_R500_list[self.halo_idx].to("kpc") != self.R500_truth.to("kpc"):
            print("Warning! R500 at halo index does not agree with R500 for this halo")
            return
        if halo_M500_list[self.halo_idx].to("solMass") != self.M500_truth.to("solMass"):
            print("Warning! M500 at halo index does not agree with M500 for this halo")
            return  
        
        #Test all more massive halos for closeness
        min_sep = 1e10 * kpc
        print("checking near halo")
        for i,_ in tqdm(enumerate(halo_R500_list)):
            if halo_M500_list[i].to("solMass") <= self.M500_truth.to("solMass"):
                continue
            else:
                test_halo_R500 = halo_R500_list[i].to('kpc')
                sep =  np.linalg.norm( halo_position_list[i].to("kpc") - self.position.to("kpc")  )
                min_sep = min(min_sep, sep)
                if sep < (N * self.R500_truth.to("kpc")) + test_halo_R500 :
                    print(f"Halo too close to halo at index {i}. Halo at index {i} is at position {halo_position_list[i]} which is too close to the current halo which is at {self.position} (separation = {sep})")
                    return True
        print(f"No halos found nearby. Nearest halo was at a distance of {min_sep} kpc. R500 for reference is {self.R500_truth.to('kpc') }.")
        return False

    def highlight_suspect_halo(self, PNGs_path = "./PNGS/"):
        files = [x for x in os.listdir(PNGs_path) if f"{self.snap_num}_{self.halo_idx}_{self.instrument}" in x ]
        print("File to move", files)
        for file in files:
            
#             shutil.copy(PNGs_path+file,"./suspect_halos/"+file)
            shutil.copytree(PNGs_path+file,"./suspect_halos/"+file)



    def measure_Xray_flux(self, emin, emax, nsteps = 10):
        '''
        To calculate flux/luminosity we create linearly-spaced energy bins between emin and emax and use SOXS to create an exposure map at the energy corresponding to the bin midpoint. We then calculate the counts flux in radial (arcsec) bins within the desired aperture (for L_x(R500), we use just a single bin from the center to R500) and multiply by the bin energy midpoint to convert to an energy flux. We repeat for all energy bins and sum the energy fluxes to give a total energy flux, which is then converted to a luminosity via the luminosity distance.
        ------------------------------------------------
        Positional Arguments:

        Keyword Arguments:
 
        Returns: 

        '''
        import soxs   
        soxs.set_soxs_config("soxs_data_dir", "./CODE/instr_files/")
        from soxs.instrument_registry import instrument_registry
        from astropy.units import arcsec, radian
        instrument_spec = instrument_registry[self.instrument_name]
        self._Xray_total_lumin_emin_RF = emin
        self._Xray_total_lumin_emax_RF = emax

        nx = instrument_spec["num_pixels"]
        plate_scale = instrument_spec["fov"]/nx/60. # arcmin to deg
        plate_scale_arcsec = plate_scale * 3600.0   #arcsec per pixel
        if self.instrument_name in ["athena_wfi"]:
            chip_width_arcsec = np.array(instrument_spec["chips"])[1][[3,4]].astype(np.float)* plate_scale_arcsec
        else:
            print("Instrument currently not supported")
            sys.exit()
        chip_radius_arcsec = max(chip_width_arcsec)/2

        img_file = f"{self.evts_path}/{self.idx_instr_tag}_img.fits"  #We never make a cleaned version of this, since it is really time-expensive as opposed to doing it annulus-by-annulus
        evt_file = f"{self.evts_path}/{self.idx_instr_tag}_evt.fits"
        bkgrnd_evt_file = f"{self._bkgrnd_evts_path}/{self._bkgrnd_idx_instr_tag}_evt.fits"
        
        Lx_path = Path(f"{self.annuli_path}/Lx_files")
        os.makedirs(Lx_path, exist_ok = True)
        
        with fits.open(img_file) as hdul:
            # wcs = WCS(hdul[0].header)   
            # center = np.array([float(hdul[0].header['CRPIX1']),float(hdul[0].header['CRPIX2'])] )
            # nx = instrument_registry[self.instrument_name]["num_pixels"]
            # plate_scale = arcsec * instrument_registry[self.instrument_name]["fov"]/nx  ### arcsec per pixel  
            
            ang_diam_dist = self.cosmo.angular_diameter_distance(z = self.redshift)/radian
            Xray_HSE_R500_upper = self.Xray_HSE_R500 + self.Xray_HSE_R500_spread[1]
            Xray_HSE_R500_lower = self.Xray_HSE_R500 - self.Xray_HSE_R500_spread[0]   
            print("Xray_HSE_R500_upper", Xray_HSE_R500_upper)
            print("self.Xray_HSE_R500", self.Xray_HSE_R500)
            print("Xray_HSE_R500_lower", Xray_HSE_R500_lower)
            arcsec_Xray_HSE_R500 = (self.Xray_HSE_R500 / ang_diam_dist).to("arcsec")
            arcsec_Xray_HSE_R500_upper = (Xray_HSE_R500_upper / ang_diam_dist).to("arcsec")
            arcsec_Xray_HSE_R500_lower = (Xray_HSE_R500_lower / ang_diam_dist).to("arcsec")
            
        
        if arcsec_Xray_HSE_R500_upper.to("arcsec").value > chip_radius_arcsec:
            self._logger.warning(f"Halo: {self.halo_idx}:  X-ray measured R500 upper limit in arcsec = {arcsec_Xray_HSE_R500_upper} > chip radius for {self.instrument_name} = {chip_radius_arcsec} at current redshift = {self.redshift}. (Best-fit R500 in arcsec = {arcsec_Xray_HSE_R500} ). Quitting...")
            return   
        else:
            self._logger.info(f"Halo: {self.halo_idx}:  X-ray measured R500 upper limit in arcsec = {arcsec_Xray_HSE_R500_upper} <= chip radius for {self.instrument_name} = {chip_radius_arcsec} at current redshift = {self.redshift}. (Best-fit R500 in arcsec = {arcsec_Xray_HSE_R500} ). ")
             
        
        
            
            
        
        flux_unit = u.keV / u.s / u.cm**2

        flux_Xray_HSE_R500 = 0
        flux_Xray_HSE_R500_upper = 0
        flux_Xray_HSE_R500_lower = 0

        bkgrnd_flux_Xray_HSE_R500 = 0
        bkgrnd_flux_Xray_HSE_R500_upper = 0
        bkgrnd_flux_Xray_HSE_R500_lower = 0

        

        emin_OF = self._Xray_total_lumin_emin_RF / (1+self.redshift)
        emax_OF = self._Xray_total_lumin_emax_RF / (1+self.redshift)
        nsteps += 1
        step = (emax_OF-emin_OF)/(nsteps-1)

        

    
        self._logger.info(f"Will create {nsteps-1} exposure maps for linearly spaced observer-frame energies between {emin_OF} and {emax_OF} keV (corresponding to rest-frame energies {self._Xray_total_lumin_emin_RF} and {self._Xray_total_lumin_emax_RF} keV) to calculate the R500 fluxes")
        for e in np.linspace(emin_OF,emax_OF , nsteps)[:-1]:
            e_low = e
            e_high = min(e + step, emax_OF)
            e_av = 0.5 * (e_low + e_high)
            e_low = round(e_low, 4)
            e_high = round(e_high, 4)
            e_av = round(e_av, 4)
            print(e_low, e_av, e_high,)# round(e_high-e_low,4), round(e_high-e_av,4), round(e_av-e_low,4))
            # print("\n")


            self._logger.info(f"Creating signal exposure map at {e_av} keV & Calculating flux between {e_low}-{e_high} keV")
            soxs.make_exposure_map(evt_file, f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_region_expmap.fits", e_av, overwrite=True)  
            self._logger.info(f"Creating background exposure map at {e_av} keV & Calculating flux between {e_low}-{e_high} keV")
            soxs.make_exposure_map(bkgrnd_evt_file, f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_expmap.fits", e_av, overwrite=True)  
            self._logger.info(f"Using Background file {bkgrnd_evt_file}")
            

            soxs.write_radial_profile(evt_file, f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_region_profile.fits", [30.0, 45.0],
                          0, arcsec_Xray_HSE_R500.to("arcsec").value, 1, emin=e_low, emax=e_high, expmap_file=f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_region_expmap.fits", overwrite=True)
            with pyfits.open(f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_region_profile.fits") as f:
                flux_Xray_HSE_R500 += np.sum(e_av*f["profile"].data["net_flux"] * flux_unit)
                
            soxs.write_radial_profile(bkgrnd_evt_file, f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_profile.fits", [30.0, 45.0],
                          0, arcsec_Xray_HSE_R500.to("arcsec").value, 1, emin=e_low, emax=e_high, expmap_file=f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_expmap.fits", overwrite=True)
            with pyfits.open(f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_profile.fits") as f:
                bkgrnd_flux_Xray_HSE_R500 += np.sum(e_av*f["profile"].data["net_flux"] * flux_unit)
                
 
            soxs.write_radial_profile(evt_file, f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_region_profile.fits", [30.0, 45.0],
                          0, arcsec_Xray_HSE_R500_upper.to("arcsec").value, 1, emin=e_low, emax=e_high, expmap_file=f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_region_expmap.fits", overwrite=True)
            with pyfits.open(f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_region_profile.fits") as f:
                flux_Xray_HSE_R500_upper += np.sum(e_av*f["profile"].data["net_flux"] * flux_unit)
                
            soxs.write_radial_profile(bkgrnd_evt_file, f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_profile.fits", [30.0, 45.0],
                          0, arcsec_Xray_HSE_R500_upper.to("arcsec").value, 1, emin=e_low, emax=e_high, expmap_file=f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_expmap.fits", overwrite=True)
            with pyfits.open(f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_profile.fits") as f:
                bkgrnd_flux_Xray_HSE_R500_upper += np.sum(e_av*f["profile"].data["net_flux"] * flux_unit)      


            soxs.write_radial_profile(evt_file, f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_region_profile.fits", [30.0, 45.0],
                          0, arcsec_Xray_HSE_R500_lower.to("arcsec").value, 1, emin=e_low, emax=e_high, expmap_file=f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_region_expmap.fits", overwrite=True)
            with pyfits.open(f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_region_profile.fits") as f:
                flux_Xray_HSE_R500_lower += np.sum(e_av*f["profile"].data["net_flux"] * flux_unit)
                
            soxs.write_radial_profile(bkgrnd_evt_file, f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_profile.fits", [30.0, 45.0],
                          0, arcsec_Xray_HSE_R500_lower.to("arcsec").value, 1, emin=e_low, emax=e_high, expmap_file=f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_expmap.fits", overwrite=True)
            with pyfits.open(f"{Lx_path}/{self.idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_profile.fits") as f:
                bkgrnd_flux_Xray_HSE_R500_lower += np.sum(e_av*f["profile"].data["net_flux"] * flux_unit)     
                
                
            
        print(f"Background flux = {round(100*bkgrnd_flux_Xray_HSE_R500.value/flux_Xray_HSE_R500.value,3)}% for Xray R500")    
        flux_Xray_HSE_R500 -= bkgrnd_flux_Xray_HSE_R500
        flux_Xray_HSE_R500_lower -= bkgrnd_flux_Xray_HSE_R500_lower
        flux_Xray_HSE_R500_upper -= bkgrnd_flux_Xray_HSE_R500_upper
            


        d_l = self.cosmo.luminosity_distance(self.redshift)
        self.Xray_L_in_Xray_HSE_R500 = (flux_Xray_HSE_R500 * 4 * math.pi * d_l**2).to("erg/s")
        Xray_L_in_Xray_HSE_R500_upper = (flux_Xray_HSE_R500_upper * 4 * math.pi * d_l**2).to("erg/s")
        Xray_L_in_Xray_HSE_R500_lower = (flux_Xray_HSE_R500_lower * 4 * math.pi * d_l**2).to("erg/s")
        self.Xray_L_in_Xray_HSE_R500_spread = (abs(np.array([Xray_L_in_Xray_HSE_R500_lower.value,Xray_L_in_Xray_HSE_R500_upper.value])*u.erg/u.s - self.Xray_L_in_Xray_HSE_R500)).to("erg/s")


            

    def profile_plotting(self, save_dir = None, chip_rad_arcmin = None, plot_MW = False, plot_EW = False, plot_LW = False, plot_Xray_ODR = False, savetag = None):
        import matplotlib.gridspec as gridspec
        
        if savetag == None:
            savetag = ""
            if plot_MW: savetag += "_MW"
            if plot_EW: savetag += "_EW"
            if plot_LW: savetag += "_LW"
            if plot_Xray_ODR: savetag += "_Xray_ODR"

        if save_dir == None:
            self.profiles_save_path = Path(self._top_save_path/self.instrument_name/"PROFILES")
        else:
            self.profiles_save_path = Path(save_dir)            
        os.makedirs(self.profiles_save_path, exist_ok = True)        
        
        kT_Xray_ml_yvals = self.kT_model(self.r_values_fine.value, **self.Xray_kT_model_best_fit_pars) * u.keV
        kT_Xray_ml_yvals_at_data_r = self.kT_model(self.r.value, **self.Xray_kT_model_best_fit_pars) * u.keV
        ne_Xray_ml_yvals = self.ne_model(self.r_values_fine.value, **self.Xray_ne_model_best_fit_pars) * u.cm**-3
        ne_Xray_ml_yvals_at_data_r = self.ne_model(self.r.value, **self.Xray_ne_model_best_fit_pars) * u.cm**-3
        try:
            kT_EW_ml_yvals = self.kT_model(self.r_values_fine.value, **self.EW_kT_model_best_fit_pars) * u.keV
            ne_EW_ml_yvals = self.ne_model(self.r_values_fine.value, **self.EW_ne_model_best_fit_pars) * u.cm**-3
        except Exception as e:
            # print(e)
            pass
        try:    
            kT_MW_ml_yvals = self.kT_model(self.r_values_fine.value, **self.MW_kT_model_best_fit_pars) * u.keV
            ne_MW_ml_yvals = self.ne_model(self.r_values_fine.value, **self.MW_ne_model_best_fit_pars) * u.cm**-3
        except Exception as e:
            # print(e)
            pass
        try:
            kT_LW_ml_yvals = self.kT_model(self.r_values_fine.value, **self.LW_kT_model_best_fit_pars) * u.keV
            ne_LW_ml_yvals = self.ne_model(self.r_values_fine.value, **self.LW_ne_model_best_fit_pars) * u.cm**-3
        except Exception as e:
            # print(e)
            pass
        try:
            kT_Xray_ODR_ml_yvals = self.kT_model(self.r_values_fine.value, **self.Xray_ODR_kT_model_best_fit_pars) * u.keV
            ne_Xray_ODR_ml_yvals = self.ne_model(self.r_values_fine.value, **self.Xray_ODR_ne_model_best_fit_pars) * u.cm**-3
        except Exception as e:
            # print(e)
            pass
        data_range_idxs = [(self.r_values_fine > 1*min(self.r)) & (self.r_values_fine < 1*max(self.r)) ][0]
        
        
        
        fig = plt.figure(figsize = (10,10), facecolor = 'w')
        
        frame1 = fig.add_axes((.1,1.1,.8,.6))
        frame1.plot(self.r_values_fine[data_range_idxs]/self.R500_truth.value, kT_Xray_ml_yvals[data_range_idxs], label = "X-ray", color = "tomato")
        frame1.plot(self.r_values_fine/self.R500_truth.value, kT_Xray_ml_yvals,  color = "peru", ls = "dashed")
        try:
            # print("kT_Xray_ml_yvals", kT_Xray_ml_yvals)
            # print("self.Xray_kT_spread[0]", self.Xray_kT_spread[0])
            frame1.fill_between( (self.r_values_fine[data_range_idxs]/self.R500_truth).value, (kT_Xray_ml_yvals - self.Xray_kT_spread[0])[data_range_idxs].value, (kT_Xray_ml_yvals + self.Xray_kT_spread[1])[data_range_idxs].value, alpha = 0.3, color = "tomato")
        except Exception as e:
            print(e)
            pass
        frame1.errorbar( (self.r/self.R500_truth).value, self.kT_y.value, yerr = (abs(self.kT_yerr[:,0].value),abs(self.kT_yerr[:,1].value)) , xerr = (abs(self.r_err[:,0].value/self.R500_truth.value),abs(self.r_err[:,1].value/self.R500_truth.value)),capsize = 5, fmt = 'None', color = 'tomato')
        # frame1.errorbar( (self.r/self.R500_truth).value, self.kT_y.value, yerr = (abs(self.kT_yerr[:,0].value),abs(self.kT_yerr[:,1].value)) , xerr = (abs(self.r_err[:,0].value/self.R500_truth.value),abs(self.r_err[:,1].value/self.R500_truth.value)), capsize = 5, ecolor = "tomato") ### Re-plot in case using old version of mpl in which case fmt = "None" is bugged
        frame1.scatter( (self.r[self.Xray_non_outlier_idxs]/self.R500_truth).value, self.kT_y[self.Xray_non_outlier_idxs].value,  color = 'darkred')
        
        try : 
            if plot_EW:
                frame1.plot(self.r_values_fine/self.R500_truth, kT_EW_ml_yvals, label = "EW fit", color = "steelblue")
                frame1.scatter(self.EW_kT_x.value[::self.yT_slice]/self.R500_truth.value, self.EW_kT_y.value[::self.yT_slice], color = "steelblue", marker = "D", facecolors='none')
                frame1.scatter(self.EW_kT_x.value[::self.yT_slice][self.EW_non_outlier_idxs]/self.R500_truth.value, self.EW_kT_y.value[::self.yT_slice][self.EW_non_outlier_idxs], color = "steelblue", marker = "D", )
        except Exception as e:
            print(e)
        try : 
            if plot_MW:
                frame1.plot(self.r_values_fine/self.R500_truth, kT_MW_ml_yvals, label = "MW fit", color = "olive")
                frame1.scatter(self.MW_kT_x.value[::self.yT_slice]/self.R500_truth.value, self.MW_kT_y.value[::self.yT_slice],  color = "olive", marker = "D", facecolors='none')
                frame1.scatter(self.MW_kT_x.value[::self.yT_slice][self.MW_non_outlier_idxs]/self.R500_truth.value, self.MW_kT_y.value[::self.yT_slice][self.MW_non_outlier_idxs],  color = "olive", marker = "D", )
        except Exception as e:
            print(e)
        try : 
            if plot_LW:
                frame1.plot(self.r_values_fine/self.R500_truth, kT_LW_ml_yvals, label = "LW fit", color = "lime")
                frame1.scatter(self.LW_kT_x.value[::self.yT_slice]/self.R500_truth.value, self.LW_kT_y.value[::self.yT_slice],  color = "olive", marker = "D", facecolors='none')
                frame1.scatter(self.LW_kT_x.value[::self.yT_slice][self.LW_non_outlier_idxs]/self.R500_truth.value, self.LW_kT_y.value[::self.yT_slice][self.LW_non_outlier_idxs],  color = "lime", marker = "D", )
        except Exception as e:
            print(e)
        try : 
            if plot_Xray_ODR:
                frame1.plot(self.r_values_fine/self.R500_truth, kT_Xray_ODR_ml_yvals, label = "Xray ODR fit", color = "orchid")
        except Exception as e:
            print(e)
            
            
            
            
        frame2 = fig.add_axes((.1,0.9,.8,.2))  
        frame2.hlines(0, xmin = 0.01, xmax = 5, color = "black", alpha = 0.8)
        frame2.errorbar( (self.r/self.R500_truth).value, np.log10(self.kT_y.value) - np.log10(kT_Xray_ml_yvals_at_data_r.value), yerr = (abs(self.kT_yerr[:,0].value),abs(self.kT_yerr[:,1].value))/(np.log(10) * self.kT_y.value) , xerr = (abs(self.r_err[:,0].value/self.R500_truth.value),abs(self.r_err[:,1].value/self.R500_truth.value)),capsize = 5, fmt = 'None', color = 'tomato')
        # frame2.errorbar( (self.r/self.R500_truth).value, np.log10(self.kT_y.value) - np.log10(kT_Xray_ml_yvals_at_data_r.value), yerr = (abs(self.kT_yerr[:,0].value),abs(self.kT_yerr[:,1].value))/(np.log(10) * self.kT_y.value) , xerr = (abs(self.r_err[:,0].value/self.R500_truth.value),abs(self.r_err[:,1].value/self.R500_truth.value)),capsize = 5, capsize = 5, ecolor = "tomato")
        
        
        frame2.scatter( (self.r[self.Xray_non_outlier_idxs]/self.R500_truth).value, (np.log10(self.kT_y.value)-np.log10(kT_Xray_ml_yvals_at_data_r.value))[self.Xray_non_outlier_idxs],  color = 'darkred')
        
        
        frame3 = fig.add_axes((.9,1.1,.8,.6)) 
        frame3.plot(self.r_values_fine[data_range_idxs]/self.R500_truth, ne_Xray_ml_yvals[data_range_idxs], label = "X-ray", color = "tomato")
        frame3.plot(self.r_values_fine/self.R500_truth, ne_Xray_ml_yvals,  color = "tomato", ls = "dashed")
        try:
            frame3.fill_between( (self.r_values_fine[data_range_idxs]/self.R500_truth).value, (ne_Xray_ml_yvals - self.Xray_ne_spread[0])[data_range_idxs].value, (ne_Xray_ml_yvals + self.Xray_ne_spread[1])[data_range_idxs].value, alpha = 0.3, color = "tomato")
            # axes[0,1].fill_between(self.r_values_fine[data_range_idxs]/self.R500_truth, (ne_EW_ml_yvals - self.EW_ne_spread)[data_range_idxs], (ne_EW_ml_yvals + self.EW_ne_spread)[data_range_idxs], alpha = 0.3, color = "tomato")
        except Exception as e:
            print(e)
            pass
        
        frame3.errorbar(self.r.value/self.R500_truth.value, self.ne_y.value, yerr = (self.ne_yerr[:,0].value, self.ne_yerr[:,1].value),   xerr = (self.r_err[:,0].value/self.R500_truth.value,self.r_err[:,1].value/self.R500_truth.value),capsize = 5,  fmt = 'None', color = 'tomato')
        # frame3.errorbar(self.r.value/self.R500_truth.value, self.ne_y.value, yerr = (self.ne_yerr[:,0].value, self.ne_yerr[:,1].value),   xerr = (self.r_err[:,0].value/self.R500_truth.value,self.r_err[:,1].value/self.R500_truth.value),capsize = 5, capsize = 5, ecolor = "tomato")
        frame3.scatter( (self.r[self.Xray_non_outlier_idxs]/self.R500_truth).value, self.ne_y[self.Xray_non_outlier_idxs].value, color = 'darkred')
        try : 
            if plot_EW:
                frame3.plot(self.r_values_fine/self.R500_truth, ne_EW_ml_yvals, label = "EW Fit", color = "steelblue")
                frame3.scatter(self.EW_ne_x.value[::self.yT_slice]/self.R500_truth.value, self.EW_ne_y.value[::self.yT_slice],color = "steelblue", marker = "D", facecolors='none')
                frame3.scatter(self.EW_ne_x.value[::self.yT_slice][self.EW_non_outlier_idxs]/self.R500_truth.value, self.EW_ne_y.value[::self.yT_slice][self.EW_non_outlier_idxs],  color = "steelblue", marker = "D", )
        except Exception as e:
            print(e)
        try : 
            if plot_MW:
                frame3.plot(self.r_values_fine/self.R500_truth, ne_MW_ml_yvals, label = "MW Fit", color = "olive")
                frame3.scatter(self.MW_ne_x.value[::self.yT_slice]/self.R500_truth.value, self.MW_ne_y.value[::self.yT_slice],  color = "olive", marker = "D", facecolors='none')
                frame3.scatter(self.MW_ne_x.value[::self.yT_slice][self.MW_non_outlier_idxs]/self.R500_truth.value, self.MW_ne_y.value[::self.yT_slice][self.MW_non_outlier_idxs],  color = "olive", marker = "D", )
        except Exception as e:
            print(e)
        try : 
            if plot_LW:
                frame3.plot(self.r_values_fine/self.R500_truth, ne_LW_ml_yvals, label = "LW Fit", color = "lime")
                frame3.scatter(self.LW_ne_x.value[::self.yT_slice]/self.R500_truth.value, self.LW_ne_y.value[::self.yT_slice],  color = "lime", marker = "D", facecolors='none')
                frame3.scatter(self.LW_ne_x.value[::self.yT_slice][self.LW_non_outlier_idxs]/self.R500_truth.value, self.LW_ne_y.value[::self.yT_slice][self.LW_non_outlier_idxs],  color = "lime", marker = "D", )
        except Exception as e:
            print(e)
        try : 
            if plot_Xray_ODR:
                frame3.plot(self.r_values_fine/self.R500_truth, ne_Xray_ODR_ml_yvals, label = "Xray ODR fit", color = "orchid")
        except Exception as e:
            print(e)
            
        
        frame4 = fig.add_axes((.9,0.9,.8,.2))            
        frame4.hlines(0, xmin = 0.01, xmax = 5, color = "black", alpha = 0.8)
        frame4.errorbar( (self.r/self.R500_truth).value, np.log10(self.ne_y.value) - np.log10(ne_Xray_ml_yvals_at_data_r.value), yerr = (self.ne_yerr[:,0].value,self.ne_yerr[:,1].value)/(np.log(10) * self.ne_y.value) , xerr = (self.r_err[:,0].value/self.R500_truth.value,self.r_err[:,1].value/self.R500_truth.value),capsize = 5, fmt = 'None', color = 'tomato')
        # frame4.errorbar( (self.r/self.R500_truth).value, np.log10(self.ne_y.value) - np.log10(ne_Xray_ml_yvals_at_data_r.value), yerr = (self.ne_yerr[:,0].value,self.ne_yerr[:,1].value)/(np.log(10) * self.ne_y.value) , xerr = (self.r_err[:,0].value/self.R500_truth.value,self.r_err[:,1].value/self.R500_truth.value),capsize = 5, capsize = 5, ecolor = "tomato")
        
        frame4.scatter( (self.r[self.Xray_non_outlier_idxs]/self.R500_truth).value, (np.log10(self.ne_y.value)-np.log10(ne_Xray_ml_yvals_at_data_r.value))[self.Xray_non_outlier_idxs],  color = 'darkred')
            
        frame5 = fig.add_axes((.1,.3,.8,.6))
        frame5.plot(self.r_values_fine[data_range_idxs]/self.R500_truth,self.Xray_entropy_profile[data_range_idxs], label = "X-ray", color = "tomato") 
        frame5.plot(self.r_values_fine/self.R500_truth,self.Xray_entropy_profile,color = "tomato", ls = "dashed") 
        # frame.plot(self.r_values_fine[data_range_idxs]/self.R500_truth,self.EW_entropy_profile[data_range_idxs], label = "EW", color = "steelblue") 
        try:
            frame5.fill_between( (self.r_values_fine[data_range_idxs]/self.R500_truth).value, np.array(self.Xray_entropy_profile - self.Xray_entropy_spread )[data_range_idxs], np.array(self.Xray_entropy_profile + self.Xray_entropy_spread )[data_range_idxs], alpha = 0.3, color = "wheat")
        except:
            pass
        
        
        
        frame6 = fig.add_axes((.9,.3,.8,.6))
        frame6.plot(self.r_values_fine.value[data_range_idxs]/self.R500_truth.value,self.Xray_HSE_total_mass_profile[data_range_idxs]/self.M500_truth, label = "X-ray HSE", color = "tomato")
        frame6.plot(self.r_values_fine.value/self.R500_truth.value,self.Xray_HSE_total_mass_profile/self.M500_truth,  color = "tomato", ls = "dashed")
        try:
            frame6.fill_between( (self.r_values_fine[data_range_idxs]/self.R500_truth).value, (np.array(self.Xray_HSE_total_mass_profile - self.Xray_HSE_mass_spread[0])[data_range_idxs]/self.M500_truth).value, (np.array(self.Xray_HSE_total_mass_profile + self.Xray_HSE_mass_spread[1])[data_range_idxs]/self.M500_truth).value, alpha = 0.3, color = "tomato")
        except:
            pass
        try : 
            frame6.plot(self.yT_profile_mass_x/self.R500_truth.value, self.yT_profile_mass_y/self.M500_truth, label = "True Mass", color = "black")
            if plot_EW: frame6.plot(self.r_values_fine.value/self.R500_truth.value,self.EW_HSE_total_mass_profile/self.M500_truth, label = "EW HSE", color = "steelblue")
            if plot_MW: frame6.plot(self.r_values_fine.value/self.R500_truth.value,self.MW_HSE_total_mass_profile/self.M500_truth, label = "MW HSE", color = "olive")
            if plot_LW: frame6.plot(self.r_values_fine.value/self.R500_truth.value,self.LW_HSE_total_mass_profile/self.M500_truth, label = "LW HSE", color = "lime")
            if plot_Xray_ODR: frame6.plot(self.r_values_fine.value/self.R500_truth.value,self.Xray_ODR_HSE_total_mass_profile/self.M500_truth, label = "Xray ODR HSE", color = "orchid")
        except Exception as e:
            print(e)

        

        if chip_rad_arcmin != None:
            chip_width_kpc = (chip_rad_arcmin*(u.arcmin/u.radian)*self.cosmo.angular_diameter_distance(z = self.redshift)).to("kpc").value
            frame1.vlines(x =chip_width_kpc/self.R500_truth.value,  ymin = 1e-3, ymax =10, ls = "dotted", color = "grey" )
            frame3.vlines(x =chip_width_kpc/self.R500_truth.value,  ymin = 1e-5, ymax = 1, ls = "dotted", color = "grey" )
            # axes[1,0].vlines(x =chip_width_kpc/self.R500_truth.value,  ymin = 0.5*min(self.Xray_HSE_total_mass_profile.value[data_range_idxs]), ymax = 2*max(max(self.Xray_HSE_total_mass_profile.value[data_range_idxs]),self.M500_truth.value), ls = "dotted", color = "green" )
            frame6.vlines(x =chip_width_kpc/self.R500_truth.value,  ymin = 0.5*min(self.Xray_HSE_total_mass_profile.value[data_range_idxs]), ymax = 2*max(max(self.Xray_HSE_total_mass_profile.value[data_range_idxs]),self.M500_truth.value), ls = "dotted", color = "grey" )
            
            frame1.set_xlim(left = 0.08, right = 1.2*max(1, chip_width_kpc/self.R500_truth.value))
            frame2.set_xlim(left = 0.08, right = 1.2*max(1, chip_width_kpc/self.R500_truth.value))
            frame3.set_xlim(left = 0.08, right = 1.2*max(1, chip_width_kpc/self.R500_truth.value))
            frame4.set_xlim(left = 0.08, right = 1.2*max(1, chip_width_kpc/self.R500_truth.value))
            frame5.set_xlim(left = 0.08, right = 1.2*max(1, chip_width_kpc/self.R500_truth.value))
            frame6.set_xlim(left = 0.08, right = 1.2*max(1, chip_width_kpc/self.R500_truth.value))
            
            frame1.axvspan(chip_width_kpc/self.R500_truth.value, 1.2*max(1, chip_width_kpc/self.R500_truth.value), alpha=0.5, color='grey')
            frame3.axvspan(chip_width_kpc/self.R500_truth.value, 1.2*max(1, chip_width_kpc/self.R500_truth.value), alpha=0.5, color='grey')
            frame5.axvspan(chip_width_kpc/self.R500_truth.value, 1.2*max(1, chip_width_kpc/self.R500_truth.value), alpha=0.5, color='grey')
            frame6.axvspan(chip_width_kpc/self.R500_truth.value, 1.2*max(1, chip_width_kpc/self.R500_truth.value), alpha=0.5, color='grey')

        frame1.set_yscale("log")
        frame1.set_xscale("log")
        frame1.set_xlim(left =0.08, right = 1.2)
        frame2.set_xlim(left =0.08, right = 1.2)
        frame2.set_ylim(-0.5,0.5)
        frame2.set_yscale("linear")
        frame2.set_xscale("log")
        frame1.set_ylim(bottom = 0.3*min(kT_Xray_ml_yvals.value), top = 3*max(kT_Xray_ml_yvals.value))
        frame1.set_ylabel(r"$\mathtt{kT \ [keV]}$")
        frame1.legend(fontsize = 18)
        frame1.xaxis.set_ticks_position('top')
        frame1.set_xlabel(r"")
        frame1.set_xticklabels([])

        
        frame3.set_yscale("log")
        frame3.set_xscale("log")
        frame3.set_xlim(left = 0.08, right = 1.2)
        frame4.set_xlim(left = 0.08, right = 1.2)
        frame3.set_ylim(bottom = 0.3*min(ne_Xray_ml_yvals.value), top = 3*max(ne_Xray_ml_yvals.value))
        frame3.set_ylabel(r"$\mathtt{n}_\mathtt{e} \mathtt{[cm}^{-3}\mathtt{]}$")
        frame3.legend(fontsize = 18)
        frame3.yaxis.set_label_position("right")
        frame3.yaxis.set_ticks_position('right')
        frame4.yaxis.set_ticks_position('right')
        frame3.xaxis.set_ticks_position('top')
        frame3.set_xlabel(r"")
        frame3.set_xticklabels([])

        frame4.set_xlim(left =0.08, right = 1.2)
        frame4.set_ylim(-0.5,0.5)
        frame4.set_yscale("linear")
        frame4.set_xscale("log")

        
        frame5.set_xlabel(r"$\mathtt{R/R}_\mathtt{500,SO}$")
        frame5.set_ylabel(r"$\mathtt{K} \mathtt{[keV cm}^\mathtt{2} \mathtt{]}$")
        frame5.set_xscale("log")
        frame5.set_yscale("log")
        frame5.set_xlim(left = 0.08, right = 1.2)
        # axes[1,0].set_ylim(top = max(2*((self.Xray_entropy_profile - self.Xray_entropy_spread)[data_range_idxs]).value))
        frame5.legend(fontsize = 18)

        
        frame6.set_xscale("log")
        frame6.set_yscale("log")
        frame6.set_xlim(left = 0.08, right = 1.2)
        frame6.set_ylim(1e-2,3)
        frame6.set_xlabel(r"$\mathtt{R/R}_\mathtt{500,SO}$")
        frame6.set_ylabel(r"$\mathtt{M(<R)}/\mathtt{M}_\mathtt{500,SO}$")
        frame6.hlines(y = self.M500_truth.value, xmin = 0.0001, xmax = 1.5, ls = "dashed", color = "dimgrey" )
        frame6.vlines(x =1,  ymin = 1e-2, ymax =30, ls = "dashed", color = "dimgrey" )
        frame6.legend(fontsize = 18)
        frame6.yaxis.set_label_position("right")
        frame6.yaxis.set_ticks_position('right')
   
        
        frame1.set_xticklabels([])
        frame3.set_xticklabels([])
        
        ml = MultipleLocator(0.1)
        frame2.yaxis.set_minor_locator(ml)
        frame4.yaxis.set_minor_locator(ml)
        
        frame5.set_xticks([0.1, 0.5, 1])
        frame5.set_xticklabels([r"0.1", r"0.5", r"1"])
        frame6.set_xticks([0.1, 0.5, 1])
        frame6.set_xticklabels([r"0.1", r"0.5", r"1"])   
        
#         frame1.set_yticks([0.1,1,10]) ### If don';t do these manually the font will be wrong
#         frame1.set_yticklabels(["0.1","1","10"])
#         frame3.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1]) ### If don';t do these manually the font will be wrong
#         frame3.set_yticklabels([r"$10^{-5}$",r"$10^{-4}$",r"$10^{-3}$",r"$10^{-2}$",r"$10^{-1}$","1"])
        
#         frame2.set_yticks([-0.5,0,0.5]) ### If don';t do these manually the font will be wrong
#         frame2.set_yticklabels(["-0.5","0","0.5"])
#         frame4.set_yticks([-0.5,0,0.5]) ### If don';t do these manually the font will be wrong
#         frame4.set_yticklabels(["-0.5","0","0.5"])
        
#         frame6.set_yticks([1e-2,1e-1,1,10]) ### If don';t do these manually the font will be wrong
#         frame6.set_yticklabels([r"$10^{-2}$",r"$10^{-1}$","1", "10"])
        # frame3.set_yticklabels(["0.1","1","10"])
        
        fig.patch.set_facecolor('white')
        self._logger.info(f"Saving profile at {self.profiles_save_path}/{self.idx_instr_tag}_profiles{savetag}_4x4.png")
        plt.savefig(f"{self.profiles_save_path}/{self.idx_instr_tag}_profiles{savetag}_4x4.png",  bbox_inches='tight')
        plt.clf()
        plt.close() 
        
      
        
        

        
        
        
        
        
        
class HaloSample():
    def __init__(self, save_dir = "./Halo_Sample/", h = None):
        
        # self.profiles_save_path = Path(self._top_save_path/self.instrument_name/"PROFILES"/self.idx_instr_tag)
        # os.makedirs(self.profiles_save_path, exist_ok = True)
        
        self.sample_arr = []
        self.save_dir = save_dir
        self.r_values_fine = []
        os.makedirs(save_dir, exist_ok = True)
        if h == None:
            print("h not set. Will use h=0.68")
            self.hubble = 0.68
        else:
            self.hubble = h
        self.cosmo = FlatLambdaCDM(H0 = 100*self.hubble, Om0 = 0.3, Ob0 = 0.048)    
        
    def add_halo(self, halo_analysis, replace_existing = False, savetag = ""):    
        
        halo_dict = {}
        
        halo_dict["idx"] = halo_analysis.halo_idx
        halo_dict["instrument_name"] = halo_analysis.instrument_name 
        halo_dict["idx_tag"] = halo_analysis.idx_tag 
        halo_dict["idx_instr_tag"] = halo_analysis.idx_instr_tag 
        halo_dict["annuli_path"] = halo_analysis.annuli_path 
        halo_dict["evts_path"] = halo_analysis.evts_path 
        halo_dict["bkgrnd_idx_tag"] = halo_analysis._bkgrnd_idx_tag 
        halo_dict["bkgrnd_idx_instr_tag"] = halo_analysis._bkgrnd_idx_instr_tag
        halo_dict["bkgrnd_evts_path"] = halo_analysis._bkgrnd_evts_path    
        halo_dict["snap_num"] = halo_analysis.snap_num
        halo_dict["rho_crit"] = halo_analysis.rho_crit
        halo_dict["redshift"] = halo_analysis.redshift
        halo_dict["r_values_fine"] = halo_analysis.r_values_fine

        try:
            halo_dict["marker color"] = halo_analysis.marker_colour   
        except:
            pass
        try:
            halo_dict["label"] = halo_analysis.label 
        except:
            pass
        
        
        try:
            halo_dict["emin_for_yT_EW_values"] = halo_analysis.emin_for_EW_values
            halo_dict["emax_for_yT_EW_values"] = halo_analysis.emax_for_EW_values
        except Exception as e:
            print(e)
            pass           
        
        try:
            halo_dict["kT_MW_ml_yvals"] = halo_analysis.kT_model(halo_analysis.r_values_fine.value, **halo_analysis.MW_kT_model_best_fit_pars) * u.keV
            halo_dict["ne_MW_ml_yvals"] = halo_analysis.ne_model(halo_analysis.r_values_fine.value, **halo_analysis.MW_ne_model_best_fit_pars) * u.cm**-3
            halo_dict["MW_kT_model_best_fit_pars"] = halo_analysis.MW_kT_model_best_fit_pars
            halo_dict["MW_ne_model_best_fit_pars"] = halo_analysis.MW_ne_model_best_fit_pars
            halo_dict["MW_entropy_profile"] = halo_analysis.MW_entropy_profile
            halo_dict["MW_pressure_profile"] = halo_analysis.MW_pressure_profile
            halo_dict["MW_HSE_total_mass_profile"] = halo_analysis.MW_HSE_total_mass_profile
        except Exception as e:
            print(e)
            pass         
        try:
            halo_dict["kT_Xray_ml_yvals"] = halo_analysis.kT_model(halo_analysis.r_values_fine.value, **halo_analysis.Xray_kT_model_best_fit_pars) * u.keV
            halo_dict["ne_Xray_ml_yvals"] = halo_analysis.ne_model(halo_analysis.r_values_fine.value, **halo_analysis.Xray_ne_model_best_fit_pars) * u.cm**-3
            halo_dict["Xray_kT_model_best_fit_pars"] = halo_analysis.Xray_kT_model_best_fit_pars
            halo_dict["Xray_ne_model_best_fit_pars"] = halo_analysis.Xray_ne_model_best_fit_pars
            halo_dict["Xray_entropy_profile"] = halo_analysis.Xray_entropy_profile
            halo_dict["Xray_pressure_profile"] = halo_analysis.Xray_pressure_profile
            halo_dict["Xray_HSE_total_mass_profile"] = halo_analysis.Xray_HSE_total_mass_profile
            halo_dict["radii_used_to_fit_Xray"] = halo_analysis.radii_used_to_fit_Xray
        except:
            print("Not all X-ray fit data was saved for this halo")
        try:
            halo_dict["kT_EW_ml_yvals"] = halo_analysis.kT_model(halo_analysis.r_values_fine.value, **halo_analysis.EW_kT_model_best_fit_pars) * u.keV
            halo_dict["ne_EW_ml_yvals"] = halo_analysis.ne_model(halo_analysis.r_values_fine.value, **halo_analysis.EW_ne_model_best_fit_pars) * u.cm**-3
            halo_dict["EW_kT_model_best_fit_pars"] = halo_analysis.EW_kT_model_best_fit_pars
            halo_dict["EW_ne_model_best_fit_pars"] = halo_analysis.EW_ne_model_best_fit_pars
            halo_dict["EW_entropy_profile"] = halo_analysis.EW_entropy_profile
            halo_dict["EW_pressure_profile"] = halo_analysis.EW_pressure_profile
            halo_dict["EW_HSE_total_mass_profile"] = halo_analysis.EW_HSE_total_mass_profile
        except Exception as e:
            print(e)
            pass
        try:
            halo_dict["kT_LW_ml_yvals"] = halo_analysis.kT_model(halo_analysis.r_values_fine.value, **halo_analysis.LW_kT_model_best_fit_pars) * u.keV
            halo_dict["ne_LW_ml_yvals"] = halo_analysis.ne_model(halo_analysis.r_values_fine.value, **halo_analysis.LW_ne_model_best_fit_pars) * u.cm**-3
            halo_dict["LW_kT_model_best_fit_pars"] = halo_analysis.LW_kT_model_best_fit_pars
            halo_dict["LW_ne_model_best_fit_pars"] = halo_analysis.LW_ne_model_best_fit_pars
            halo_dict["LW_entropy_profile"] = halo_analysis.LW_entropy_profile
            halo_dict["LW_pressure_profile"] = halo_analysis.LW_pressure_profile
            halo_dict["LW_HSE_total_mass_profile"] = halo_analysis.LW_HSE_total_mass_profile
        except Exception as e:
            print(e)
            pass
        
        try:
            print("Adding ODR values to sample")
            halo_dict["Xray_ODR_HSE_M500"] = halo_analysis.Xray_ODR_HSE_M500
            halo_dict["Xray_ODR_kT_model_best_fit_pars"] = halo_analysis.Xray_ODR_kT_model_best_fit_pars
            halo_dict["Xray_ODR_ne_model_best_fit_pars"] = halo_analysis.Xray_ODR_ne_model_best_fit_pars
            halo_dict["Xray_ODR_HSE_total_mass_profile"] = halo_analysis.Xray_ODR_HSE_total_mass_profile
        except Exception as e:
            print(e)
            pass

        try:
            halo_dict["M500_truth"] = halo_analysis.M500_truth
        except Exception as e:
            print(e)
            pass
        try:
            halo_dict["R500_truth"] = halo_analysis.R500_truth
        except Exception as e:
            print(e)
            pass
        try:
            halo_dict[f"Lx_in_R500_truth_{halo_analysis.emin_for_total_Lx}_{halo_analysis.emax_for_total_Lx}_keV"] = halo_analysis.Lx_in_R500_truth
        except Exception as e:
            print(e)
            pass
        

        ######################
        
        try:
            halo_dict["Xray_HSE_M500"] = halo_analysis.Xray_HSE_M500
        except Exception as e:
            print(e)
        try:
            halo_dict["Xray_HSE_M500_uncertainty"] = halo_analysis.Xray_HSE_M500_spread
        except Exception as e:
            print(e)        
        try:
            halo_dict["Xray_HSE_R500"] = halo_analysis.Xray_HSE_R500
        except Exception as e:
            print(e)
        try:
            halo_dict["Xray_HSE_R500_uncertainty"] = halo_analysis.Xray_HSE_R500_spread
        except Exception as e:
            print(e)
        try:
            halo_dict["kT_Xray_at_Xray_HSE_R500"] = halo_analysis.kT_Xray_at_Xray_HSE_R500  
        except Exception as e:
            print(e)
        try:
            halo_dict["kT_Xray_at_Xray_HSE_R500_spread"] = halo_analysis.kT_Xray_at_Xray_HSE_R500_spread 
        except Exception as e:
            print(e)
        try:
            halo_dict["S_Xray_at_Xray_HSE_R500"] = halo_analysis.S_Xray_at_Xray_HSE_R500  
        except Exception as e:
            print(e)
        try:
            halo_dict["S_Xray_at_Xray_HSE_R500_spread"] = halo_analysis.S_Xray_at_Xray_HSE_R500_spread 
        except Exception as e:
            print(e)
        try: 
            halo_dict[f"Xray_L_in_Xray_HSE_R500_{halo_analysis._Xray_total_lumin_emin_RF}-{halo_analysis._Xray_total_lumin_emax_RF}_RF_keV"] = halo_analysis.Xray_L_in_Xray_HSE_R500    
        except Exception as e:
            print(e)
        try:
            halo_dict[f"Xray_L_in_Xray_HSE_R500_spread_{halo_analysis._Xray_total_lumin_emin_RF}-{halo_analysis._Xray_total_lumin_emax_RF}_RF_keV"] = halo_analysis.Xray_L_in_Xray_HSE_R500_spread
        except Exception as e:
            print(e)
        try:
            halo_dict["Xray_entropy_spread"] = halo_analysis.Xray_entropy_spread
        except Exception as e:
            print(e)
        try:
            halo_dict["Xray_pressure_profile"] = halo_analysis.Xray_pressure_profile 
        except Exception as e:
            print(e)
        try:
            halo_dict["Xray_lumin_emin"] = halo_analysis._Xray_lumin_emin 
        except Exception as e:
            print(e)
        try:
            halo_dict["Xray_lumin_emax"] = halo_analysis._Xray_lumin_emax 
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_kT_Xray"] = halo_analysis.weighted_kT_Xray
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_kT_Xray_spread"] = halo_analysis.weighted_kT_Xray_spread
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_S_Xray"] = halo_analysis.weighted_S_Xray
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_S_Xray_spread"] = halo_analysis.weighted_S_Xray_spread
        except Exception as e:
            print(e)
        try:
            halo_dict["Xray_mgas_in_HSE_R500"] = halo_analysis.Xray_mgas_in_HSE_R500
        except Exception as e:
            print(e)
        try:
            halo_dict["Xray_Y"] = halo_analysis.Xray_Y
        except Exception as e:
            print(e)
        try:
            halo_dict["Xray_mgas_in_HSE_R500_spread"] = halo_analysis.Xray_mgas_in_HSE_R500_spread
        except Exception as e:
            print(e)
        try:
            halo_dict["Xray_Y_spread"] = halo_analysis.Xray_Y_spread
        except Exception as e:
            print(e)
            
        ######################
        
        try:
            halo_dict["MW_HSE_M500"] = halo_analysis.MW_HSE_M500
        except Exception as e:
            print(e)
        try:
            halo_dict["MW_HSE_M500_uncertainty"] = halo_analysis.MW_HSE_M500_spread
        except Exception as e:
            print(e)      
        try:
            halo_dict["MW_HSE_R500"] = halo_analysis.MW_HSE_R500
        except Exception as e:
            print(e)
        try:
            halo_dict["MW_HSE_R500_uncertainty"] = halo_analysis.MW_HSE_R500_spread
        except Exception as e:
            print(e) 
        try:
            halo_dict["kT_MW_at_MW_HSE_R500"] = halo_analysis.kT_MW_at_MW_HSE_R500  
        except Exception as e:
            print(e)
        try:
            halo_dict["kT_MW_at_MW_HSE_R500_spread"] = halo_analysis.kT_MW_at_MW_HSE_R500_spread 
        except Exception as e:
            print(e)
        try:
            halo_dict["S_MW_at_MW_HSE_R500"] = halo_analysis.S_MW_at_MW_HSE_R500  
        except Exception as e:
            print(e)
        try:
            halo_dict["S_MW_at_MW_HSE_R500_spread"] = halo_analysis.S_MW_at_MW_HSE_R500_spread 
        except Exception as e:
            print(e)
        try:
            halo_dict["MW_entropy_spread"] = halo_analysis.MW_entropy_spread
        except Exception as e:
            print(e)
        try:
            halo_dict["MW_pressure_profile"] = halo_analysis.MW_pressure_profile 
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_kT_MW"] = halo_analysis.weighted_kT_MW
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_kT_MW_spread"] = halo_analysis.weighted_kT_MW_spread
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_S_MW"] = halo_analysis.weighted_S_MW
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_S_MW_spread"] = halo_analysis.weighted_S_MW_spread
        except Exception as e:
            print(e)
        try:
            halo_dict["MW_Y"] = halo_analysis.MW_Y
        except Exception as e:
            print(e)
        try:
            halo_dict["MW_Y_spread"] = halo_analysis.MW_Y_spread
        except Exception as e:
            print(e)
        
    ######################
    
        try:
            halo_dict["EW_HSE_M500"] = halo_analysis.EW_HSE_M500
        except Exception as e:
            print(e)
        try:
            halo_dict["EW_HSE_M500_uncertainty"] = halo_analysis.EW_HSE_M500_spread
        except Exception as e:
            print(e)      
        try:
            halo_dict["EW_HSE_R500"] = halo_analysis.EW_HSE_R500
        except Exception as e:
            print(e)
        try:
            halo_dict["EW_HSE_R500_uncertainty"] = halo_analysis.EW_HSE_R500_spread
        except Exception as e:
            print(e) 
        try:
            halo_dict["kT_EW_at_EW_HSE_R500"] = halo_analysis.kT_EW_at_EW_HSE_R500  
        except Exception as e:
            print(e)
        try:
            halo_dict["kT_EW_at_EW_HSE_R500_spread"] = halo_analysis.kT_EW_at_EW_HSE_R500_spread 
        except Exception as e:
            print(e)
        try:
            halo_dict["S_EW_at_EW_HSE_R500"] = halo_analysis.S_EW_at_EW_HSE_R500  
        except Exception as e:
            print(e)
        try:
            halo_dict["S_EW_at_EW_HSE_R500_spread"] = halo_analysis.S_EW_at_EW_HSE_R500_spread 
        except Exception as e:
            print(e)
        try:
            halo_dict["EW_entropy_spread"] = halo_analysis.EW_entropy_spread
        except Exception as e:
            print(e)
        try:
            halo_dict["EW_pressure_profile"] = halo_analysis.EW_pressure_profile 
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_kT_EW"] = halo_analysis.weighted_kT_EW
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_kT_EW_spread"] = halo_analysis.weighted_kT_EW_spread
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_S_EW"] = halo_analysis.weighted_S_EW
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_S_EW_spread"] = halo_analysis.weighted_S_EW_spread
        except Exception as e:
            print(e)
        try:
            halo_dict["EW_Y"] = halo_analysis.EW_Y
        except Exception as e:
            print(e)
        try:
            halo_dict["EW_Y_spread"] = halo_analysis.EW_Y_spread
        except Exception as e:
            print(e)
                
    ######################
    
        try:
            halo_dict["LW_HSE_M500"] = halo_analysis.LW_HSE_M500
        except Exception as e:
            print(e)
        try:
            halo_dict["LW_HSE_M500_uncertainty"] = halo_analysis.LW_HSE_M500_spread
        except Exception as e:
            print(e)      
        try:
            halo_dict["LW_HSE_R500"] = halo_analysis.LW_HSE_R500
        except Exception as e:
            print(e)
        try:
            halo_dict["LW_HSE_R500_uncertainty"] = halo_analysis.LW_HSE_R500_spread
        except Exception as e:
            print(e) 
        try:
            halo_dict["kT_LW_at_LW_HSE_R500"] = halo_analysis.kT_LW_at_LW_HSE_R500  
        except Exception as e:
            print(e)
        try:
            halo_dict["kT_LW_at_LW_HSE_R500_spread"] = halo_analysis.kT_LW_at_LW_HSE_R500_spread 
        except Exception as e:
            print(e)
        try:
            halo_dict["S_LW_at_LW_HSE_R500"] = halo_analysis.S_LW_at_LW_HSE_R500  
        except Exception as e:
            print(e)
        try:
            halo_dict["S_LW_at_LW_HSE_R500_spread"] = halo_analysis.S_LW_at_LW_HSE_R500_spread 
        except Exception as e:
            print(e)
        try:
            halo_dict["LW_entropy_spread"] = halo_analysis.LW_entropy_spread
        except Exception as e:
            print(e)
        try:
            halo_dict["LW_pressure_profile"] = halo_analysis.LW_pressure_profile 
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_kT_LW"] = halo_analysis.weighted_kT_LW
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_kT_LW_spread"] = halo_analysis.weighted_kT_LW_spread
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_S_LW"] = halo_analysis.weighted_S_LW
        except Exception as e:
            print(e)
        try:
            halo_dict["weighted_S_LW_spread"] = halo_analysis.weighted_S_LW_spread
        except Exception as e:
            print(e)
        try:
            halo_dict["LW_Y"] = halo_analysis.LW_Y
        except Exception as e:
            print(e)
        try:
            halo_dict["LW_Y_spread"] = halo_analysis.LW_Y_spread
        except Exception as e:
            print(e)
                
        
        
        
        
        ######################
        
        try:
            halo_dict["gas_vel_disp"] = halo_analysis.gas_vel_disp
        except:
            pass
        try:
            halo_dict["dm_vel_disp"] = halo_analysis.dm_vel_disp
        except:
            pass
        try:
            halo_dict["total_vel_disp"] = halo_analysis.total_vel_disp
        except:
            pass
        try:
            halo_dict["gas_kappa_rot"] = halo_analysis.gas_kappa_rot
        except:
            pass
        try:
            halo_dict["dm_kappa_rot"] = halo_analysis.dm_kappa_rot
        except:
            pass
        try:
            halo_dict["total_kappa_rot"] = halo_analysis.total_kappa_rot
        except:
            pass

        try:
            halo_dict["gas_over_total_vel_disp"] = halo_analysis.gas_vel_disp/halo_analysis.total_vel_disp
        except:
            pass
        try:
            halo_dict["gas_over_dm_vel_disp"] = halo_analysis.gas_vel_disp/halo_analysis.dm_vel_disp
        except:
            pass
        try:
            halo_dict["gas_over_total_kappa_rot"] = halo_analysis.gas_kappa_rot/halo_analysis.total_kappa_rot
        except Exception as e:
            print(e)
            pass
        try:
            halo_dict["gas_over_dm_kappa_rot"] = halo_analysis.gas_kappa_rot/halo_analysis.dm_kappa_rot
        except Exception as e:
            print(e)
            pass
        
        try:
            halo_dict["gas_vel_disp_over_circ_vel"] = halo_analysis.gas_vel_disp/halo_analysis.virial_circular_vel
        except:
            pass        

        try:
            halo_dict["sfr"] = halo_analysis.sfr
        except:
            pass    
        try:
            halo_dict["sfr_100"] = halo_analysis.sfr_100
        except:
            pass  
        try:
            halo_dict["stellar_mass"] = halo_analysis.stellar_mass
        except:
            pass  
        try:
            halo_dict["ssfr"] = halo_analysis.sfr/halo_analysis.stellar_mass
        except:
            pass    
        try:
            halo_dict["ssfr_100"] = halo_analysis.sfr_100/halo_analysis.stellar_mass
        except:
            pass  
        try:
            halo_dict["bhmdot"] = halo_analysis.bhmdot
        except:
            pass    
        try:
            halo_dict["bh_fedd"] = halo_analysis.bh_fedd
        except:
            pass    
        
        
        if replace_existing == False:
            if len([i for i in range(len(self.sample_arr)) if self.sample_arr[i]["idx"]==halo_dict["idx"]]) != 0:
                print("This index already exists and 'replace_existing' is set to False, so won't append this halo to the data array. Returning...")
                return
            else:
                self.sample_arr.append(halo_dict) 
        if replace_existing == True:
            print(f"Replacing index {halo_dict['idx']} in saved data file")
            index_into_sample_arr = int(*[i for i in range(len(self.sample_arr)) if self.sample_arr[i]["idx"]==halo_dict["idx"]])
        
        np.save(f"{self.save_dir}/halo_sample_data_dict{savetag}.npy", self.sample_arr)
        
    def load_analysis(self, load_location = "./Halo_Sample//halo_sample_data_dict.npy"):
        print(f"Loading Halo Analysis from {load_location}")
        self.sample_arr = list(np.load(load_location, allow_pickle = True))
        self._load_location = load_location
        
        
    def change_color(self,color, replacement):
        for i,halo in enumerate(self.sample_arr):
            if halo["marker color"] == color:
                self.sample_arr[i]["marker color"] = replacement
        
    @staticmethod
    def set_plotstyle():
        import matplotlib.font_manager
        plt.rcParams.update({ "text.usetex": False})
        
        # print("font.monospace", plt.rcParams["font.monospace"])
        font = {'family' : 'monospace',
                'monospace':'Courier',
                'weight' : 'normal',
                'size'   : 45}
        plt.rc("font", **font)
        # plt.rc("text",usetex = True )
        plt.rc("axes", linewidth=2)
        plt.rc("xtick.major", width=3, size=20)
        plt.rc("xtick.minor", width=2, size=10)
        plt.rc("ytick.major", width=3, size=20)
        plt.rc("ytick.minor", width=2, size=10)
        
        plt.rcParams["xtick.labelsize"] = 35
        plt.rcParams["ytick.labelsize"] = 35
        plt.rcParams["legend.fontsize"] =  30
        plt.rcParams["legend.framealpha"] =  0.2
        # print("font.monospace", plt.rcParams["font.monospace"])
        # plt.rcParams["font.monospace"] = "Courier"



        
        # plt.rcParams['text.usemathtext'] = True
        # # plt.rcParams['mathtext.fontset'] = 'custom'
        # # plt.rcParams["mathtext.fontfamily"]  = "monospace"
        # plt.rcParams["mathtext.rm"]  = "monospace"
        # plt.rcParams["mathtext.it"]  = "monospace:italic"
        # plt.rcParams["mathtext.bf"]  = "monospace:bold"
        # plt.rcParams["mathtext.fontset"] = "custom"
        # plt.rcParams['font.family'] = 'Monospace' #'STIXGeneral'
    def measure_Xray_flux(self, emin, emax, halo_idx):
        '''
        To calculate flux/luminosity we create linearly-spaced energy bins between emin and emax and use SOXS to create an exposure map at the energy corresponding to the bin midpoint. We then calculate the counts flux in radial (arcsec) bins within the desired aperture (for L_x(R500), we use just a single bin from the center to R500) and multiply by the bin energy midpoint to convert to an energy flux. We repeat for all energy bins and sum the energy fluxes to give a total energy flux, which is then converted to a luminosity via the luminosity distance. Here we do this in post-processing so that we can quickly add luminosites to tweaked fits without having to do this time-consumin step whilst calculating profile fits.
        ------------------------------------------------
        Positional Arguments:

        Keyword Arguments:

        Returns: 

        '''
        import soxs   
        soxs.set_soxs_config("soxs_data_dir", "./CODE/instr_files/")
        from soxs.instrument_registry import instrument_registry
        from astropy.units import arcsec, radian


        index_into_sample_arr = int(*[i for i in range(len(sample_arr)) if sample_arr[i]["idx"]== halo_idx])
        halo_dict = self.sample_arr[index_into_sample_arr]

        halo_idx             = halo_dict["idx"]
        instrument_name      = halo_dict["instrument_name"]
        idx_tag              = halo_dict["idx_tag"]
        idx_instr_tag        = halo_dict["idx_instr_tag"]
        annuli_path          = halo_dict["annuli_path"]
        evts_path            = halo_dict["evts_path"]
        bkgrnd_idx_tag       = halo_dict["bkgrnd_idx_tag"]
        bkgrnd_idx_instr_tag = halo_dict["bkgrnd_idx_instr_tag"] 
        bkgrnd_evts_path     = halo_dict["bkgrnd_evts_path"]  
        snap_num             = halo_dict["snap_num"] 
        r_values_fine        = halo_dict["r_values_fine"]  

        instrument_spec = instrument_registry[instrument_name]
        nx = instrument_spec["num_pixels"]
        plate_scale = instrument_spec["fov"]/nx/60. # arcmin to deg
        plate_scale_arcsec = plate_scale * 3600.0   #arcsec per pixel
        if instrument_name in ["athena_wfi"]:
            chip_width_arcsec = np.array(instrument_spec["chips"])[1][[3,4]].astype(np.float)* plate_scale_arcsec
        else:
            print("Instrument currently not supported")
            sys.exit()
        chip_radius_arcsec = max(chip_width_arcsec)/2

        img_file = f"{evts_path}/{idx_instr_tag}_img.fits"  #We never make a cleaned version of this, since it is really time-expensive as opposed to doing it annulus-by-annulus
        evt_file = f"{evts_path}/{idx_instr_tag}_evt.fits"
        bkgrnd_evt_file = f"{_bkgrnd_evts_path}/{_bkgrnd_idx_instr_tag}_evt.fits"

        Lx_path = Path(f"{annuli_path}/Lx_files")
        os.makedirs(Lx_path, exist_ok = True)

        with fits.open(img_file) as hdul:
            # wcs = WCS(hdul[0].header)   
            # center = np.array([float(hdul[0].header['CRPIX1']),float(hdul[0].header['CRPIX2'])] )
            # nx = instrument_registry[instrument_name]["num_pixels"]
            # plate_scale = arcsec * instrument_registry[instrument_name]["fov"]/nx  ### arcsec per pixel  

            ang_diam_dist = cosmo.angular_diameter_distance(z = redshift)/radian

            arcsec_Xray_HSE_R500 = (Xray_HSE_R500 / ang_diam_dist).to("arcsec")
            arcsec_Xray_HSE_R500_upper = (Xray_HSE_R500_upper / ang_diam_dist).to("arcsec")
            arcsec_Xray_HSE_R500_lower = (Xray_HSE_R500_lower / ang_diam_dist).to("arcsec")


        if arcsec_Xray_HSE_R500_upper.to("arcsec").value > chip_radius_arcsec:
            _logger.warning(f"Halo: {halo_idx}:  X-ray measured R500 upper limit in arcsec = {arcsec_Xray_HSE_R500_upper} > chip radius for {instrument_name} = {chip_radius_arcsec} at current redshift = {redshift}. (Best-fit R500 in arcsec = {arcsec_Xray_HSE_R500} ). Quitting...")
            return   
        else:
            _logger.info(f"Halo: {halo_idx}:  X-ray measured R500 upper limit in arcsec = {arcsec_Xray_HSE_R500_upper} <= chip radius for {instrument_name} = {chip_radius_arcsec} at current redshift = {redshift}. (Best-fit R500 in arcsec = {arcsec_Xray_HSE_R500} ). ")


        flux_unit = u.keV / u.s / u.cm**2

        flux_Xray_HSE_R500 = 0
        flux_Xray_HSE_R500_upper = 0
        flux_Xray_HSE_R500_lower = 0

        bkgrnd_flux_Xray_HSE_R500 = 0
        bkgrnd_flux_Xray_HSE_R500_upper = 0
        bkgrnd_flux_Xray_HSE_R500_lower = 0



        step = 0.2
        _logger.info(f"Will create { -(emax + step -emin)//-step} exposure maps for linearly spaced energies between {emin} and {emax} to calculate the R500 fluxes")
        for e in range(int(1000*emin),int(1000*emax) , int(1000*step)): 
            e_low = e/1000  
            e_high = e/1000  + step
            e_av = 0.5 * (e_low + e_high)
            e_low = round(e_low, 3)
            e_high = round(e_high, 3)
            e_av = round(e_av, 3)

            _logger.info(f"Creating signal exposure map at {e_av} keV & Calculating flux between {e_low}-{e_high} keV")
            soxs.make_exposure_map(evt_file, f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_region_expmap.fits", e_av, overwrite=True)  
            _logger.info(f"Creating background exposure map at {e_av} keV & Calculating flux between {e_low}-{e_high} keV")
            soxs.make_exposure_map(bkgrnd_evt_file, f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_expmap.fits", e_av, overwrite=True)  
            _logger.info(f"Using Background file {bkgrnd_evt_file}")


            soxs.write_radial_profile(evt_file, f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_region_profile.fits", [30.0, 45.0],
                          0, arcsec_Xray_HSE_R500.to("arcsec").value, 1, emin=e_low, emax=e_high, expmap_file=f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_region_expmap.fits", overwrite=True)
            with pyfits.open(f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_region_profile.fits") as f:
                flux_Xray_HSE_R500 += np.sum(e_av*f["profile"].data["net_flux"] * flux_unit)

            soxs.write_radial_profile(bkgrnd_evt_file, f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_profile.fits", [30.0, 45.0],
                          0, arcsec_Xray_HSE_R500.to("arcsec").value, 1, emin=e_low, emax=e_high, expmap_file=f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_expmap.fits", overwrite=True)
            with pyfits.open(f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_profile.fits") as f:
                bkgrnd_flux_Xray_HSE_R500 += np.sum(e_av*f["profile"].data["net_flux"] * flux_unit)


            soxs.write_radial_profile(evt_file, f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_region_profile.fits", [30.0, 45.0],
                          0, arcsec_Xray_HSE_R500_upper.to("arcsec").value, 1, emin=e_low, emax=e_high, expmap_file=f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_region_expmap.fits", overwrite=True)
            with pyfits.open(f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_region_profile.fits") as f:
                flux_Xray_HSE_R500_upper += np.sum(e_av*f["profile"].data["net_flux"] * flux_unit)

            soxs.write_radial_profile(bkgrnd_evt_file, f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_profile.fits", [30.0, 45.0],
                          0, arcsec_Xray_HSE_R500_upper.to("arcsec").value, 1, emin=e_low, emax=e_high, expmap_file=f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_expmap.fits", overwrite=True)
            with pyfits.open(f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_profile.fits") as f:
                bkgrnd_flux_Xray_HSE_R500_upper += np.sum(e_av*f["profile"].data["net_flux"] * flux_unit)      


            soxs.write_radial_profile(evt_file, f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_region_profile.fits", [30.0, 45.0],
                          0, arcsec_Xray_HSE_R500_lower.to("arcsec").value, 1, emin=e_low, emax=e_high, expmap_file=f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_region_expmap.fits", overwrite=True)
            with pyfits.open(f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_region_profile.fits") as f:
                flux_Xray_HSE_R500_lower += np.sum(e_av*f["profile"].data["net_flux"] * flux_unit)

            soxs.write_radial_profile(bkgrnd_evt_file, f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_profile.fits", [30.0, 45.0],
                          0, arcsec_Xray_HSE_R500_lower.to("arcsec").value, 1, emin=e_low, emax=e_high, expmap_file=f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_expmap.fits", overwrite=True)
            with pyfits.open(f"{Lx_path}/{idx_instr_tag}_Xray_HSE_R500_bkgrnd_region_profile.fits") as f:
                bkgrnd_flux_Xray_HSE_R500_lower += np.sum(e_av*f["profile"].data["net_flux"] * flux_unit)     



        print(f"Background flux = {round(100*bkgrnd_flux_Xray_HSE_R500.value/flux_Xray_HSE_R500.value,3)}% for Xray R500")    
        flux_Xray_HSE_R500 -= bkgrnd_flux_Xray_HSE_R500
        flux_Xray_HSE_R500_lower -= bkgrnd_flux_Xray_HSE_R500_lower
        flux_Xray_HSE_R500_upper -= bkgrnd_flux_Xray_HSE_R500_upper



        d_l = cosmo.luminosity_distance(redshift)
        Xray_L_in_Xray_HSE_R500 = (flux_Xray_HSE_R500 * 4 * math.pi * d_l**2).to("erg/s")
        Xray_L_in_Xray_HSE_R500_upper = (flux_Xray_HSE_R500_upper * 4 * math.pi * d_l**2).to("erg/s")
        Xray_L_in_Xray_HSE_R500_lower = (flux_Xray_HSE_R500_lower * 4 * math.pi * d_l**2).to("erg/s")

        halo_dict["Xray_L_in_Xray_HSE_R500"] = Xray_L_in_Xray_HSE_R500
        halo_dict["Xray_L_in_Xray_HSE_R500_upper"] = Xray_L_in_Xray_HSE_R500_upper
        halo_dict["Xray_L_in_Xray_HSE_R500_lower"] = Xray_L_in_Xray_HSE_R500_lower

        self.sample_arr[index_into_sample_arr] = halo_dict  ### Update sample array with updated entry for this halo idx including the Lx
        np.save(self._load_location, self.sample_arr)       ### Save the sample arr in the same place it was loaded from
        self.load_analysis(load_location = self._load_location)  ### Reload the sample array 


        
        
    def plot_Xray_mass_bias(self, plot_running_median=True, median_kernel_size=3,  median_alpha = 0.5,  median_lims = (0,-1), plot_running_percentiles = (16,84), plot_legend = False):
        xmin_plt_lim = 8e11 
        xmax_plt_lim = 1e15

        fig1 = plt.figure(figsize = (10,10), facecolor = 'w')
        frame1 = fig1.add_axes((.1,.3,.8,.6))
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.ylim(bottom = xmin_plt_lim, top = xmax_plt_lim)
        plt.yscale("log")
        plt.xscale('log')
        plt.xticks([], [])
        frame2 = fig1.add_axes((.1,.1,.8,.2))
        plt.hlines(y = 1, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "green", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.9, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "orange", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.7, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "sienna", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.4, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "teal", lw = 2, ls = "solid", alpha = 0.514)
        
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.xlabel(r"$\frac{\mathtt{M}_\mathtt{500,SO}}{\mathtt{M}_\odot}$")
        plt.ylim(bottom = 0,top = 2.5)
        plt.xscale("log")
        plt.yscale("linear")
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,X}}{\mathtt{M}_\mathtt{500,SO}}$")
        plt.sca(frame1)

        N = 29
        zscore_thresh = 5000

        plotted_labels = []
        for halo in self.sample_arr:
            if "Xray_HSE_M500" not in halo.keys():
                continue
                
            plt.sca(frame1)
            plt.errorbar(halo["M500_truth"].value,halo["Xray_HSE_M500"].value, yerr = abs(np.reshape(halo["Xray_HSE_M500_uncertainty"].value, (2,1))) , capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
            if halo['label'] in plotted_labels:
                plt.scatter(halo["M500_truth"],halo["Xray_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100, linewidths = 0.6)
            else:
                plt.scatter(halo["M500_truth"],halo["Xray_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100, linewidths = 0.6, label = halo['label'])
            plt.sca(frame2)
            plt.errorbar(halo["M500_truth"].value,halo["Xray_HSE_M500"].value/halo["M500_truth"].value, yerr = abs(np.reshape(halo["Xray_HSE_M500_uncertainty"].value, (2,1))) /halo["M500_truth"].value, capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
            plt.scatter(halo["M500_truth"],halo["Xray_HSE_M500"]/halo["M500_truth"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100, linewidths = 0.6)
        
        
        plt.sca(frame1)
        plt.plot(10**np.arange(10,20,step = 0.5), 10**np.arange(10,20,step = 0.5),  color = "green", lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.9 * 10**np.arange(10,20,step = 0.5),  color = "orange",  lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.7 * 10**np.arange(10,20,step = 0.5),  color = "sienna",  lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.4 * 10**np.arange(10,20,step = 0.5),  color = "teal",  lw = 2, ls = "solid", alpha = 0.514)
        
        if plot_running_median:
            plt.sca(frame2)
            x_data = np.array([halo["M500_truth"].value for halo in self.sample_arr if "Xray_HSE_M500" in halo.keys()])
            y_data = np.array([halo["Xray_HSE_M500"].value for halo in self.sample_arr if "Xray_HSE_M500" in halo.keys()])
            sort_idxs = np.argsort(x_data)
            x_data = x_data[sort_idxs]
            y_data = y_data[sort_idxs]
            running_median = median_filter(y_data/x_data, size = median_kernel_size)
            running_perc1 = percentile_filter(y_data/x_data,percentile = min(plot_running_percentiles), size = median_kernel_size)
            running_perc2 = percentile_filter(y_data/x_data,percentile = max(plot_running_percentiles), size = median_kernel_size)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)  
            plt.sca(frame1)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median*x_data)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)    
            plt.sca(frame1)
            if plot_running_percentiles != False:
                    plt.sca(frame2)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = running_perc1[median_lims[0]:median_lims[1]],y2 = running_perc2[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)   
                    plt.sca(frame1)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = (running_perc1*x_data)[median_lims[0]:median_lims[1]],y2 = (running_perc2*x_data)[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)       
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        if not plot_legend:
            plt.gca().get_legend().remove()
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,X}}{\mathtt{M}_\odot}$")
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.sca(frame2)
        ml = MultipleLocator(0.1)
        frame2.yaxis.set_minor_locator(ml)
        ml = MultipleLocator(1)
        frame2.yaxis.set_major_locator(ml)
        plt.sca(frame1)
        ax = plt.gca()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/Mx_vs_MSphOv.png", bbox_inches='tight', overwrite = True)
        plt.clf()
        plt.close() 

        
        
    def plot_Xray_ODR_mass_bias(self, plot_running_median=True, median_kernel_size=3,  median_alpha = 0.5,  median_lims = (0,-1), plot_running_percentiles = (16,84), plot_legend = False):
        xmin_plt_lim = 8e11 
        xmax_plt_lim = 1e15 

        fig1 = plt.figure(figsize = (10,10), facecolor = 'w')
        frame1 = fig1.add_axes((.1,.3,.8,.6))
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.ylim(bottom = xmin_plt_lim, top = xmax_plt_lim)
        plt.yscale("log")
        plt.xscale('log')
        plt.xticks([], [])
        frame2 = fig1.add_axes((.1,.1,.8,.2))
        plt.hlines(y = 1, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "green", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.9, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "orange", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.7, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "sienna", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.4, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "teal", lw = 2, ls = "solid", alpha = 0.514)
        
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.xlabel(r"$\frac{\mathtt{M}_\mathtt{500,SO}}{\mathtt{M}_\odot}$")
        plt.ylim(bottom = 0,top = 2.5)
        plt.xscale("log")
        plt.yscale("linear")
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,X_ODR}}{\mathtt{M}_\mathtt{500,SO}}$")
        plt.sca(frame1)

        N = 29
        zscore_thresh = 5000

        plotted_labels = []
        for halo in self.sample_arr:
            if "Xray_ODR_HSE_M500" not in halo.keys():
                continue
                
            plt.sca(frame1)
            # plt.errorbar(halo["M500_truth"].value,halo["Xray_ODR_HSE_M500"].value, yerr = abs(np.reshape(halo["Xray_ODR_HSE_M500_uncertainty"].value, (2,1))) , capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
            if halo['label'] in plotted_labels:
                plt.scatter(halo["M500_truth"],halo["Xray_ODR_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100, linewidths = 0.6)
            else:
                plt.scatter(halo["M500_truth"],halo["Xray_ODR_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100, linewidths = 0.6, label = halo['label'])
            plt.sca(frame2)
            # plt.errorbar(halo["M500_truth"].value,halo["Xray_ODR_HSE_M500"].value/halo["M500_truth"].value, yerr = abs(np.reshape(halo["Xray_ODR_HSE_M500_uncertainty"].value, (2,1))) /halo["M500_truth"].value, capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
            plt.scatter(halo["M500_truth"],halo["Xray_ODR_HSE_M500"]/halo["M500_truth"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100, linewidths = 0.6)
        
        
        plt.sca(frame1)
        plt.plot(10**np.arange(10,20,step = 0.5), 10**np.arange(10,20,step = 0.5),  color = "green", lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.9 * 10**np.arange(10,20,step = 0.5),  color = "orange",  lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.7 * 10**np.arange(10,20,step = 0.5),  color = "sienna",  lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.4 * 10**np.arange(10,20,step = 0.5),  color = "teal",  lw = 2, ls = "solid", alpha = 0.514)
        
        if plot_running_median:
            plt.sca(frame2)
            x_data = np.array([halo["M500_truth"].value for halo in self.sample_arr if "Xray_ODR_HSE_M500" in halo.keys()])
            y_data = np.array([halo["Xray_ODR_HSE_M500"].value for halo in self.sample_arr if "Xray_ODR_HSE_M500" in halo.keys()])
            sort_idxs = np.argsort(x_data)
            x_data = x_data[sort_idxs]
            y_data = y_data[sort_idxs]
            running_median = median_filter(y_data/x_data, size = median_kernel_size)
            running_perc1 = percentile_filter(y_data/x_data,percentile = min(plot_running_percentiles), size = median_kernel_size)
            running_perc2 = percentile_filter(y_data/x_data,percentile = max(plot_running_percentiles), size = median_kernel_size)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)  
            plt.sca(frame1)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median*x_data)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)    
            plt.sca(frame1)
            if plot_running_percentiles != False:
                    plt.sca(frame2)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = running_perc1[median_lims[0]:median_lims[1]],y2 = running_perc2[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)   
                    plt.sca(frame1)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = (running_perc1*x_data)[median_lims[0]:median_lims[1]],y2 = (running_perc2*x_data)[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)       
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        if not plot_legend:
            plt.gca().get_legend().remove()
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,X_ODR}}{\mathtt{M}_\odot}$")
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.sca(frame2)
        ml = MultipleLocator(0.1)
        frame2.yaxis.set_minor_locator(ml)
        ml = MultipleLocator(1)
        frame2.yaxis.set_major_locator(ml)
        plt.sca(frame1)
        ax = plt.gca()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/Mx_ODR_vs_MSphOv.png", bbox_inches='tight', overwrite = True)
        plt.clf()
        plt.close() 
        
        
        
    def plot_MW_mass_bias(self, plot_running_median=True, median_kernel_size=3,  median_alpha = 0.5,  median_lims = (0,-1), plot_running_percentiles = (16,84), plot_legend = False):
        xmin_plt_lim = 8e11 
        xmax_plt_lim = 1e15 

        fig1 = plt.figure(figsize = (10,10), facecolor = 'w')
        frame1 = fig1.add_axes((.1,.3,.8,.6))
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.ylim(bottom = xmin_plt_lim, top = xmax_plt_lim)
        plt.yscale("log")
        plt.xscale('log')
        plt.xticks([], [])
        frame2 = fig1.add_axes((.1,.1,.8,.2))
        plt.hlines(y = 1, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "green", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.9, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "orange", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.7, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "sienna", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.4, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "teal", lw = 2, ls = "solid", alpha = 0.514)
        plt.ylim(bottom = 0,top = 2.5)       
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.xlabel(r"$\frac{\mathtt{M}_\mathtt{500,SO}}{\mathtt{M}_\odot}$")
        plt.xscale("log")
        plt.yscale("linear")
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,MW}}{\mathtt{M}_\mathtt{500,SO}}$")
        plt.sca(frame1)

        N = 29
        zscore_thresh = 5000

        
        for halo in self.sample_arr:
            plt.sca(frame1)
            # print("profile_yerr",halo["profile_HSE_M500_uncertainty"].value)
            plt.errorbar(halo["M500_truth"].value,halo["MW_HSE_M500"].value, yerr = np.reshape(halo["MW_HSE_M500_uncertainty"].value, (2,1)) , capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
            plt.scatter(halo["M500_truth"],halo["MW_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100)
            plt.sca(frame2)
            plt.scatter(halo["M500_truth"],halo["MW_HSE_M500"]/halo["M500_truth"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100)
            plt.errorbar(halo["M500_truth"].value,halo["MW_HSE_M500"].value/halo["M500_truth"].value, yerr = np.reshape(halo["MW_HSE_M500_uncertainty"].value, (2,1)) /halo["M500_truth"].value, capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
      
        
        plt.sca(frame1)
        plt.plot(10**np.arange(10,20,step = 0.5), 10**np.arange(10,20,step = 0.5),  color = "green", label = "b =  0", lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.9 * 10**np.arange(10,20,step = 0.5),  color = "orange", label = "b =  0.1", lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.7 * 10**np.arange(10,20,step = 0.5),  color = "sienna", label = "b =  0.3", lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.4 * 10**np.arange(10,20,step = 0.5),  color = "teal", label = "b =  0.6", lw = 2, ls = "solid", alpha = 0.514)
        
        if plot_running_median:
            plt.sca(frame2)
            x_data = np.array([halo["M500_truth"].value for halo in self.sample_arr])
            y_data = np.array([halo["MW_HSE_M500"].value for halo in self.sample_arr])
            sort_idxs = np.argsort(x_data)
            x_data = x_data[sort_idxs]
            y_data = y_data[sort_idxs]
            running_median = median_filter(y_data/x_data, size = median_kernel_size)
            running_perc1 = percentile_filter(y_data/x_data,percentile = min(plot_running_percentiles), size = median_kernel_size)
            running_perc2 = percentile_filter(y_data/x_data,percentile = max(plot_running_percentiles), size = median_kernel_size)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)  
            plt.sca(frame1)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median*x_data)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)    
            plt.sca(frame1)
            if plot_running_percentiles != False:
                    plt.sca(frame2)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = running_perc1[median_lims[0]:median_lims[1]],y2 = running_perc2[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)   
                    plt.sca(frame1)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = (running_perc1*x_data)[median_lims[0]:median_lims[1]],y2 = (running_perc2*x_data)[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)  
                
        
        
        
        
        plt.legend()
        if not plot_legend:
            plt.gca().get_legend().remove()
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,MW}}{\mathtt{M}_\odot}$")
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.sca(frame2)
        ml = MultipleLocator(0.1)
        frame2.yaxis.set_minor_locator(ml)
        ml = MultipleLocator(1)
        frame2.yaxis.set_major_locator(ml)
        plt.sca(frame1)
        ax = plt.gca()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/M_MW_vs_MSphOv.png", bbox_inches='tight', overwrite = True)
        plt.clf()
        plt.close()      
        

        
    def plot_EW_mass_bias(self, plot_running_median=True, median_kernel_size=3,  median_alpha = 0.5,  median_lims = (0,-1), plot_running_percentiles = (16,84), plot_legend = False):
        xmin_plt_lim = 8e11 
        xmax_plt_lim = 1e15 

        fig1 = plt.figure(figsize = (10,10), facecolor = 'w')
        frame1 = fig1.add_axes((.1,.3,.8,.6))
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.ylim(bottom = xmin_plt_lim, top = xmax_plt_lim)
        plt.yscale("log")
        plt.xscale('log')
        plt.xticks([], [])
        frame2 = fig1.add_axes((.1,.1,.8,.2))
        plt.hlines(y = 1, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "green", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.9, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "orange", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.7, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "sienna", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.4, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "teal", lw = 2, ls = "solid", alpha = 0.514)
        plt.ylim(bottom = 0,top = 2.5)       
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.xlabel(r"$\frac{\mathtt{M}_\mathtt{500,SO}}{\mathtt{M}_\odot}$")
        plt.xscale("log")
        plt.yscale("linear")
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,EW}}{\mathtt{M}_\mathtt{500,SO}}$")
        plt.sca(frame1)

        N = 29
        zscore_thresh = 5000

        
        for halo in self.sample_arr:
            plt.sca(frame1)
            # print("profile_yerr",halo["profile_HSE_M500_uncertainty"].value)
            plt.errorbar(halo["M500_truth"].value,halo["EW_HSE_M500"].value, yerr = np.reshape(halo["EW_HSE_M500_uncertainty"].value, (2,1)) , capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
            plt.scatter(halo["M500_truth"],halo["EW_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100)
            plt.sca(frame2)
            plt.scatter(halo["M500_truth"],halo["EW_HSE_M500"]/halo["M500_truth"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100)
            plt.errorbar(halo["M500_truth"].value,halo["EW_HSE_M500"].value/halo["M500_truth"].value, yerr = np.reshape(halo["EW_HSE_M500_uncertainty"].value, (2,1)) /halo["M500_truth"].value, capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
      
        
        plt.sca(frame1)
        plt.plot(10**np.arange(10,20,step = 0.5), 10**np.arange(10,20,step = 0.5),  color = "green", label = "b =  0", lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.9 * 10**np.arange(10,20,step = 0.5),  color = "orange", label = "b =  0.1", lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.7 * 10**np.arange(10,20,step = 0.5),  color = "sienna", label = "b =  0.3", lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.4 * 10**np.arange(10,20,step = 0.5),  color = "teal", label = "b =  0.6", lw = 2, ls = "solid", alpha = 0.514)
        
        if plot_running_median:
            plt.sca(frame2)
            x_data = np.array([halo["M500_truth"].value for halo in self.sample_arr])
            y_data = np.array([halo["EW_HSE_M500"].value for halo in self.sample_arr])
            sort_idxs = np.argsort(x_data)
            x_data = x_data[sort_idxs]
            y_data = y_data[sort_idxs]
            running_median = median_filter(y_data/x_data, size = median_kernel_size)
            running_perc1 = percentile_filter(y_data/x_data,percentile = min(plot_running_percentiles), size = median_kernel_size)
            running_perc2 = percentile_filter(y_data/x_data,percentile = max(plot_running_percentiles), size = median_kernel_size)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)  
            plt.sca(frame1)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median*x_data)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)    
            plt.sca(frame1)
            if plot_running_percentiles != False:
                    plt.sca(frame2)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = running_perc1[median_lims[0]:median_lims[1]],y2 = running_perc2[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)   
                    plt.sca(frame1)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = (running_perc1*x_data)[median_lims[0]:median_lims[1]],y2 = (running_perc2*x_data)[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)  

        plt.legend()
        if not plot_legend:
            plt.gca().get_legend().remove()
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,EW}}{\mathtt{M}_\odot}$")
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.sca(frame2)
        ml = MultipleLocator(0.1)
        frame2.yaxis.set_minor_locator(ml)
        ml = MultipleLocator(1)
        frame2.yaxis.set_major_locator(ml)
        plt.sca(frame1)
        ax = plt.gca()
        plt.tight_layout()

        plt.savefig(f"{self.save_dir}/M_EW_vs_MSphOv.png", bbox_inches='tight', overwrite = True)
        plt.clf()
        plt.close()      


    def plot_LW_mass_bias(self, plot_running_median=True, median_kernel_size=3,  median_alpha = 0.5,  median_lims = (0,-1), plot_running_percentiles = (16,84), plot_legend = False):
        xmin_plt_lim = 8e11 
        xmax_plt_lim = 1e15 

        fig1 = plt.figure(figsize = (10,10), facecolor = 'w')
        frame1 = fig1.add_axes((.1,.3,.8,.6))
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.ylim(bottom = xmin_plt_lim, top = xmax_plt_lim)
        plt.yscale("log")
        plt.xscale('log')
        plt.xticks([], [])
        frame2 = fig1.add_axes((.1,.1,.8,.2))
        plt.hlines(y = 1, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "green", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.9, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "orange", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.7, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "sienna", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.4, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "teal", lw = 2, ls = "solid", alpha = 0.514)
        plt.ylim(bottom = 0,top = 2.5)       
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.xlabel(r"$\frac{\mathtt{M}_\mathtt{500,SO}}{\mathtt{M}_\odot}$")
        plt.xscale("log")
        plt.yscale("linear")
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,LW}}{\mathtt{M}_\mathtt{500,SO}}$")
        plt.sca(frame1)

        N = 29
        zscore_thresh = 5000

        
        for halo in self.sample_arr:
            plt.sca(frame1)
            # print("profile_yerr",halo["profile_HSE_M500_uncertainty"].value)
            plt.errorbar(halo["M500_truth"].value,halo["LW_HSE_M500"].value, yerr = np.reshape(halo["LW_HSE_M500_uncertainty"].value, (2,1)) , capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
            plt.scatter(halo["M500_truth"],halo["LW_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100)
            plt.sca(frame2)
            plt.scatter(halo["M500_truth"],halo["LW_HSE_M500"]/halo["M500_truth"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100)
            plt.errorbar(halo["M500_truth"].value,halo["LW_HSE_M500"].value/halo["M500_truth"].value, yerr = np.reshape(halo["LW_HSE_M500_uncertainty"].value, (2,1)) /halo["M500_truth"].value, capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
      
        
        plt.sca(frame1)
        plt.plot(10**np.arange(10,20,step = 0.5), 10**np.arange(10,20,step = 0.5),  color = "green", label = "b =  0", lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.9 * 10**np.arange(10,20,step = 0.5),  color = "orange", label = "b =  0.1", lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.7 * 10**np.arange(10,20,step = 0.5),  color = "sienna", label = "b =  0.3", lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.4 * 10**np.arange(10,20,step = 0.5),  color = "teal", label = "b =  0.6", lw = 2, ls = "solid", alpha = 0.514)
        
        if plot_running_median:
            plt.sca(frame2)
            x_data = np.array([halo["M500_truth"].value for halo in self.sample_arr])
            y_data = np.array([halo["LW_HSE_M500"].value for halo in self.sample_arr])
            sort_idxs = np.argsort(x_data)
            x_data = x_data[sort_idxs]
            y_data = y_data[sort_idxs]
            running_median = median_filter(y_data/x_data, size = median_kernel_size)
            running_perc1 = percentile_filter(y_data/x_data,percentile = min(plot_running_percentiles), size = median_kernel_size)
            running_perc2 = percentile_filter(y_data/x_data,percentile = max(plot_running_percentiles), size = median_kernel_size)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)  
            plt.sca(frame1)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median*x_data)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)    
            plt.sca(frame1)
            if plot_running_percentiles != False:
                    plt.sca(frame2)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = running_perc1[median_lims[0]:median_lims[1]],y2 = running_perc2[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)   
                    plt.sca(frame1)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = (running_perc1*x_data)[median_lims[0]:median_lims[1]],y2 = (running_perc2*x_data)[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)  

                
        
        
        
        
        plt.legend()
        if not plot_legend:
            plt.gca().get_legend().remove()
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,LW}}{\mathtt{M}_\odot}$")
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.sca(frame2)
        ml = MultipleLocator(0.1)
        frame2.yaxis.set_minor_locator(ml)
        ml = MultipleLocator(1)
        frame2.yaxis.set_major_locator(ml)
        plt.sca(frame1)
        ax = plt.gca()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/M_LW_vs_MSphOv.png", bbox_inches='tight', overwrite = True)
        plt.clf()
        plt.close()      

        

        
 


    def plot_Xray_mass_vs_EW_mass(self, plot_running_median = True, error_alpha = 0.5, scatter_linewidth = 0.9, median_kernel_size = 5, median_alpha = 0.5, median_lims = (0,-1), xmin_plt_lim = 3e12,xmax_plt_lim = 9.9e14, plot_running_percentiles = (16,84), plot_legend = False,  plot_ODR_median = False, bad_ODR_idxs = []):
        print("NEED TO MAKE SURE ALL THIS FUNCTION CORRECT! (plot_Xray_mass_vs_EW_mass)")

        fig1 = plt.figure(figsize = (10,10), facecolor = 'w')
        frame1 = fig1.add_axes((.1,.3,.8,.6))
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.ylim(bottom = xmin_plt_lim, top = xmax_plt_lim)
        plt.yscale("log")
        plt.xscale('log')
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,X}}{\mathtt{M}_\odot}$")
        plt.xticks([], [])
        
        frame2 = fig1.add_axes((.1,.1,.8,.2))
        plt.hlines(y = 1, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "green", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.9, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "orange", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.7, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "sienna", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.4, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "teal", lw = 2, ls = "solid", alpha = 0.514)
        
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,X}}{\mathtt{M}_\mathtt{500,EW}}$")
        plt.ylim(bottom = 0,top = 2.5)
        plt.xscale("log")
        plt.yscale("linear")
        plt.xlabel(r"$\frac{\mathtt{M}_\mathtt{500,EW}}{\mathtt{M}_\odot}$")
        plt.sca(frame1)

        N = 29
        zscore_thresh = 5000

        plotted_labels = []
        for halo in self.sample_arr:
            if "Xray_HSE_M500" not in halo.keys():
                continue
            plt.sca(frame1)
            # print("yerr", halo["Xray_HSE_M500_uncertainty"].value)
            # print("reshaped yerr", np.reshape(halo["Xray_HSE_M500_uncertainty"].value, (2,1)))
            # print("\n")
            try:
                plt.errorbar(halo["EW_HSE_M500"].value, halo["Xray_HSE_M500"].value, xerr = np.reshape(halo["EW_HSE_M500_uncertainty"].value, (2,1)), yerr = np.reshape(halo["Xray_HSE_M500_uncertainty"].value, (2,1)) , capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3, alpha = error_alpha)
            except:
                plt.errorbar(halo["EW_HSE_M500"].value, halo["Xray_HSE_M500"].value,  yerr = np.reshape(halo["Xray_HSE_M500_uncertainty"].value, (2,1)) , capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3, alpha = error_alpha)
                pass
            
            if halo['label'] in plotted_labels:
                plt.scatter(halo["EW_HSE_M500"],   halo["Xray_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100, linewidths = scatter_linewidth)
            else:
                plt.scatter(halo["EW_HSE_M500"],   halo["Xray_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100, linewidths = scatter_linewidth, label = halo['label'])
                
                
            plt.sca(frame2)
            try:
                plt.errorbar(halo["EW_HSE_M500"].value, halo["Xray_HSE_M500"].value/halo["EW_HSE_M500"].value, xerr = np.reshape(halo["EW_HSE_M500_uncertainty"].value, (2,1)), yerr = np.reshape(halo["Xray_HSE_M500_uncertainty"].value, (2,1)) /halo["EW_HSE_M500"].value, capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3, alpha = error_alpha)
            except:
                plt.errorbar(halo["EW_HSE_M500"].value, halo["Xray_HSE_M500"].value/halo["EW_HSE_M500"].value, yerr = np.reshape(halo["Xray_HSE_M500_uncertainty"].value, (2,1)) /halo["EW_HSE_M500"].value, capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3, alpha = error_alpha)
                pass
                         
            plt.scatter(halo["EW_HSE_M500"],  halo["Xray_HSE_M500"]/halo["EW_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100, linewidths = scatter_linewidth)
        
        
        plt.sca(frame1)
        plt.plot(10**np.arange(10,20,step = 0.5), 10**np.arange(10,20,step = 0.5),  color = "green", lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.9 * 10**np.arange(10,20,step = 0.5),  color = "orange", lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.7 * 10**np.arange(10,20,step = 0.5),  color = "sienna", lw = 2, ls = "solid", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.4 * 10**np.arange(10,20,step = 0.5),  color = "teal", lw = 2, ls = "solid", alpha = 0.514)
        
        
        if plot_running_median:
            plt.sca(frame2)
            x_data = np.array([halo["EW_HSE_M500"].value for halo in self.sample_arr if "Xray_HSE_M500" in halo.keys()])
            y_data = np.array([halo["Xray_HSE_M500"].value for halo in self.sample_arr if "Xray_HSE_M500" in halo.keys()])
            sort_idxs = np.argsort(x_data)
            x_data = x_data[sort_idxs]
            y_data = y_data[sort_idxs]
            running_median = median_filter(y_data/x_data, size = median_kernel_size)
            running_perc1 = percentile_filter(y_data/x_data,percentile = min(plot_running_percentiles), size = median_kernel_size)
            running_perc2 = percentile_filter(y_data/x_data,percentile = max(plot_running_percentiles), size = median_kernel_size)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)  
            plt.sca(frame1)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median*x_data)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)    
            if plot_running_percentiles != False:
                    plt.sca(frame2)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = running_perc1[median_lims[0]:median_lims[1]],y2 = running_perc2[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)   
                    plt.sca(frame1)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = (running_perc1*x_data)[median_lims[0]:median_lims[1]],y2 = (running_perc2*x_data)[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)  
            if plot_ODR_median:
                plt.sca(frame2)
                x_data = np.array([halo["EW_HSE_M500"].value for halo in self.sample_arr if "Xray_HSE_M500" in halo.keys() and  int(halo["idx"]) not in bad_ODR_idxs])
                y_data = np.array([halo["Xray_ODR_HSE_M500"].value for halo in self.sample_arr if "Xray_HSE_M500" in halo.keys() and  int(halo["idx"]) not in bad_ODR_idxs])
                sort_idxs = np.argsort(x_data)
                x_data = x_data[sort_idxs]
                y_data = y_data[sort_idxs]
                running_median = median_filter(y_data/x_data, size = median_kernel_size)
                running_perc1 = percentile_filter(y_data/x_data,percentile = min(plot_running_percentiles), size = median_kernel_size)
                running_perc2 = percentile_filter(y_data/x_data,percentile = max(plot_running_percentiles), size = median_kernel_size)
                plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median)[median_lims[0]:median_lims[1]], color = "red", lw = 3, ls = "dashed", alpha = median_alpha)  
                plt.sca(frame1)
                plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median*x_data)[median_lims[0]:median_lims[1]], color = "red", lw = 3, ls = "dashed", alpha = median_alpha) 

                    

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        if not plot_legend:
            plt.gca().get_legend().remove()
        # plt.ylabel(r"$\frac{M500_\mathtt{SphOv}}{M_\odot}$", fontsize = 30)

        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.sca(frame2)
        ml = MultipleLocator(0.1)
        frame2.yaxis.set_minor_locator(ml)
        ml = MultipleLocator(1)
        frame2.yaxis.set_major_locator(ml)
        plt.sca(frame1)
        ax = plt.gca()
        self.set_plotstyle()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/Mx_vs_M_EW.png", bbox_inches='tight', overwrite = True)
        plt.clf()
        plt.close() 

    def plot_Xray_mass_vs_LW_mass(self, plot_running_median = True, error_alpha = 0.5, scatter_linewidth = 0.9, median_kernel_size = 5, median_alpha = 0.5, median_lims = (0,-1), plot_running_percentiles = (16,84), plot_legend = False,  plot_ODR_median = False, bad_ODR_idxs = []):
        print("NEED TO MAKE SURE ALL THIS FUNCTION CORRECT! (plot_Xray_mass_vs_LW_mass)")
        xmin_plt_lim = 8e11 
        xmax_plt_lim = 1e15 

        fig1 = plt.figure(figsize = (10,10), facecolor = 'w')
        frame1 = fig1.add_axes((.1,.3,.8,.6))
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.ylim(bottom = xmin_plt_lim, top = xmax_plt_lim)
        plt.yscale("log")
        plt.xscale('log')
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,X}}{\mathtt{M}_\odot}$")
        plt.xticks([], [])
        
        frame2 = fig1.add_axes((.1,.1,.8,.2))
        plt.hlines(y = 1, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "green", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.9, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "orange", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.7, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "sienna", lw = 2, ls = "solid", alpha = 0.514)
        plt.hlines(y = 0.4, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "teal", lw = 2, ls = "solid", alpha = 0.514)
        
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,X}}{\mathtt{M}_\mathtt{500,LW}}$")
        plt.ylim(bottom = 0,top = 2.5)
        plt.xscale("log")
        plt.yscale("linear")
        plt.xlabel(r"$\frac{\mathtt{M}_\mathtt{500,LW}}{\mathtt{M}_\odot}$")
        plt.sca(frame1)

        N = 29
        zscore_thresh = 5000

        plotted_labels = []
        for halo in self.sample_arr:
            plt.sca(frame1)
            try:
                plt.errorbar(halo["LW_HSE_M500"].value, halo["Xray_HSE_M500"].value, xerr = np.reshape(halo["LW_HSE_M500_uncertainty"].value, (2,1)), yerr = np.reshape(halo["Xray_HSE_M500_uncertainty"].value, (2,1)) , capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3, alpha = error_alpha)
            except:
                plt.errorbar(halo["LW_HSE_M500"].value, halo["Xray_HSE_M500"].value,  yerr = np.reshape(halo["Xray_HSE_M500_uncertainty"].value, (2,1)) , capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3, alpha = error_alpha)
                pass
            
            if halo['label'] in plotted_labels:
                plt.scatter(halo["LW_HSE_M500"],   halo["Xray_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100, linewidths = scatter_linewidth)
            else:
                plt.scatter(halo["LW_HSE_M500"],   halo["Xray_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100, linewidths = scatter_linewidth, label = halo['label'])
                
                
            plt.sca(frame2)
            try:
                plt.errorbar(halo["LW_HSE_M500"].value, halo["Xray_HSE_M500"].value/halo["LW_HSE_M500"].value, xerr = np.reshape(halo["LW_HSE_M500_uncertainty"].value, (2,1)), yerr = np.reshape(halo["Xray_HSE_M500_uncertainty"].value, (2,1)) /halo["LW_HSE_M500"].value, capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3, alpha = error_alpha)
            except:
                plt.errorbar(halo["LW_HSE_M500"].value, halo["Xray_HSE_M500"].value/halo["LW_HSE_M500"].value, yerr = np.reshape(halo["Xray_HSE_M500_uncertainty"].value, (2,1)) /halo["LW_HSE_M500"].value, capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3, alpha = error_alpha)
                pass
                         
            plt.scatter(halo["LW_HSE_M500"],  halo["Xray_HSE_M500"]/halo["LW_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100, linewidths = scatter_linewidth)
        
        
        plt.sca(frame1)
        plt.plot(10**np.arange(10,20,step = 0.5), 10**np.arange(10,20,step = 0.5),  color = "green", lw = 2, ls = "dashed", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.9 * 10**np.arange(10,20,step = 0.5),  color = "orange", lw = 2, ls = "dashed", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.7 * 10**np.arange(10,20,step = 0.5),  color = "sienna", lw = 2, ls = "dashed", alpha = 0.514)
        plt.plot(10**np.arange(10,20,step = 0.5), 0.4 * 10**np.arange(10,20,step = 0.5),  color = "teal", lw = 2, ls = "dashed", alpha = 0.514)
        
        
        if plot_running_median:
            plt.sca(frame2)
            x_data = np.array([halo["LW_HSE_M500"].value for halo in self.sample_arr])
            y_data = np.array([halo["Xray_HSE_M500"].value for halo in self.sample_arr])
            sort_idxs = np.argsort(x_data)
            x_data = x_data[sort_idxs]
            y_data = y_data[sort_idxs]
            running_median = median_filter(y_data/x_data, size = median_kernel_size)
            running_perc1 = percentile_filter(y_data/x_data,percentile = min(plot_running_percentiles), size = median_kernel_size)
            running_perc2 = percentile_filter(y_data/x_data,percentile = max(plot_running_percentiles), size = median_kernel_size)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)  
            plt.sca(frame1)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median*x_data)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)    
            if plot_running_percentiles != False:
                    plt.sca(frame2)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = running_perc1[median_lims[0]:median_lims[1]],y2 = running_perc2[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)   
                    plt.sca(frame1)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = (running_perc1*x_data)[median_lims[0]:median_lims[1]],y2 = (running_perc2*x_data)[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)  
            if plot_ODR_median:
                plt.sca(frame2)
                x_data = np.array([halo["LW_HSE_M500"].value for halo in self.sample_arr if int(halo["idx"]) not in bad_ODR_idxs])
                y_data = np.array([halo["Xray_ODR_HSE_M500"].value for halo in self.sample_arr if int(halo["idx"]) not in bad_ODR_idxs])
                sort_idxs = np.argsort(x_data)
                x_data = x_data[sort_idxs]
                y_data = y_data[sort_idxs]
                running_median = median_filter(y_data/x_data, size = median_kernel_size)
                running_perc1 = percentile_filter(y_data/x_data,percentile = min(plot_running_percentiles), size = median_kernel_size)
                running_perc2 = percentile_filter(y_data/x_data,percentile = max(plot_running_percentiles), size = median_kernel_size)
                plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median*x_data)[median_lims[0]:median_lims[1]], color = "red", lw = 3, ls = "dashed", alpha = median_alpha)  

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        if not plot_legend:
            plt.gca().get_legend().remove()
        # plt.ylabel(r"$\frac{M500_\mathtt{SphOv}}{M_\odot}$", fontsize = 30)

        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.sca(frame2)
        ml = MultipleLocator(0.1)
        frame2.yaxis.set_minor_locator(ml)
        ml = MultipleLocator(1)
        frame2.yaxis.set_major_locator(ml)
        plt.sca(frame1)
        ax = plt.gca()
        self.set_plotstyle()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/Mx_vs_M_LW.png", bbox_inches='tight', overwrite = True)
        plt.clf()
        plt.close() 

        

    
    def plot_loglog_scaling(self, x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, best_fit_min_mass = 0, best_fit_max_mass = 1e50, cscale = None, cscale_scale = "log", data_labels = True,  plot_self_similar_best_fit = False, plot_best_fit=True, external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed"}, legend_fontsize = 20 ):
        import bces.bces as BCES
        import nmmn.stats
        if mass_key == "Xray_HSE_M500": 
            marker_color = "indianred" # halo["marker color"]
        elif mass_key == "MW_HSE_M500": 
            marker_color = "cornflowerblue" # halo["marker color"]
        else:
            marker_color = "seagreen" # halo["marker color"]
            
        print(f"\n Plotting {y_property_key} vs {x_property_key}")
        fig1 = plt.figure(figsize = (10,10), facecolor = 'w')
        halo_samples = np.array([halo for halo in self.sample_arr if y_property_key in halo.keys() and x_property_key in halo.keys()])
        halo_samples = np.array([halo for halo in halo_samples if halo[y_property_key].value > 0])
        x_data = np.array([halo[x_property_key].value for halo in halo_samples])
        mass_data = np.array([halo[mass_key].value for halo in halo_samples])
        
        if xerr_key != None:
            x_data_err = np.array([halo[xerr_key].value for halo in halo_samples])
        else:
            x_data_err = np.zeros_like(x_data)
        
        if hubble_correction:
            y_data = np.array([halo[y_property_key].value * (self.cosmo.H(z=halo["redshift"])/self.cosmo.H(z=0))**(Ez_power) for halo in halo_samples])
            if yerr_key != None:
                y_data_err = np.array([halo[yerr_key].value * (self.cosmo.H(z=halo["redshift"])/self.cosmo.H(z=0))**(Ez_power) for halo in halo_samples])
            else:
                y_data_err = np.zeros_like(x_data)
        elif not hubble_correction:
            y_data = np.array([halo[y_property_key].value for halo in halo_samples])
            if yerr_key != None:
                y_data_err = np.array([halo[yerr_key].value for halo in halo_samples])
            else:
                y_data_err = np.zeros_like(x_data)
            Ez_correction = 1
 
        if cscale == None:
            plotted_labels = []
            for halo in halo_samples:
                if hubble_correction:
                    Ez_correction = (self.cosmo.H(z=halo["redshift"])/self.cosmo.H(z=0))**(Ez_power)
                else:
                    Ez_correction = 1
                
                try:
                    if xerr_key != None:
                        xerr = np.reshape(halo[xerr_key].value, (2,1))
                    else:
                        xerr = 0
                except:
                    xerr = halo[xerr_key].value
                try:
                    if yerr_key != None:
                        yerr = np.reshape(halo[yerr_key].value, (2,1))
                    else:
                        yerr = 0
                except:
                    yerr = halo[yerr_key].value
                if xerr_key == None and yerr_key == None:
                    plt.scatter(halo[x_property_key],    Ez_correction * halo[y_property_key], facecolors='none', edgecolors=marker_color, marker = "^", s = 100, linewidths = 0.6) 
                    pass
                elif xerr_key == None:
                    plt.errorbar(halo[x_property_key].value, Ez_correction * halo[y_property_key].value,  yerr = Ez_correction * yerr, capsize = 5, fmt = 'None', color = marker_color, elinewidth = 0.3)                    
                elif yerr_key == None:
                    plt.errorbar(halo[x_property_key].value, Ez_correction * halo[y_property_key].value, xerr = xerr, capsize = 5, fmt = 'None', color = marker_color, elinewidth = 0.3)
                else:
                    plt.errorbar(halo[x_property_key].value, Ez_correction * halo[y_property_key].value, xerr = xerr, yerr = Ez_correction * yerr, capsize = 5, fmt = 'None', color = marker_color, elinewidth = 0.3)
                if halo['label'] in plotted_labels or data_labels == False:
                    plt.scatter(halo[x_property_key],    Ez_correction * halo[y_property_key], facecolors='none', edgecolors=marker_color, marker = "^", s = 100, linewidths = 0.9)
                else:
                    plt.scatter(halo[x_property_key],    Ez_correction * halo[y_property_key], facecolors='none', edgecolors=marker_color, marker = "^", s = 100, linewidths = 0.9, label = halo['label'])           
        else:
            z_data = np.array([halo[cscale].value for halo in halo_samples])
            if cscale_scale == "log":
                sc = plt.scatter(x_data, y_data, c = z_data, norm=matplotlib.colors.LogNorm(), s = 100, linewidths = 0.6)
            else: 
                sc = plt.scatter(x_data, y_data, c = z_data, s = 100, linewidths = 0.6)
            axes = np.array(fig1.get_axes())
            plt.colorbar(sc, ax = axes.ravel().tolist())

        
        def best_fit_slope(log_x,b,m):
            return m*log_x + b

        cov = np.zeros_like(x_data)
        errx = np.array( [x_data_err[i]/(float(x_data[i])*np.log(10)) for i in range(len(x_data_err))])
        erry = np.array( [y_data_err[i]/(float(y_data[i])*np.log(10)) for i in range(len(y_data_err))])
        errx = np.array([np.sum(err)/2 for err in errx])
        erry = np.array([np.sum(err)/2 for err in erry])
        
        sort_idxs = np.argsort(mass_data)
        
        bces_x_data = np.array(x_data[sort_idxs])
        bces_y_data = np.array(y_data[sort_idxs])
        bces_m_data = np.array(mass_data[sort_idxs])
        bces_errx = np.array(errx[sort_idxs])
        bces_erry = np.array(erry[sort_idxs])
        if best_fit_min_mass != None and best_fit_min_mass > 0: print(f"BCES fit will include halos with {mass_key} > 10$^{{{np.log10(best_fit_min_mass)}}}$")
        mass_range_idxs = np.where( (bces_m_data >= best_fit_min_mass) & (bces_m_data <= best_fit_max_mass) )[0]
        print(f"{len(mass_range_idxs)} out of {len(bces_m_data)} fall in the required mass range ({mass_key} > 10$^{{{np.log10(best_fit_min_mass)}}}$)")
        bces_x_data = np.array(bces_x_data[mass_range_idxs])
        bces_y_data = np.array(bces_y_data[mass_range_idxs])
        bces_m_data = np.array(bces_m_data[mass_range_idxs])
        bces_errx = np.array(bces_errx[mass_range_idxs])
        bces_erry = np.array(bces_erry[mass_range_idxs])
        min_length = 10
        if len(bces_x_data) < min_length:
            print("Less than required number for bces")
            return [None,], None, len(bces_x_data)
        
        bces_x_data = np.log10(bces_x_data)
        bces_y_data = np.log10(bces_y_data)
        #print("bces_x_data", bces_x_data)
        nboot=100_000
        bcesMethod = 3
        a,b,erra,errb,covab=BCES.bcesp(y1= bces_x_data,y1err = bces_errx,y2 = bces_y_data,y2err=bces_erry,cerr=cov,nsim=nboot)  
        # From https://github.com/rsnemmen/BCES/blob/master/bces-examples.ipynb
        # array with best-fit parameters
        fitm=np.array([ a[bcesMethod],b[bcesMethod] ])	
        # covariance matrix of parameter uncertainties
        print(f"BCES Method = {bcesMethod}, with {nboot} bootstrap samples")
        covm=np.array([ (erra[bcesMethod]**2,covab[bcesMethod]), (covab[bcesMethod],errb[bcesMethod]**2) ])	   
        def nmmnfunc(x): return x[1]*x[0]+x[2]
        confidence_interval = 0.68 #0.999999426 #0.997 #0.68 #0.997 #0.68

        ## http://rsnemmen.github.io/nmmn/nmmn.html
        #### Naturally Sometimes fails if not enough data
        lcb,ucb,xcb=nmmn.stats.confbandnl(bces_x_data,bces_y_data,nmmnfunc,fitm,covm,2,confidence_interval,np.linspace(bces_x_data.min() - 3, bces_x_data.max() + 3, num = 1000))
        if plot_best_fit:
            if best_fit_min_mass != None and best_fit_min_mass>0:
                mass_tag = mass_key.split("_")[0]
                if mass_key == "M500_truth": mass_tag = "SO"
                if mass_tag == "Xray": mass_tag = "X"
                plt.plot([xmin_plt_lim,xmax_plt_lim],  (10**b[bcesMethod]) * (np.array([xmin_plt_lim,xmax_plt_lim])**(a[bcesMethod])), **best_fit_kwargs, label = f"Best Fit ($\mathtt{{M}}_\mathtt{{500,{mass_tag}}}$ > 10$^{{{np.log10(best_fit_min_mass)}}}$), m = {round(a[bcesMethod],2)}" )  

            else:
                plt.plot([xmin_plt_lim,xmax_plt_lim],  (10**b[bcesMethod]) * (np.array([xmin_plt_lim,xmax_plt_lim])**(a[bcesMethod])), **best_fit_kwargs, label = f"Best Fit (m = {round(a[bcesMethod],2)})" )               
            plt.fill_between(10**xcb, 10**lcb, 10**ucb, alpha = 0.1, color = "teal")
            
            

        if len(external_dsets) != 0:
            import csv
            for dset_dict in external_dsets:
                with open(f'external_data/{dset_dict["filename"]}.csv') as csv_file:
                    if not dset_dict.get("actually_plot", True):
                        continue
                    print(f"Plotting from external_data/{dset_dict['filename']}.csv")
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    dset_x_data = []
                    dset_y_data = []
                    for row in csv_reader:
                        if dset_dict.get("flip_x_and_y", False) == True:
                            dset_y_data.append(float(row[0]))
                            dset_x_data.append(float(row[1]))                           
                        else:
                            dset_x_data.append(float(row[0]))
                            dset_y_data.append(float(row[1]))
                if dset_dict.get("only_plot_ends", False):
                    dset_x_data = [dset_x_data[0], dset_x_data[-1]]
                    dset_y_data = [dset_y_data[0], dset_y_data[-1]]
                sort_idxs = np.argsort(dset_x_data)
                dset_x_data = np.array(dset_x_data)[sort_idxs]
                dset_y_data = np.array(dset_y_data)[sort_idxs]
                dset_m = ( np.log10(dset_y_data[-1])-np.log10(dset_y_data[0]) )/( np.log10(dset_x_data[-1])-np.log10(dset_x_data[0]) )
                dset_b = np.log10(dset_y_data[0]) - dset_m*np.log10(dset_x_data[0])
                if dset_dict.get('extend', False):
                    dset_x_data[0] = 1e-5 * dset_x_data[0]
                    dset_y_data[0] = 10**(dset_m*np.log10(dset_x_data[0]) + dset_b)
                    dset_x_data[-1] = 1e5 * dset_x_data[-1]
                    dset_y_data[-1] = 10**(dset_m*np.log10(dset_x_data[-1]) + dset_b)
                if dset_dict.get("needs_hubble_correct", False):
                    print(f"Dataset needs E(z) correction, so we will multiply the Y value by $(H{{{dset_dict['hubble_correct_z']}}}/H0)^{{{Ez_power}}}$")
                    Ez_correction = (self.cosmo.H(z=dset_dict['hubble_correct_z'])/self.cosmo.H(z=0))**(Ez_power)
                    dset_y_data = dset_y_data * Ez_correction
                if dset_dict["plot type"] == "plot":
                    plt.plot(dset_dict.get("x_adjust", 1)*dset_x_data, dset_dict.get("y_adjust", 1)*dset_y_data, label = dset_dict["label"], **dset_dict["plot_kwds"] )
                if dset_dict["plot type"] == "scatter":
                    plt.scatter(dset_dict.get("x_adjust", 1)*dset_x_data, dset_dict.get("y_adjust", 1)*dset_y_data, label = dset_dict["label"], **dset_dict["plot_kwds"] )

          
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize = legend_fontsize)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
        plt.ylim(bottom = ymin_plt_lim, top = ymax_plt_lim)
        plt.yscale("log")
        plt.xscale('log')
        os.makedirs(f"{self.save_dir}/scaling_relations/", exist_ok=True)
        plt.savefig(f"{self.save_dir}/scaling_relations/{save_name}.png", bbox_inches='tight')
        plt.clf()
        plt.close() 
        
        
        
        
    def plot_weighted_kT_Xray_scaling(self, cscale = None,mass_key=None, cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 600, plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,  legend_fontsize = 20  ):
        Ez_power = 2/3
        self_sim_scaling_power = 2/3
        
        x_property_key = "Xray_HSE_M500"
        y_property_key = "weighted_kT_Xray"
        xerr_key = "Xray_HSE_M500_uncertainty"
        yerr_key = "weighted_kT_Xray_spread"
        xlabel = r"$\mathtt{M}_\mathtt{500,X}$  [$\mathtt{M}_\odot$]"
        if mass_key == None: mass_key =  "Xray_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{-2/3}}$kT$_\mathtt{X}$[keV]"
        if not hubble_correction:
            ylabel = r"$kT_x$[keV]"
        if cscale == None:
            if hubble_correction:
                save_name = f"weighted_kT_Xray_Ez_corrected"
            else:
                save_name = f"weighted_kT_Xray"
        else:
            if hubble_correction:
                save_name = f"weighted_kT_Xray_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"weighted_kT_Xray_vs_{cscale}"   
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass, legend_fontsize = legend_fontsize )    
        
    def plot_kTR500_Xray_scaling(self, cscale = None,mass_key=None, cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 600, plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,  legend_fontsize = 20  ):
        Ez_power = 2/3
        self_sim_scaling_power = 2/3
        
        x_property_key = "Xray_HSE_M500"
        y_property_key = "kT_Xray_at_Xray_HSE_R500"
        xerr_key = "Xray_HSE_M500_uncertainty"
        yerr_key = "kT_Xray_at_Xray_HSE_R500_spread"
        xlabel = r"$\mathtt{M}_\mathtt{500,X}$  [$\mathtt{M}_\odot$]"
        if mass_key == None: mass_key =  "Xray_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{-2/3}}$kT$_\mathtt{X}$($R_\mathtt{500,X}$) [keV]"
        if not hubble_correction:
            ylabel = r"kT$_\mathtt{X}$($R_\mathtt{500,X}$) [keV]"
        if cscale == None:
            if hubble_correction:
                save_name = f"kTR500_Xray_Ez_corrected"
            else:
                save_name = f"kTR500_Xray"
        else:
            if hubble_correction:
                save_name = f"kTR500_Xray_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"kTR500_Xray_vs_{cscale}"   
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass, legend_fontsize = legend_fontsize )  
   

    def plot_weighted_kT_MW_scaling(self, cscale = None,mass_key=None, cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 600,   plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,    legend_fontsize = 20  ):
        Ez_power = 2/3
        self_sim_scaling_power = 2/3
        
        x_property_key = "MW_HSE_M500"
        y_property_key = "weighted_kT_MW"
        xerr_key = "MW_HSE_M500_uncertainty"
        yerr_key = "weighted_kT_MW_spread"
        xlabel = r"$\mathtt{M}_\mathtt{500,MW}$  [$\mathtt{M}_\odot$]"
        if mass_key == None: mass_key =  "MW_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{-2/3}}$kT$_\mathtt{MW}$[keV]"
        if not hubble_correction:
            ylabel = r"$kT_{MW}$[keV]"
        if cscale == None:
            if hubble_correction:
                save_name = f"weighted_kT_MW_Ez_corrected"
            else:
                save_name = f"weighted_kT_MW"
        else:
            if hubble_correction:
                save_name = f"weighted_kT_MW_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"weighted_kT_MW_vs_{cscale}"   
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )   

    def plot_kTR500_MW_scaling(self, cscale = None,mass_key=None, cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 600, plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,  legend_fontsize = 20  ):
        Ez_power = 2/3
        self_sim_scaling_power = 2/3
        
        x_property_key = "MW_HSE_M500"
        y_property_key = "kT_MW_at_MW_HSE_R500"
        xerr_key = "MW_HSE_M500_uncertainty"
        yerr_key = "kT_MW_at_MW_HSE_R500_spread"
        xlabel = r"$\mathtt{M}_\mathtt{500,MW}$  [$\mathtt{M}_\odot$]"
        if mass_key == None: mass_key =  "MW_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{-2/3}}$kT$_\mathtt{MW}$($R_\mathtt{500,MW}$) [keV]"
        if not hubble_correction:
            ylabel = r"kT$_\mathtt{MW}$($R_\mathtt{500,MW}$) [keV]"
        if cscale == None:
            if hubble_correction:
                save_name = f"kTR500_MW_Ez_corrected"
            else:
                save_name = f"kTR500_MW"
        else:
            if hubble_correction:
                save_name = f"kTR500_MW_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"kTR500_MW_vs_{cscale}"   
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass, legend_fontsize = legend_fontsize )  
           
        
        
        
        
        
        
    def plot_weighted_kT_EW_scaling(self, cscale = None,mass_key=None, cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 600,   plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,    legend_fontsize = 20  ):
        Ez_power = 2/3
        self_sim_scaling_power = 2/3
        
        x_property_key = "EW_HSE_M500"
        y_property_key = "weighted_kT_EW"
        xerr_key = "EW_HSE_M500_uncertainty"
        yerr_key = "weighted_kT_EW_spread"
        xlabel = r"$\mathtt{M}_\mathtt{500,EW}$  [$\mathtt{M}_\odot$]"
        if mass_key == None: mass_key =  "EW_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{-2/3}}$kT$_\mathtt{EW}$[keV]"
        if not hubble_correction:
            ylabel = r"$kT_{EW}$[keV]"
        if cscale == None:
            if hubble_correction:
                save_name = f"weighted_kT_EW_Ez_corrected"
            else:
                save_name = f"weighted_kT_EW"
        else:
            if hubble_correction:
                save_name = f"weighted_kT_EW_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"weighted_kT_EW_vs_{cscale}"   
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )   
        
        
    def plot_weighted_kT_LW_scaling(self, cscale = None,mass_key=None, cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 600,   plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,    legend_fontsize = 20  ):
        Ez_power = 2/3
        self_sim_scaling_power = 2/3
        
        x_property_key = "LW_HSE_M500"
        y_property_key = "weighted_kT_LW"
        xerr_key = "LW_HSE_M500_uncertainty"
        yerr_key = "weighted_kT_LW_spread"
        xlabel = r"$\mathtt{M}_\mathtt{500,LW}$  [$\mathtt{M}_\odot$]"
        if mass_key == None: mass_key =  "LW_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{-2/3}}$kT$_\mathtt{LW}$[keV]"
        if not hubble_correction:
            ylabel = r"$kT_{LW}$[keV]"
        if cscale == None:
            if hubble_correction:
                save_name = f"weighted_kT_LW_Ez_corrected"
            else:
                save_name = f"weighted_kT_LW"
        else:
            if hubble_correction:
                save_name = f"weighted_kT_LW_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"weighted_kT_LW_vs_{cscale}"   
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )     






        
    def plot_M500_vs_weighted_kT_Xray_scaling(self, cscale = None, mass_key=None,  cscale_scale = "log", ymin_plt_lim = 3e12, ymax_plt_lim = 1e15, xmin_plt_lim = 0.1, xmax_plt_lim = 10,data_labels = True,  plot_self_similar_best_fit=False, plot_best_fit=True, external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed"}, best_fit_min_mass = 0,    legend_fontsize = 20 ):
        Ez_power = 1
        self_sim_scaling_power = 3/2
        
        x_property_key = "weighted_kT_Xray"
        y_property_key = "Xray_HSE_M500"
        xerr_key = "weighted_kT_Xray_spread"
        yerr_key = "Xray_HSE_M500_uncertainty"
        xlabel = r"kT$_\mathtt{X}$ [keV]"
        if mass_key == None: mass_key =  "Xray_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$\times \frac{\mathtt{M}_\mathtt{500,X}}{\mathtt{M}_\odot}$"
        if not hubble_correction:
            ylabel = r"$\frac{\mathtt{M500}_\mathtt{X}}{\mathtt{M}_\odot}$"
        if cscale == None:
            if hubble_correction:
                save_name = f"weighted_kT_Xray_Ez_corrected_flipped"
            else:
                save_name = f"weighted_kT_Xray_flipped"
        else:
            if hubble_correction:
                save_name = f"weighted_kT_Xray_Ez_corrected_vs_{cscale}_flipped"
            else:
                save_name = f"weighted_kT_Xray_vs_{cscale}_flipped"  
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )
        
    def plot_M500_vs_kTR500_Xray_scaling(self, cscale = None, mass_key=None,  cscale_scale = "log", ymin_plt_lim = 3e12, ymax_plt_lim = 1e15, xmin_plt_lim = 0.1, xmax_plt_lim = 10,data_labels = True,  plot_self_similar_best_fit=False, plot_best_fit=True, external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed"}, best_fit_min_mass = 0,    legend_fontsize = 20 ):
        Ez_power = 1
        self_sim_scaling_power = 3/2
        
        x_property_key = "kT_Xray_at_Xray_HSE_R500"
        y_property_key = "Xray_HSE_M500"
        xerr_key = "kT_Xray_at_Xray_HSE_R500_spread"
        yerr_key = "Xray_HSE_M500_uncertainty"
        xlabel = r"kT$_\mathtt{X}$($R_\mathtt{500,X}$) [keV]"
        if mass_key == None: mass_key =  "Xray_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$\times \frac{\mathtt{M}_\mathtt{500,X}}{\mathtt{M}_\odot}$"
        if not hubble_correction:
            ylabel = r"$\frac{\mathtt{M500}_\mathtt{X}}{\mathtt{M}_\odot}$"
        if cscale == None:
            if hubble_correction:
                save_name = f"kT_Xray_at_Xray_HSE_R500_Ez_corrected_flipped"
            else:
                save_name = f"kT_Xray_at_Xray_HSE_R500_flipped"
        else:
            if hubble_correction:
                save_name = f"kT_Xray_at_Xray_HSE_R500_Ez_corrected_vs_{cscale}_flipped"
            else:
                save_name = f"kT_Xray_at_Xray_HSE_R500_vs_{cscale}_flipped"  
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )

    def plot_M500_vs_kTR500_LW_scaling(self, cscale = None, mass_key=None,  cscale_scale = "log", ymin_plt_lim = 3e12, ymax_plt_lim = 1e15, xmin_plt_lim = 0.1, xmax_plt_lim = 10,data_labels = True,  plot_self_similar_best_fit=False, plot_best_fit=True, external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed"}, best_fit_min_mass = 0,    legend_fontsize = 20 ):
        Ez_power = 1
        self_sim_scaling_power = 3/2
        
        x_property_key = "kT_LW_at_LW_HSE_R500"
        y_property_key = "LW_HSE_M500"
        xerr_key = "kT_LW_at_LW_HSE_R500_spread"
        yerr_key = "LW_HSE_M500_uncertainty"
        xlabel = r"kT$_\mathtt{LW}$($R_\mathtt{500,LW}$) [keV]"
        if mass_key == None: mass_key =  "LW_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$\times \frac{\mathtt{M}_\mathtt{500,LW}}{\mathtt{M}_\odot}$"
        if not hubble_correction:
            ylabel = r"$\frac{\mathtt{M500}_\mathtt{LW}}{\mathtt{M}_\odot}$"
        if cscale == None:
            if hubble_correction:
                save_name = f"kT_LW_at_LW_HSE_R500_Ez_corrected_flipped"
            else:
                save_name = f"kT_LW_at_LW_HSE_R500_flipped"
        else:
            if hubble_correction:
                save_name = f"kT_LW_at_LW_HSE_R500_Ez_corrected_vs_{cscale}_flipped"
            else:
                save_name = f"kT_LW_at_LW_HSE_R500_vs_{cscale}_flipped"  
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )
        
        
        

    def plot_M500_vs_kTR500_EW_scaling(self, cscale = None, mass_key=None,  cscale_scale = "log", ymin_plt_lim = 3e12, ymax_plt_lim = 1e15, xmin_plt_lim = 0.1, xmax_plt_lim = 10,data_labels = True,  plot_self_similar_best_fit=False, plot_best_fit=True, external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed"}, best_fit_min_mass = 0,    legend_fontsize = 20 ):
        Ez_power = 1
        self_sim_scaling_power = 3/2
        
        x_property_key = "kT_EW_at_EW_HSE_R500"
        y_property_key = "EW_HSE_M500"
        xerr_key = "kT_EW_at_EW_HSE_R500_spread"
        yerr_key = "EW_HSE_M500_uncertainty"
        xlabel = r"kT$_\mathtt{EW}$($R_\mathtt{500,EW}$) [keV]"
        if mass_key == None: mass_key =  "EW_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$\times \frac{\mathtt{M}_\mathtt{500,EW}}{\mathtt{M}_\odot}$"
        if not hubble_correction:
            ylabel = r"$\frac{\mathtt{M500}_\mathtt{EW}}{\mathtt{M}_\odot}$"
        if cscale == None:
            if hubble_correction:
                save_name = f"kT_EW_at_EW_HSE_R500_Ez_corrected_flipped"
            else:
                save_name = f"kT_EW_at_EW_HSE_R500_flipped"
        else:
            if hubble_correction:
                save_name = f"kT_EW_at_EW_HSE_R500_Ez_corrected_vs_{cscale}_flipped"
            else:
                save_name = f"kT_EW_at_EW_HSE_R500_vs_{cscale}_flipped"  
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )
        
        
    def plot_M500_vs_kTR500_MW_scaling(self, cscale = None, mass_key=None,  cscale_scale = "log", ymin_plt_lim = 3e12, ymax_plt_lim = 1e15, xmin_plt_lim = 0.1, xmax_plt_lim = 10,data_labels = True,  plot_self_similar_best_fit=False, plot_best_fit=True, external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed"}, best_fit_min_mass = 0,    legend_fontsize = 20 ):
        Ez_power = 1
        self_sim_scaling_power = 3/2
        
        x_property_key = "kT_MW_at_MW_HSE_R500"
        y_property_key = "MW_HSE_M500"
        xerr_key = "kT_MW_at_MW_HSE_R500_spread"
        yerr_key = "MW_HSE_M500_uncertainty"
        xlabel = r"kT$_\mathtt{MW}$($R_\mathtt{500,MW}$) [keV]"
        if mass_key == None: mass_key =  "MW_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$\times \frac{\mathtt{M}_\mathtt{500,MW}}{\mathtt{M}_\odot}$"
        if not hubble_correction:
            ylabel = r"$\frac{\mathtt{M500}_\mathtt{MW}}{\mathtt{M}_\odot}$"
        if cscale == None:
            if hubble_correction:
                save_name = f"kT_MW_at_MW_HSE_R500_Ez_corrected_flipped"
            else:
                save_name = f"kT_MW_at_MW_HSE_R500_flipped"
        else:
            if hubble_correction:
                save_name = f"kT_MW_at_MW_HSE_R500_Ez_corrected_vs_{cscale}_flipped"
            else:
                save_name = f"kT_MW_at_MW_HSE_R500_vs_{cscale}_flipped"  
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )
        
        
        
        
    def plot_M500_vs_weighted_kT_MW_scaling(self, cscale = None, mass_key=None,  cscale_scale = "log", ymin_plt_lim = 3e12, ymax_plt_lim = 1e15, xmin_plt_lim = 0.1, xmax_plt_lim = 10,data_labels = True,  plot_self_similar_best_fit=False, plot_best_fit=True, external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed"}, best_fit_min_mass = 0,    legend_fontsize = 20 ):
        Ez_power = 1
        self_sim_scaling_power = 3/2
        
        x_property_key = "weighted_kT_MW"
        y_property_key = "MW_HSE_M500"
        xerr_key = "weighted_kT_MW_spread"
        yerr_key = "MW_HSE_M500_uncertainty"
        xlabel = r"kT$_\mathtt{MW}$ [keV]"
        if mass_key == None: mass_key =  "MW_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$\times \frac{\mathtt{M}_\mathtt{500,MW}}{\mathtt{M}_\odot}$"
        if not hubble_correction:
            ylabel = r"$\frac{\mathtt{M500}_\mathtt{MW}}{\mathtt{M}_\odot}$"
        if cscale == None:
            if hubble_correction:
                save_name = f"weighted_kT_MW_Ez_corrected_flipped"
            else:
                save_name = f"weighted_kT_MW_flipped"
        else:
            if hubble_correction:
                save_name = f"weighted_kT_MW_Ez_corrected_vs_{cscale}_flipped"
            else:
                save_name = f"weighted_kT_MW_vs_{cscale}_flipped"  
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )
    
    def plot_M500_vs_weighted_kT_LW_scaling(self, cscale = None, mass_key=None,  cscale_scale = "log", ymin_plt_lim = 3e12, ymax_plt_lim = 1e15, xmin_plt_lim = 0.1, xmax_plt_lim = 10,data_labels = True,  plot_self_similar_best_fit=False, plot_best_fit=True, external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed"}, best_fit_min_mass = 0,    legend_fontsize = 20 ):
        Ez_power = 1
        self_sim_scaling_power = 3/2
        
        x_property_key = "weighted_kT_LW"
        y_property_key = "LW_HSE_M500"
        xerr_key = "weighted_kT_LW_spread"
        yerr_key = "LW_HSE_M500_uncertainty"
        xlabel = r"kT$_\mathtt{LW}$ [keV]"
        if mass_key == None: mass_key =  "LW_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$\times \frac{\mathtt{M}_\mathtt{500,LW}}{\mathtt{M}_\odot}$"
        if not hubble_correction:
            ylabel = r"$\frac{\mathtt{M500}_\mathtt{LW}}{\mathtt{M}_\odot}$"
        if cscale == None:
            if hubble_correction:
                save_name = f"weighted_kT_LW_Ez_corrected_flipped"
            else:
                save_name = f"weighted_kT_LW_flipped"
        else:
            if hubble_correction:
                save_name = f"weighted_kT_LW_Ez_corrected_vs_{cscale}_flipped"
            else:
                save_name = f"weighted_kT_LW_vs_{cscale}_flipped"  
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )
  
    def plot_M500_vs_weighted_kT_EW_scaling(self, cscale = None, mass_key=None,  cscale_scale = "log", ymin_plt_lim = 3e12, ymax_plt_lim = 1e15, xmin_plt_lim = 0.1, xmax_plt_lim = 10,data_labels = True,  plot_self_similar_best_fit=False, plot_best_fit=True, external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed"}, best_fit_min_mass = 0,    legend_fontsize = 20 ):
        Ez_power = 1
        self_sim_scaling_power = 3/2
        
        x_property_key = "weighted_kT_EW"
        y_property_key = "EW_HSE_M500"
        xerr_key = "weighted_kT_EW_spread"
        yerr_key = "EW_HSE_M500_uncertainty"
        xlabel = r"kT$_\mathtt{EW}$ [keV]"
        if mass_key == None: mass_key =  "EW_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$\times \frac{\mathtt{M}_\mathtt{500,EW}}{\mathtt{M}_\odot}$"
        if not hubble_correction:
            ylabel = r"$\frac{\mathtt{M500}_\mathtt{EW}}{\mathtt{M}_\odot}$"
        if cscale == None:
            if hubble_correction:
                save_name = f"weighted_kT_EW_Ez_corrected_flipped"
            else:
                save_name = f"weighted_kT_EW_flipped"
        else:
            if hubble_correction:
                save_name = f"weighted_kT_EW_Ez_corrected_vs_{cscale}_flipped"
            else:
                save_name = f"weighted_kT_EW_vs_{cscale}_flipped"  
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )
        
        
    def plot_S_Xray_scaling(self, cscale = None, mass_key=None,  cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 600,  plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,    legend_fontsize = 20  ):
        Ez_power = 2/3
        self_sim_scaling_power = 2/3
        
        x_property_key = "Xray_HSE_M500"
        y_property_key = "weighted_S_Xray"
        xerr_key = "Xray_HSE_M500_uncertainty"
        yerr_key = "weighted_S_Xray_spread"
        xlabel = r"$\mathtt{M}_\mathtt{500,X}$  [$\mathtt{M}_\odot$]"
        if mass_key == None: mass_key =  "Xray_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{2/3}}$S$_\mathtt{X}$[keV cm$^2$]"
        if not hubble_correction:
            ylabel = r"$\mathtt{S}_\mathtt{X}$[keV/cm$^2$]"
        if cscale == None:
            if hubble_correction:
                save_name = f"weighted_S_Xray_Ez_corrected"
            else:
                save_name = f"weighted_S_Xray"
        else:
            if hubble_correction:
                save_name = f"weighted_S_Xray_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"weighted_S_Xray_vs_{cscale}"   
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )
    
    
    
    
    
    def plot_S_Xray_at_Xray_R500_scaling(self, cscale = None, mass_key=None,  cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 1000,  plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,    legend_fontsize = 20  ):
        Ez_power = 2/3
        self_sim_scaling_power = 2/3
        
        x_property_key = "Xray_HSE_M500"
        y_property_key = "S_Xray_at_Xray_HSE_R500"
        xerr_key = "Xray_HSE_M500_uncertainty"
        yerr_key = "S_Xray_at_Xray_HSE_R500_spread"
        xlabel = r"$\mathtt{M}_\mathtt{500,X}$  [$\mathtt{M}_\odot$]"
        if mass_key == None: mass_key =  "Xray_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{2/3}}$S$_\mathtt{X}(\mathtt{R}_\mathtt{500,X})$[keV cm$^2$]"
        if not hubble_correction:
            ylabel = r"$\mathtt{S}_\mathtt{X}(\mathtt{R}_\mathtt{500,X})$[keV/cm$^2$]"
        if cscale == None:
            if hubble_correction:
                save_name = f"S_Xray_at_Xray_R500_Ez_corrected"
            else:
                save_name = f"S_Xray_at_Xray_R500"
        else:
            if hubble_correction:
                save_name = f"S_Xray_at_Xray_R500_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"S_Xray_at_Xray_R500_vs_{cscale}"   
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )
        
    def plot_S_MW_at_MW_R500_scaling(self, cscale = None, mass_key=None,  cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 1000,  plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,    legend_fontsize = 20  ):
        Ez_power = 2/3
        self_sim_scaling_power = 2/3
        
        x_property_key = "MW_HSE_M500"
        y_property_key = "S_MW_at_MW_HSE_R500"
        xerr_key = "MW_HSE_M500_uncertainty"
        yerr_key = "S_MW_at_MW_HSE_R500_spread"
        xlabel = r"$\mathtt{M}_\mathtt{500,MW}$  [$\mathtt{M}_\odot$]"
        if mass_key == None: mass_key =  "MW_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{2/3}}$S$_\mathtt{MW}(\mathtt{R}_\mathtt{500,MW})$[keV cm$^2$]"
        if not hubble_correction:
            ylabel = r"$\mathtt{S}_\mathtt{MW}(\mathtt{R}_\mathtt{500,MW})$[keV/cm$^2$]"
        if cscale == None:
            if hubble_correction:
                save_name = f"S_MW_at_MW_R500_Ez_corrected"
            else:
                save_name = f"S_MW_at_MW_R500"
        else:
            if hubble_correction:
                save_name = f"S_MW_at_MW_R500_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"S_MW_at_MW_R500_vs_{cscale}"   
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )    
    
    
    
    def plot_Xray_Lx_scaling(self, emin, emax, cscale = None, mass_key=None,  cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 600,  plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,    legend_fontsize = 20  ):
        Ez_power = -7/3
        self_sim_scaling_power = 4/3
        
        Xray_total_lumin_emin_RF = 0.5
        Xray_total_lumin_emax_RF = 2.0
        
        
        x_property_key = "Xray_HSE_M500"
        y_property_key = f"Xray_L_in_Xray_HSE_R500_{Xray_total_lumin_emin_RF}-{Xray_total_lumin_emax_RF}_RF_keV"
        xerr_key = "Xray_HSE_M500_uncertainty"
        yerr_key = f"Xray_L_in_Xray_HSE_R500_spread_{Xray_total_lumin_emin_RF}-{Xray_total_lumin_emax_RF}_RF_keV"
        xlabel = r"$\mathtt{M}_\mathtt{500,X}$  [$\mathtt{M}_\odot$]"
        if mass_key == None: mass_key =  "Xray_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{-7/3}}$L$_\mathtt{X}$" + f"$_\mathrm{{({Xray_total_lumin_emin_RF}-{Xray_total_lumin_emax_RF})}}$[erg/s]"
        if not hubble_correction:
            ylabel = r"$L_\mathtt{X}$[erg/s]"
        if cscale == None:
            if hubble_correction:
                save_name = f"L_Xray_Ez_corrected"
            else:
                save_name = f"L_Xray"
        else:
            if hubble_correction:
                save_name = f"L_Xray_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"L_Xray_vs_{cscale}"   
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,    legend_fontsize = legend_fontsize )
        
        
    def plot_true_Lx_scaling(self,emin, emax, cscale = None, mass_key=None,  cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 600,   plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,    legend_fontsize = 20  ):
        Ez_power = -7/3
        self_sim_scaling_power = 4/3
        
        Xray_total_lumin_emin_RF = 0.5
        Xray_total_lumin_emax_RF = 2       
        
        
        x_property_key = "M500_truth"
        y_property_key = f"Lx_in_R500_truth_{Xray_total_lumin_emin_RF}_{Xray_total_lumin_emax_RF}_keV"
        xerr_key = None
        yerr_key = None
        xlabel = r"$\mathtt{M}_\mathtt{500,SO}$  [$\mathtt{M}_\odot$]"
        if mass_key == None: mass_key =  "M500_truth"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{-7/3}}$L$_\mathtt{truth}$" + f"$_\mathrm{{({Xray_total_lumin_emin_RF}-{Xray_total_lumin_emax_RF})}}$[erg/s]"
        if not hubble_correction:
            ylabel = r"$L_\mathtt{truth}$[erg/s]"
        if cscale == None:
            if hubble_correction:
                save_name = f"L_truth_Ez_corrected"
            else:
                save_name = f"L_truth"
        else:
            if hubble_correction:
                save_name = f"L_truth_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"L_truth_vs_{cscale}"   
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass, legend_fontsize = legend_fontsize)        
        
        
    
    
    
    def plot_Xray_Lx_vs_weighted_kT_Xray_scaling(self,emin,emax, cscale = None, mass_key=None,  cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 600,   plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,    legend_fontsize = 20  ):
        Ez_power = -1
        self_sim_scaling_power = 2
        Xray_total_lumin_emin_RF = 0.5
        Xray_total_lumin_emax_RF = 2.0
        
        
        y_property_key = f"Xray_L_in_Xray_HSE_R500_{Xray_total_lumin_emin_RF}-{Xray_total_lumin_emax_RF}_RF_keV"
        yerr_key = f"Xray_L_in_Xray_HSE_R500_spread_{Xray_total_lumin_emin_RF}-{Xray_total_lumin_emax_RF}_RF_keV"
        
        x_property_key = "weighted_kT_Xray"
        xerr_key = "weighted_kT_Xray_spread"
        xlabel = r"kT$_\mathtt{X}$ [keV]"
        if mass_key == None: mass_key =  "Xray_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{-7/3}}$L$_\mathtt{X}$" + f"$_\mathrm{{({Xray_total_lumin_emin_RF}-{Xray_total_lumin_emax_RF})}}$[erg/s]"
        if not hubble_correction:
            ylabel = r"$L_x$[erg/s]"
        if cscale == None:
            if hubble_correction:
                save_name = f"L_Xray_vs_weighted_kT_Xray_Ez_corrected"
            else:
                save_name = f"L_Xray_vs_weighted_kT_Xray"
        else:
            if hubble_correction:
                save_name = f"L_Xray_vs_weighted_kT_Xray_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"L_Xray_vs_weighted_kT_Xray_vs_{cscale}"   
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )
        
    def plot_Xray_Lx_vs_kTR500_Xray_scaling(self,emin,emax, cscale = None, mass_key=None,  cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 600,   plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,    legend_fontsize = 20  ):
        Ez_power = -1
        self_sim_scaling_power = 2
        x_property_key = "kT_Xray_at_Xray_HSE_R500"
        y_property_key = f"Xray_L_in_Xray_HSE_R500_{emin}-{emax}_RF_keV"
        xerr_key = "kT_Xray_at_Xray_HSE_R500_spread"
        yerr_key = f"Xray_L_in_Xray_HSE_R500_spread_{emin}-{emax}_RF_keV"
        xlabel = r"kT$_\mathtt{X}(R_{500,X})$ [keV]"
        mass_key = "Xray_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{-1}}$L$_\mathtt{X}$[erg/s]"
        if not hubble_correction:
            ylabel = r"$L_x$[erg/s]"
        if cscale == None:
            if hubble_correction:
                save_name = f"L_Xray_vs_kTR500_Xray_Ez_corrected"
            else:
                save_name = f"L_Xray_vs_kTR500_Xray"
        else:
            if hubble_correction:
                save_name = f"L_Xray_vs_kTR500_Xray_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"L_Xray_vs_kTR500_Xray_vs_{cscale}"   
        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,   legend_fontsize = legend_fontsize )
        
        
    def plot_Xray_Y_scaling(self, cscale = None, mass_key=None,  cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 600,  plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,    legend_fontsize = 20  ):
        Ez_power = -2/3
        self_sim_scaling_power = 5/3
        
        x_property_key = "Xray_HSE_M500"
        y_property_key = "Xray_Y"
        xerr_key = "Xray_HSE_M500_uncertainty"
        yerr_key = "Xray_Y_spread"
        xlabel = r"$\mathtt{M}_\mathtt{500,X}$  [$\mathtt{M}_\odot$ ]"
        mass_key = "Xray_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{-2/3}}$Y$_\mathtt{X}$[keV M$_\odot$/cm$^2$]"
        if not hubble_correction:
            ylabel = r"$S_x$[keV M$_\odot$]"
        if cscale == None:
            if hubble_correction:
                save_name = f"Xray_Y_Ez_corrected"
            else:
                save_name = f"Xray_Y"
        else:
            if hubble_correction:
                save_name = f"Xray_Y_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"Xray_Y_vs_{cscale}"  

        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,    legend_fontsize = legend_fontsize )
        
    def plot_EW_Y_scaling(self, cscale = None, mass_key=None,  cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 600,  plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,    legend_fontsize = 20  ):
        Ez_power = -2/3
        self_sim_scaling_power = 5/3
        
        x_property_key = "EW_HSE_M500"
        y_property_key = "EW_Y"
        xerr_key = "EW_HSE_M500_uncertainty"
        yerr_key = "EW_Y_spread"
        xlabel = r"$\mathtt{M}_\mathtt{500,EW}$  [$\mathtt{M}_\odot$ ]"
        mass_key = "EW_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{-2/3}}$Y$_\mathtt{EW}$[keV M$_\odot$/cm$^2$]"
        if not hubble_correction:
            ylabel = r"$S_x$[keV M$_\odot$]"
        if cscale == None:
            if hubble_correction:
                save_name = f"EW_Y_Ez_corrected"
            else:
                save_name = f"EW_Y"
        else:
            if hubble_correction:
                save_name = f"EW_Y_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"EW_Y_vs_{cscale}"  

        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,    legend_fontsize = legend_fontsize )
        
    def plot_MW_Y_scaling(self, cscale = None, mass_key=None,  cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 600,  plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,    legend_fontsize = 20  ):
        Ez_power = -2/3
        self_sim_scaling_power = 5/3
        
        x_property_key = "MW_HSE_M500"
        y_property_key = "MW_Y"
        xerr_key = "MW_HSE_M500_uncertainty"
        yerr_key = "MW_Y_spread"
        xlabel = r"$\mathtt{M}_\mathtt{500,MW}$  [$\mathtt{M}_\odot$ ]"
        mass_key = "MW_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{-2/3}}$Y$_\mathtt{MW}$[keV M$_\odot$/cm$^2$]"
        if not hubble_correction:
            ylabel = r"$S_x$[keV M$_\odot$]"
        if cscale == None:
            if hubble_correction:
                save_name = f"MW_Y_Ez_corrected"
            else:
                save_name = f"MW_Y"
        else:
            if hubble_correction:
                save_name = f"MW_Y_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"MW_Y_vs_{cscale}"  

        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,    legend_fontsize = legend_fontsize )   
        
    def plot_LW_Y_scaling(self, cscale = None, mass_key=None,  cscale_scale = "log", xmin_plt_lim = 3e12, xmax_plt_lim = 1e15, ymin_plt_lim = 100, ymax_plt_lim = 600,  plot_self_similar_best_fit=False, plot_best_fit=True, data_labels = True,  external_dsets = [], hubble_correction = True, best_fit_kwargs={"color":"grey", "lw":3, "ls":"dashed","label":"best fit"}, self_similar_best_fit_kwargs={"color":"blue", "lw":3, "ls":"dashed", "label":"Self-Similar best fit"}, best_fit_min_mass = 0,    legend_fontsize = 20  ):
        Ez_power = -2/3
        self_sim_scaling_power = 5/3
        
        x_property_key = "LW_HSE_M500"
        y_property_key = "LW_Y"
        xerr_key = "LW_HSE_M500_uncertainty"
        yerr_key = "LW_Y_spread"
        xlabel = r"$\mathtt{M}_\mathtt{500,LW}$  [$\mathtt{M}_\odot$ ]"
        mass_key = "LW_HSE_M500"
        if hubble_correction:
            ylabel = r"E(z)$^{\mathtt{-2/3}}$Y$_\mathtt{LW}$[keV M$_\odot$/cm$^2$]"
        if not hubble_correction:
            ylabel = r"$S_x$[keV M$_\odot$]"
        if cscale == None:
            if hubble_correction:
                save_name = f"LW_Y_Ez_corrected"
            else:
                save_name = f"LW_Y"
        else:
            if hubble_correction:
                save_name = f"LW_Y_Ez_corrected_vs_{cscale}"
            else:
                save_name = f"LW_Y_vs_{cscale}"  

        self.plot_loglog_scaling(x_property_key, y_property_key, xerr_key, yerr_key, mass_key, xlabel, ylabel, Ez_power, self_sim_scaling_power, xmin_plt_lim , xmax_plt_lim, ymin_plt_lim, ymax_plt_lim, save_name, cscale = cscale, cscale_scale = cscale_scale, data_labels = data_labels,  plot_self_similar_best_fit = plot_self_similar_best_fit, plot_best_fit=plot_best_fit, external_dsets = external_dsets, hubble_correction = hubble_correction, best_fit_kwargs=best_fit_kwargs, self_similar_best_fit_kwargs=self_similar_best_fit_kwargs, best_fit_min_mass = best_fit_min_mass,    legend_fontsize = legend_fontsize )
        
        
     

    def add_stacked_profiles(self, filter_field = None, filter_field_min = -np.inf, filter_field_max = np.inf, R500_min_frac = 0.1, R500_max_frac = 1, chip_rad_arcmin = None, plot_individuals = False, plot_medians = True, plot_percentiles = True, percentiles = (16,84), color = None, median_color = "black", label = None, percentiles_kwds = {}, R500_type = "Xray", fit_s_powerlaw = False, s_powerlaw_offset = 0, s_powerlaw_kwds = {}):
        
        self.set_plotstyle()
        
        self._stacked_R500_type = R500_type
        
        try:
            plt.figure(self.stacked_fig)
        except:
            print("Initialising stacked figure")
            figsize = (22,22)
            self.stacked_fig, self.stacked_axes = plt.subplots(2, 2,  figsize = figsize, facecolor='w',)  
            self._stacked_existing_labels = []
        if color == None:
            col = iter(cm.viridis(np.linspace(0, 1, len(self.sample_arr)+1)))
        else:
            c = color
            
            
        if plot_medians:
            kT_stack = []
            ne_stack = []
            s_stack = []
            mtot_stack = []
            
        
        
        num_plotted = 0
        for halo in self.sample_arr:

            if filter_field != None:
                try:
                    if halo[filter_field] <= filter_field_min or halo[filter_field] > filter_field_max:
                        # print(f"Value of {halo[filter_field]} rejected for min = {filter_field_min}, max = {filter_field_max}")
                        continue
                except: 
                    if halo[filter_field].value <= filter_field_min or halo[filter_field].value > filter_field_max:
                        # print(f"Value of {halo[filter_field]} rejected for min = {filter_field_min}, max = {filter_field_max}")
                        continue
            # print(f"Value of {halo[filter_field]} ACCEPTED for min = {filter_field_min}, max = {filter_field_max}")
            
            
            if color == None:
                c = next(col)

            kT_Xray_ml_yvals = halo["kT_Xray_ml_yvals"] * u.keV
            ne_Xray_ml_yvals = halo["ne_Xray_ml_yvals"] * u.cm**-3
            Xray_entropy_profile = halo["Xray_entropy_profile"]
            Xray_HSE_total_mass_profile = halo["Xray_HSE_total_mass_profile"]
            r_values_fine = halo["r_values_fine"]
            
            if R500_type == "Xray":
                R500 = halo["Xray_HSE_R500"]
                
            if R500_type == "truth":
                R500 = halo["R500_truth"]
                 
            
            R500_idx = np.abs(r_values_fine-R500).argmin()
            data_range_idxs = [(r_values_fine > R500_min_frac*R500) & (r_values_fine < R500_max_frac*R500) ][0]

        
            if plot_individuals:
                self.stacked_axes[0,0].plot(r_values_fine[data_range_idxs]/R500, kT_Xray_ml_yvals[data_range_idxs]/kT_Xray_ml_yvals[R500_idx], color = c,  zorder = 1)
                self.stacked_axes[0,0].plot(r_values_fine/R500, kT_Xray_ml_yvals/kT_Xray_ml_yvals[R500_idx], color = c,   ls = "dashed", zorder = 1)

                self.stacked_axes[0,1].plot(r_values_fine[data_range_idxs]/R500, ne_Xray_ml_yvals[data_range_idxs]/ne_Xray_ml_yvals[R500_idx], color = c,  zorder = 1)
                self.stacked_axes[0,1].plot(r_values_fine/R500, ne_Xray_ml_yvals/ne_Xray_ml_yvals[R500_idx], color = c, ls = "dashed", zorder = 1)

                self.stacked_axes[1,0].plot(r_values_fine[data_range_idxs]/R500,Xray_entropy_profile[data_range_idxs]/Xray_entropy_profile[R500_idx], color = c, zorder = 1 ) 
                self.stacked_axes[1,0].plot(r_values_fine/R500,Xray_entropy_profile/Xray_entropy_profile[R500_idx], color = c, ls = "dashed", zorder = 1) 

                self.stacked_axes[1,1].plot(r_values_fine[data_range_idxs]/R500,Xray_HSE_total_mass_profile[data_range_idxs]/Xray_HSE_total_mass_profile[R500_idx], color = c, zorder = 1 )
                self.stacked_axes[1,1].plot(r_values_fine/R500,Xray_HSE_total_mass_profile/Xray_HSE_total_mass_profile[R500_idx], color = c,   ls = "dashed", zorder = 1)

            
            ### If we want to stack, we need to remap the profiles onto a regular grid shared by all halo profiles between e.g. 0.1-1 R500
            if plot_medians:
                    x_remap = np.linspace(R500_min_frac, R500_max_frac, 100)

                    remapped_normed_kT   = np.interp(x = x_remap, xp = r_values_fine/R500 , fp = kT_Xray_ml_yvals/kT_Xray_ml_yvals[R500_idx],)
                    remapped_normed_ne   = np.interp(x = x_remap, xp = r_values_fine/R500 , fp = ne_Xray_ml_yvals/ne_Xray_ml_yvals[R500_idx],)
                    remapped_normed_s    = np.interp(x = x_remap, xp = r_values_fine/R500 , fp = Xray_entropy_profile/Xray_entropy_profile[R500_idx],)
                    remapped_normed_mtot = np.interp(x = x_remap, xp = r_values_fine/R500 , fp = Xray_HSE_total_mass_profile/Xray_HSE_total_mass_profile[R500_idx],)


                    if len(kT_stack) == 0: 
                        kT_stack = remapped_normed_kT
                        ne_stack = remapped_normed_ne 
                        s_stack = remapped_normed_s
                        mtot_stack = remapped_normed_mtot
                    else:
                        kT_stack = np.vstack((kT_stack,remapped_normed_kT))
                        ne_stack = np.vstack((ne_stack,remapped_normed_ne))
                        s_stack = np.vstack((s_stack,remapped_normed_s))
                        mtot_stack = np.vstack((mtot_stack,remapped_normed_mtot))
                    num_plotted += 1
                    # print("num_plotted incremented")

        if plot_medians:            
            kT_median = np.median(kT_stack, axis=0)
            ne_median = np.median(ne_stack, axis=0)
            s_median = np.median(s_stack, axis=0)
            mtot_median = np.median(mtot_stack, axis=0)
                
        if plot_medians and plot_percentiles:
            kT_percentiles = [np.percentile(kT_stack, q = percentiles[0], axis=0), np.percentile(kT_stack, q = percentiles[1], axis=0)]
            ne_percentiles = [np.percentile(ne_stack, q = percentiles[0], axis=0), np.percentile(ne_stack, q = percentiles[1], axis=0)]
            s_percentiles = [np.percentile(s_stack, q = percentiles[0], axis=0), np.percentile(s_stack, q = percentiles[1], axis=0)]
            mtot_percentiles = [np.percentile(mtot_stack, q = percentiles[0], axis=0), np.percentile(mtot_stack, q = percentiles[1], axis=0)]
                    
        if plot_medians:    
            med_lw = 5
            med_ls = "dashed"
            try:
                if label == None or label in self._stacked_existing_labels:
                    self.stacked_axes[0,0].plot(x_remap , kT_median,  lw = med_lw, ls = med_ls,  color = median_color, zorder=3000 )
                else:
                    self.stacked_axes[0,0].plot(x_remap , kT_median,  lw = med_lw, ls = med_ls,  color = median_color, zorder=3000, label = label )
                    self._stacked_existing_labels.append(label)
                self.stacked_axes[0,1].plot(x_remap , ne_median,  lw = med_lw, ls = med_ls, color = median_color, zorder=3000 )
                self.stacked_axes[1,0].plot(x_remap , s_median,  lw = med_lw, ls = med_ls, color = median_color, zorder=3000 ) 
                self.stacked_axes[1,1].plot(x_remap , mtot_median,  lw = med_lw, ls = med_ls, color = median_color, zorder=3000 )  

                if plot_percentiles:
                    perc_alpha = 0.3
                    self.stacked_axes[0,0].fill_between(x_remap , kT_percentiles[0], kT_percentiles[1], color = median_color, zorder=2000, **percentiles_kwds )
                    self.stacked_axes[0,1].fill_between(x_remap , ne_percentiles[0], ne_percentiles[1], color = median_color, zorder=2000, **percentiles_kwds )
                    self.stacked_axes[1,0].fill_between(x_remap , s_percentiles[0], s_percentiles[1],  color = median_color, zorder=2000, **percentiles_kwds ) 
                    self.stacked_axes[1,1].fill_between(x_remap , mtot_percentiles[0], mtot_percentiles[1], color = median_color, zorder=2000, **percentiles_kwds )  
            except:
                pass
            
            if fit_s_powerlaw:
                
                def s_powerlaw_slope(log_x,b):
                    m = 1.1
                    return m*log_x + b

                model = Model(s_powerlaw_slope, independent_vars=['log_x'])
                result = model.fit(np.log10(s_median), log_x=np.log10(x_remap),   b = np.log10(s_median[0]) - (1.1 * np.log10(x_remap[0])))
                b = result.values['b'] + s_powerlaw_offset
                print((10**b) * (np.array([R500_min_frac,R500_max_frac])**(1.1)))
                self.stacked_axes[1,0].plot([R500_min_frac,R500_max_frac],  (10**b) * (np.array([R500_min_frac,R500_max_frac])**(1.1)), **s_powerlaw_kwds)        
        
        print("num_plotted:", num_plotted)
                
                
  
        
        
        
    def save_stacked_profiles(self, save_tag = "" ):
        plt.figure(self.stacked_fig)
        self.set_plotstyle()
        R500_type = self._stacked_R500_type
        
              
                
        self.stacked_axes[1,1].set_xscale("log")
        self.stacked_axes[1,1].set_yscale("log")
        self.stacked_axes[1,1].set_xlim(left = 0.08, right = 1.2)
        self.stacked_axes[1,1].set_ylim(bottom = 3e-3)

        
        if R500_type == "Xray":
            self.stacked_axes[1,1].set_xlabel(r"$R/R500_X$")
            self.stacked_axes[1,1].set_ylabel(r"$M(<R)/M500_X$")
            self.stacked_axes[1,0].set_xlabel(r"$R/R500_X$")
            self.stacked_axes[1,0].set_ylabel(r"$K/K500_X$")
            self.stacked_axes[0,1].set_ylabel(r"$n_e/n_e500_X$")
            self.stacked_axes[0,0].set_ylabel(r"$kT/kT500_X$")
        if R500_type == "truth":
            self.stacked_axes[1,1].set_xlabel(r"$R/R500_{SphOv}$")
            self.stacked_axes[1,1].set_ylabel(r"$M(<R)/M500_{SphOv}$")
            self.stacked_axes[1,0].set_xlabel(r"$R/R500_{SphOv}$")
            self.stacked_axes[1,0].set_ylabel(r"$K/K500_{SphOv}$")
            self.stacked_axes[0,1].set_ylabel(r"$n_e/n_e500_{SphOv}$")
            self.stacked_axes[0,0].set_ylabel(r"$kT/kT500_{SphOv}$")
            
            
            
        self.stacked_axes[1,0].set_xscale("log")
        self.stacked_axes[1,0].set_yscale("log")
        self.stacked_axes[1,0].set_xlim(left = 0.08, right = 1.2)
        self.stacked_axes[1,0].set_box_aspect(1)            
            
            
        self.stacked_axes[0,0].set_yscale("log")
        self.stacked_axes[0,0].set_xscale("log")
        self.stacked_axes[0,0].set_xlim(left =0.08, right = 1.2)

           
        if len(self._stacked_existing_labels) != 0:
            self.stacked_axes[0,0].legend()
        self.stacked_axes[0,0].xaxis.set_ticks_position('top')
        self.stacked_axes[0,0].set_xlabel(r"")
        self.stacked_axes[0,0].set_xticklabels([])
        self.stacked_axes[0,0].set_box_aspect(1)            
            
        self.stacked_axes[0,1].set_yscale("log")
        self.stacked_axes[0,1].set_xscale("log")
        self.stacked_axes[0,1].set_xlim(left = 0.08, right = 1.2)

        self.stacked_axes[0,1].yaxis.set_label_position("right")
        self.stacked_axes[0,1].yaxis.set_ticks_position('right')
        self.stacked_axes[0,1].xaxis.set_ticks_position('top')
        self.stacked_axes[0,1].set_xlabel(r"")
        self.stacked_axes[0,1].set_xticklabels([])
        self.stacked_axes[0,1].set_box_aspect(1)            
            

        self.stacked_axes[1,1].yaxis.set_label_position("right")
        self.stacked_axes[1,1].yaxis.set_ticks_position('right')
        self.stacked_axes[1,1].set_box_aspect(1)
        
        
        plt.subplots_adjust(bottom=0.1, right=0.8, wspace=0, hspace=-0.25)
        self.stacked_axes[0,0].set_xticklabels([])
        self.stacked_axes[0,1].set_xticklabels([])

        self.stacked_axes[1,0].set_xticks([0.1, 0.5, 1])
        self.stacked_axes[1,0].set_xticklabels([r"0.1", r"0.5", r"1"])
        self.stacked_axes[1,1].set_xticks([0.1, 0.5, 1])
        self.stacked_axes[1,1].set_xticklabels([r"0.1", r"0.5", r"1"])        
        self.stacked_fig.patch.set_facecolor('white')
                
        plt.savefig(f"{self.save_dir}/{R500_type}_{save_tag}_stacked_profiles_4x4.png", bbox_inches = "tight")
        plt.clf()
        plt.close()        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    def add_stacked_K_profiles(self, filter_field = None, filter_field_min = -np.inf, filter_field_max = np.inf, R500_min_frac = 0.1, R500_max_frac = 1, chip_rad_arcmin = None, plot_individuals = False, plot_medians = True, plot_percentiles = True, percentiles = (16,84), color = None, median_color = "black", label = None, percentiles_kwds = {}, R500_type = "Xray", fit_s_powerlaw = False, s_powerlaw_offset = 0, s_powerlaw_kwds = {}, K500_type = "K(R500)", entropy_median_multiplier = 1, kT_bias_correct = None):
        
        self.set_plotstyle()
        
        self._stacked_K_R500_type = R500_type
        
        try:
            plt.figure(self.stacked_K_fig)
        except:
            print("Initialising stacked_K figure")
            figsize = (15,15)
            self.stacked_K_fig, self.stacked_K_axes = plt.subplots(1, 1,  figsize = figsize, facecolor='w',)  
            self._stacked_K_existing_labels = []
        if color == None:
            col = iter(cm.viridis(np.linspace(0, 1, len(self.sample_arr)+1)))
        else:
            c = color
            
            
        if plot_medians:
            kT_stack = []
            ne_stack = []
            s_stack = []
            mtot_stack = []
            
        
        
        num_plotted = 0
        for halo in self.sample_arr:

            if filter_field != None:
                try:
                    if halo[filter_field] < filter_field_min or halo[filter_field] > filter_field_max:
                        continue
                except: 
                    if halo[filter_field].value < filter_field_min or halo[filter_field].value > filter_field_max:
                        continue
            
            
            if color == None:
                c = next(col)


            Xray_entropy_profile = halo["Xray_entropy_profile"]
            r_values_fine = halo["r_values_fine"]
            
            if R500_type == "Xray":
                R500 = halo["Xray_HSE_R500"]
                M500 = halo["Xray_HSE_M500"]
                print(f"\nHalo {halo['idx']}: Using Xray Measured virial values:\n R500 = {R500}, \n M500 = {M500}")
                
            if R500_type == "truth":
                R500 = halo["R500_truth"]
                M500 = halo["M500_truth"]
                print(f"\nHalo {halo['idx']}: Using true virial values:\n R500 = {R500}, \n M500 = {M500}")
                
            if kT_bias_correct != None:
                print(f"Correcting M500, R500, amd S profile for a kT bias by {kT_bias_correct}")
                R500 *=kT_bias_correct**(1/3)
                M500 *=kT_bias_correct      
                Xray_entropy_profile = kT_bias_correct * np.array(Xray_entropy_profile)
                
                
                
                

                 
            
            R500_idx = np.abs(r_values_fine-R500).argmin()
            data_range_idxs = [(r_values_fine > R500_min_frac*R500) & (r_values_fine < R500_max_frac*R500) ][0]

            
            if K500_type == "K(R500)":
                K500 = Xray_entropy_profile[R500_idx]
                if R500_type == "Xray":
                    self.stacked_K_axes.set_ylabel(r"$K/K500_X$")
                if R500_type == "truth":
                    self.stacked_K_axes.set_ylabel(r"$K/K500_{SphOv}$")
                
            if K500_type == "oppenheimer":
                print(f"Halo {halo['idx']}: Using Oppenheimer norm with virial values:\n R500 = {R500}, \n M500 = {M500}")
                from astropy.constants import G
                m_H = 1.6735575 * 10**-27 * u.kg
                mu = 0.59
                mu_e = 1.14
                fb = 0.16
                
                try:
                    rho_crit = halo["rho_crit"]
                except:
                    print("rho crit not present in halo dict. Will use astropy rho_crit(z) instead")
                    rho_crit = self.cosmo.critical_density(z = self.redshift)

                kT_analytic = G * M500 * mu * m_H / R500
                ne_analytic = 500 * fb * rho_crit / (mu_e * m_H)

                K500 = kT_analytic * (ne_analytic**(-2/3))
                K500 = K500.to("keV cm^2").value
                print(f"Halo {halo['idx']}: Using Oppenheimer norm with K500 = {K500}")
                if R500_type == "Xray":
                    self.stacked_K_axes.set_ylabel(r"$K/K500_{ad,X}$")
                if R500_type == "truth":
                    self.stacked_K_axes.set_ylabel(r"$K/K500_{ad,SphOv}$")
                

                
                
        
            if plot_individuals:
                print(f"Halo {halo['idx']}: Normalising r by R500 = {R500} for individuals")
                self.stacked_K_axes.plot(r_values_fine[data_range_idxs]/R500,Xray_entropy_profile[data_range_idxs]/K500, color = c, zorder = 1 ) 
                self.stacked_K_axes.plot(r_values_fine/R500,Xray_entropy_profile/K500, color = c, ls = "dashed", zorder = 1) 

            
            ### If we want to stack, we need to remap the profiles onto a regular grid shared by all halo profiles between e.g. 0.1-1 R500
            if plot_medians:
                print(f"Halo {halo['idx']}: Normalising r by R500 = {R500} for medians")
                x_remap = np.linspace(R500_min_frac, R500_max_frac, 100)
                remapped_normed_s    = np.interp(x = x_remap, xp = r_values_fine/R500 , fp = Xray_entropy_profile/K500,)
                if len(s_stack) == 0: 
                    s_stack = remapped_normed_s
                else:
                    s_stack = np.vstack((s_stack,remapped_normed_s))
            num_plotted += 1

                    
                    
                    
                    
        if plot_medians:             
            s_median = np.median(s_stack, axis=0)
        if plot_percentiles:
            s_percentiles = [np.percentile(s_stack, q = percentiles[0], axis=0), np.percentile(s_stack, q = percentiles[1], axis=0)]

        if plot_medians:    
            med_lw = 5
            med_ls = "dashed"
            try:
                if label == None or label in self._stacked_K_existing_labels:
                    self.stacked_K_axes.plot(x_remap , entropy_median_multiplier*s_median,  lw = med_lw, ls = med_ls,  color = median_color, zorder=3000 )
                else:
                    self._stacked_K_existing_labels.append(label)
                    self.stacked_K_axes.plot(x_remap , entropy_median_multiplier*s_median,  lw = med_lw, ls = med_ls, color = median_color, zorder=3000, label = label ) 

                if plot_percentiles:
                    self.stacked_K_axes.fill_between(x_remap , s_percentiles[0], s_percentiles[1],  color = median_color, zorder=2000, **percentiles_kwds ) 
            except:
                pass
            
            if fit_s_powerlaw:
                
                def s_powerlaw_slope(log_x,b):
                    m = 1.1
                    return m*log_x + b

                model = Model(s_powerlaw_slope, independent_vars=['log_x'])
                result = model.fit(np.log10(s_median.value), log_x=np.log10(x_remap),   b = np.log10(s_median[0].value) - (1.1 * np.log10(x_remap[0])))
                b = result.values['b'] + s_powerlaw_offset
                print((10**b) * (np.array([R500_min_frac,R500_max_frac])**(1.1)))
                self.stacked_K_axes.plot([R500_min_frac,R500_max_frac],  (10**b) * (np.array([R500_min_frac,R500_max_frac])**(1.1)), **s_powerlaw_kwds)   
        print("num plotted:", num_plotted)
                

                
                
    def add_external_K_profiles(self,external_dsets):   
        if len(external_dsets) != 0:
            import csv
            for dset_dict in external_dsets:
                with open(f'external_data/{dset_dict["filename"]}.csv') as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    dset_x_data = []
                    dset_y_data = []
                    for row in csv_reader:
                        dset_x_data.append(float(row[0]))
                        dset_y_data.append(float(row[1]))
                print(dset_dict["label"])        
                if dset_dict.get("needs_exponentiating", False):
                    # print(dset_y_data)
                    dset_x_data = 10**np.array(dset_x_data)
                    dset_y_data = 10**np.array(dset_y_data)
                    # print(dset_y_data)
                if dset_dict.get("only_plot_ends", False):
                    dset_x_data = [dset_x_data[0], dset_x_data[-1]]
                    dset_y_data = [dset_y_data[0], dset_y_data[-1]]
                dset_x_data = np.array(dset_x_data)
                dset_y_data = np.array(dset_y_data)
                if dset_dict["plot type"] == "plot":
                    self.stacked_K_axes.plot(dset_dict.get("x_adjust", 1)*dset_x_data, dset_dict.get("y_adjust", 1)*dset_y_data, label = dset_dict["label"], **dset_dict["plot_kwds"] )
                if dset_dict["plot type"] == "scatter":
                    self.stacked_K_axes.scatter(dset_dict.get("x_adjust", 1)*dset_x_data, dset_dict.get("y_adjust", 1)*dset_y_data, label = dset_dict["label"], **dset_dict["plot_kwds"] )
    
        
        
        
        
        
  
        
        
        
    def save_stacked_K_profile(self, xlims = [0.08,1.2]):
        plt.figure(self.stacked_K_fig)
        self.set_plotstyle()
        R500_type = self._stacked_K_R500_type

        if R500_type == "Xray":

            self.stacked_K_axes.set_xlabel(r"$R/R500_X$")
            

        if R500_type == "truth":
            self.stacked_K_axes.set_xlabel(r"$R/R500_{SphOv}$")


        self.stacked_K_axes.set_xscale("log")
        self.stacked_K_axes.set_yscale("log")
        self.stacked_K_axes.set_xlim(xlims[0], xlims[1])
        self.stacked_K_axes.set_box_aspect(1)  

           
        if len(self._stacked_K_existing_labels) != 0:
            self.stacked_K_axes.legend(fontsize = 22)
            
        self.stacked_K_fig.tight_layout()

        self.stacked_K_axes.set_xticks([0.1, 0.5, 1])
        self.stacked_K_axes.set_xticklabels([r"0.1", r"0.5", r"1"])
        plt.yticks()
        self.stacked_K_fig.patch.set_facecolor('white')
            
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{R500_type}_stacked_K_profiles.png", bbox = "tight")
        plt.clf()
        plt.close()        
        
        
        
       
        
        
    def add_stacked_P_profiles(self, filter_field = None, filter_field_min = -np.inf, filter_field_max = np.inf, R500_min_frac = 0.1, R500_max_frac = 1, chip_rad_arcmin = None, plot_individuals = False, plot_medians = True, plot_percentiles = True, percentiles = (16,84), color = None, median_color = "black", label = None, percentiles_kwds = {}, R500_type = "Xray", fit_s_powerlaw = False, s_powerlaw_offset = 0, s_powerlaw_kwds = {}, P500_type = "P(R500)", entropy_median_multiplier = 1, kT_bias_correct = None, y_data = "Xray"):
        
        self.set_plotstyle()
        self.pressure_ydata_type = y_data
        self._stacked_P_R500_type = R500_type
        self._stacked_P_filter_field = filter_field
        
        try:
            plt.figure(self.stacked_P_fig)
        except:
            print("Initialising stacked_P figure")
            figsize = (15,15)
            self.stacked_P_fig, self.stacked_P_axes = plt.subplots(1, 1,  figsize = figsize, facecolor='w',)  
            self._stacked_P_existing_labels = []
        if color == None:
            col = iter(cm.viridis(np.linspace(0, 1, len(self.sample_arr)+1)))
        else:
            c = color
            
        if plot_medians:
            kT_stack = []
            ne_stack = []
            s_stack = []
            mtot_stack = []
            
        
        num_plotted = 0
        for halo in self.sample_arr:

            if filter_field != None:
                try:
                    if halo[filter_field] < filter_field_min or halo[filter_field] > filter_field_max:
                        continue
                except: 
                    if halo[filter_field].value < filter_field_min or halo[filter_field].value > filter_field_max:
                        continue
            
            
            if color == None:
                c = next(col)


            if y_data == "Xray":
                pressure_profile = halo["Xray_pressure_profile"]
            if y_data == "profile":
                pressure_profile = halo["EW_pressure_profile"]
                
            r_values_fine = halo["r_values_fine"]
            
            if R500_type == "Xray":
                R500 = halo["Xray_HSE_R500"]
                M500 = halo["Xray_HSE_M500"]
                print(f"\nHalo {halo['idx']}: Using Xray Measured virial values:\n R500 = {R500}, \n M500 = {M500}")
                
            if R500_type == "truth":
                R500 = halo["R500_truth"]
                M500 = halo["M500_truth"]
                print(f"\nHalo {halo['idx']}: Using true virial values:\n R500 = {R500}, \n M500 = {M500}")
                
            if kT_bias_correct != None:
                print(f"Correcting M500, R500, amd S profile for a kT bias by {kT_bias_correct}")
                R500 *=kT_bias_correct**(1/3)
                M500 *=kT_bias_correct      
                pressure_profile = kT_bias_correct * np.array(pressure_profile)
                
                
                
                

                 
            
            R500_idx = np.abs(r_values_fine-R500).argmin()
            data_range_idxs = [(r_values_fine > R500_min_frac*R500) & (r_values_fine < R500_max_frac*R500) ][0]

            
            if P500_type == "P(R500)":
                P500 = pressure_profile[R500_idx]
                if R500_type == "Xray":
                    self.stacked_P_axes.set_ylabel(r"$P/P500_X$")
                if R500_type == "truth":
                    self.stacked_P_axes.set_ylabel(r"$P/P500_{SphOv}$")
                
            if P500_type == "oppenheimer":
                print(f"Halo {halo['idx']}: Using Oppenheimer norm with virial values:\n R500 = {R500}, \n M500 = {M500}")
                from astropy.constants import G
                m_H = 1.6735575 * 10**-27 * u.kg
                mu = 0.59
                mu_e = 1.14
                fb = 0.16
                try:
                    rho_crit = halo["rho_crit"]
                except:
                    print("rho crit not present in halo dict. Will use astropy rho_crit(z) instead")
                    rho_crit = self.cosmo.critical_density(z = self.redshift)

                kT_analytic = G * M500 * mu * m_H / R500
                ne_analytic = 500 * fb * rho_crit / (mu_e * m_H)

                P500 = kT_analytic * ne_analytic
                P500 = P500.to("keV cm-3").value
                print(f"Halo {halo['idx']}: Using Oppenheimer norm with P500 = {P500}")
                if R500_type == "Xray":
                    self.stacked_P_axes.set_ylabel(r"$P/P500_{ad,X}$")
                if R500_type == "truth":
                    self.stacked_P_axes.set_ylabel(r"$P/P500_{ad,SphOv}$")
                  
                
        
            if plot_individuals:
                print(f"Halo {halo['idx']}: Normalising r by R500 = {R500} for individuals")
                self.stacked_P_axes.plot(r_values_fine[data_range_idxs]/R500,pressure_profile[data_range_idxs]/P500, color = c, zorder = 1 ) 
                self.stacked_P_axes.plot(r_values_fine/R500,pressure_profile/P500, color = c, ls = "dashed", zorder = 1) 

            
            ### If we want to stack, we need to remap the profiles onto a regular grid shared by all halo profiles between e.g. 0.1-1 R500
            if plot_medians:
                print(f"Halo {halo['idx']}: Normalising r by R500 = {R500} for medians")
                x_remap = np.linspace(R500_min_frac, R500_max_frac, 100)
                remapped_normed_s    = np.interp(x = x_remap, xp = r_values_fine/R500 , fp = pressure_profile/P500,)
                if len(s_stack) == 0: 
                    s_stack = remapped_normed_s
                else:
                    s_stack = np.vstack((s_stack,remapped_normed_s))
            num_plotted += 1

                    
                    
                    
                    
        if plot_medians:             
            s_median = np.median(s_stack, axis=0)
        if plot_percentiles:
            s_percentiles = [np.percentile(s_stack, q = percentiles[0], axis=0), np.percentile(s_stack, q = percentiles[1], axis=0)]

        if plot_medians:    
            med_lw = 5
            med_ls = "dashed"
            try:
                if label == None or label in self._stacked_P_existing_labels:
                    self.stacked_P_axes.plot(x_remap , entropy_median_multiplier*s_median,  lw = med_lw, ls = med_ls,  color = median_color, zorder=3000 )
                else:
                    self._stacked_P_existing_labels.append(label)
                    self.stacked_P_axes.plot(x_remap , entropy_median_multiplier*s_median,  lw = med_lw, ls = med_ls, color = median_color, zorder=3000, label = label ) 

                if plot_percentiles:
                    self.stacked_P_axes.fill_between(x_remap , s_percentiles[0], s_percentiles[1],  color = median_color, zorder=2000, **percentiles_kwds ) 
            except:
                pass
            
            if fit_s_powerlaw:
                
                def s_powerlaw_slope(log_x,b):
                    m = 1.1
                    return m*log_x + b

                model = Model(s_powerlaw_slope, independent_vars=['log_x'])
                result = model.fit(np.log10(s_median.value), log_x=np.log10(x_remap),   b = np.log10(s_median[0].value) - (1.1 * np.log10(x_remap[0])))
                b = result.values['b'] + s_powerlaw_offset
                print((10**b) * (np.array([R500_min_frac,R500_max_frac])**(1.1)))
                self.stacked_P_axes.plot([R500_min_frac,R500_max_frac],  (10**b) * (np.array([R500_min_frac,R500_max_frac])**(1.1)), **s_powerlaw_kwds)   
        print("num plotted:", num_plotted)
                

                
                
    def add_external_P_profiles(self,external_dsets):   
        if len(external_dsets) != 0:
            import csv
            for dset_dict in external_dsets:
                with open(f'external_data/{dset_dict["filename"]}.csv') as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    dset_x_data = []
                    dset_y_data = []
                    for row in csv_reader:
                        dset_x_data.append(float(row[0]))
                        dset_y_data.append(float(row[1]))
                print(dset_dict["label"])        
                if dset_dict.get("needs_exponentiating", False):
                    # print(dset_y_data)
                    dset_x_data = 10**np.array(dset_x_data)
                    dset_y_data = 10**np.array(dset_y_data)
                    # print(dset_y_data)
                if dset_dict.get("only_plot_ends", False):
                    dset_x_data = [dset_x_data[0], dset_x_data[-1]]
                    dset_y_data = [dset_y_data[0], dset_y_data[-1]]
                dset_x_data = np.array(dset_x_data)
                dset_y_data = np.array(dset_y_data)
                if dset_dict["plot type"] == "plot":
                    self.stacked_P_axes.plot(dset_dict.get("x_adjust", 1)*dset_x_data, dset_dict.get("y_adjust", 1)*dset_y_data, label = dset_dict["label"], **dset_dict["plot_kwds"] )
                if dset_dict["plot type"] == "scatter":
                    self.stacked_P_axes.scatter(dset_dict.get("x_adjust", 1)*dset_x_data, dset_dict.get("y_adjust", 1)*dset_y_data, label = dset_dict["label"], **dset_dict["plot_kwds"] )
    
        
        
        
        
        
  
        
        
        
    def save_stacked_P_profile(self, xlims = [0.08,1.2]):
        plt.figure(self.stacked_P_fig)
        self.set_plotstyle()
        R500_type = self._stacked_P_R500_type

        if R500_type == "Xray":

            self.stacked_P_axes.set_xlabel(r"$R/R500_X$")
            

        if R500_type == "truth":
            self.stacked_P_axes.set_xlabel(r"$R/R500_{SphOv}$")


        self.stacked_P_axes.set_xscale("log")
        self.stacked_P_axes.set_yscale("log")
        self.stacked_P_axes.set_xlim(xlims[0], xlims[1])
        self.stacked_P_axes.set_box_aspect(1)  

           
        if len(self._stacked_P_existing_labels) != 0:
            self.stacked_P_axes.legend(fontsize = 22)
            
        self.stacked_P_fig.tight_layout()

        self.stacked_P_axes.set_xticks([0.1, 0.5, 1])
        self.stacked_P_axes.set_xticklabels([r"0.1", r"0.5", r"1"])
        plt.yticks()
        self.stacked_P_fig.patch.set_facecolor('white')
            
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/stacked_P_profiles.virial_type={R500_type}.ydata_type={self.pressure_ydata_type}.filter_field={self._stacked_P_filter_field}.png", bbox = "tight")
        plt.clf()
        plt.close()        
        
        
        
    def bias_plotting_4x4(self, plot_running_median = True, error_alpha = 0.5, scatter_linewidth = 0.9, median_kernel_size = 5, median_alpha = 0.5, median_lims = (0,-1), plot_running_percentiles = (16,84), xmin_plt_lim = 3e12,xmax_plt_lim = 9.9e14, plot_legend = False, bias_lines_alpha = 0.5, plot_ODR_median = False, bad_ODR_idxs = []):
        

        fig1 = plt.figure(figsize = (10,10), facecolor = 'w')
        frame1 = fig1.add_axes((.1,1.1,.8,.6))
        plt.xlabel(r"$\frac{\mathtt{M}_\mathtt{500,SO}}{\mathtt{M}_\odot}$", labelpad = 3)
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,MW}}{\mathtt{M}_\odot}$")
        frame2 = fig1.add_axes((.1,0.9,.8,.2))
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,MW}}{\mathtt{M}_\mathtt{500,SO}}$")
        plt.sca(frame1)
        for halo in self.sample_arr:
            plt.sca(frame1)
            plt.errorbar(halo["M500_truth"].value,halo["MW_HSE_M500"].value, yerr = np.reshape(halo["MW_HSE_M500_uncertainty"].value, (2,1)) , capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
            plt.scatter(halo["M500_truth"],halo["MW_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100)
            plt.sca(frame2)
            plt.scatter(halo["M500_truth"],halo["MW_HSE_M500"]/halo["M500_truth"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100)
            plt.errorbar(halo["M500_truth"].value,halo["MW_HSE_M500"].value/halo["M500_truth"].value, yerr = np.reshape(halo["MW_HSE_M500_uncertainty"].value, (2,1)) /halo["M500_truth"].value, capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
        if plot_running_median:
            plt.sca(frame2)
            x_data = np.array([halo["M500_truth"].value for halo in self.sample_arr])
            y_data = np.array([halo["MW_HSE_M500"].value for halo in self.sample_arr])
            print(f"\nMax MW bias: {max(1-np.array(y_data/x_data))}")
            print(f"Min MW bias: {min(1-np.array(y_data/x_data))}")

            sort_idxs = np.argsort(x_data)
            x_data = x_data[sort_idxs]
            y_data = y_data[sort_idxs]
            running_median = median_filter(y_data/x_data, size = median_kernel_size)
            running_perc1 = percentile_filter(y_data/x_data,percentile = min(plot_running_percentiles), size = median_kernel_size)
            running_perc2 = percentile_filter(y_data/x_data,percentile = max(plot_running_percentiles), size = median_kernel_size)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)  
            plt.sca(frame1)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median*x_data)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)    
            if plot_running_percentiles != False:
                    plt.sca(frame2)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = running_perc1[median_lims[0]:median_lims[1]],y2 = running_perc2[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)   
                    plt.sca(frame1)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = (running_perc1*x_data)[median_lims[0]:median_lims[1]],y2 = (running_perc2*x_data)[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)  


        frame3 = fig1.add_axes((.9,1.1,.8,.6))
        plt.xlabel(r"$\frac{\mathtt{M}_\mathtt{500,SO}}{\mathtt{M}_\odot}$", labelpad = 3)
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,EW}}{\mathtt{M}_\odot}$")
        frame4 = fig1.add_axes((.9,0.9,.8,.2))
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,EW}}{\mathtt{M}_\mathtt{500,SO}}$")
        plt.sca(frame3)
        for halo in self.sample_arr:
            plt.sca(frame3)
            plt.errorbar(halo["M500_truth"].value,halo["EW_HSE_M500"].value, yerr = np.reshape(halo["EW_HSE_M500_uncertainty"].value, (2,1)) , capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
            plt.scatter(halo["M500_truth"],halo["EW_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100)
            plt.sca(frame4)
            plt.scatter(halo["M500_truth"],halo["EW_HSE_M500"]/halo["M500_truth"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100)
            plt.errorbar(halo["M500_truth"].value,halo["EW_HSE_M500"].value/halo["M500_truth"].value, yerr = np.reshape(halo["EW_HSE_M500_uncertainty"].value, (2,1)) /halo["M500_truth"].value, capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
        
        if plot_running_median:
            plt.sca(frame4)
            x_data = np.array([halo["M500_truth"].value for halo in self.sample_arr])
            y_data = np.array([halo["EW_HSE_M500"].value for halo in self.sample_arr])
            print(f"\nMax EW bias: {max(1-np.array(y_data/x_data))}")
            print(f"Min EW bias: {min(1-np.array(y_data/x_data))}")


            sort_idxs = np.argsort(x_data)
            x_data = x_data[sort_idxs]
            y_data = y_data[sort_idxs]
            running_median = median_filter(y_data/x_data, size = median_kernel_size)
            running_perc1 = percentile_filter(y_data/x_data,percentile = min(plot_running_percentiles), size = median_kernel_size)
            running_perc2 = percentile_filter(y_data/x_data,percentile = max(plot_running_percentiles), size = median_kernel_size)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)  
            plt.sca(frame3)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median*x_data)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)    
            if plot_running_percentiles != False:
                    plt.sca(frame4)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = running_perc1[median_lims[0]:median_lims[1]],y2 = running_perc2[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)   
                    plt.sca(frame3)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = (running_perc1*x_data)[median_lims[0]:median_lims[1]],y2 = (running_perc2*x_data)[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)  
                    
        frame5 = fig1.add_axes((.1,.3,.8,.6))
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,LW}}{\mathtt{M}_\odot}$")
        frame6 = fig1.add_axes((.1,.1,.8,.2))
        plt.xlabel(r"$\frac{\mathtt{M}_\mathtt{500,SO}}{\mathtt{M}_\odot}$")
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,LW}}{\mathtt{M}_\mathtt{500,SO}}$")
        plt.sca(frame5)
        for halo in self.sample_arr:
            plt.sca(frame5)
            plt.errorbar(halo["M500_truth"].value,halo["LW_HSE_M500"].value, yerr = np.reshape(halo["LW_HSE_M500_uncertainty"].value, (2,1)) , capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
            plt.scatter(halo["M500_truth"],halo["LW_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100)
            plt.sca(frame6)
            plt.scatter(halo["M500_truth"],halo["LW_HSE_M500"]/halo["M500_truth"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100)
            plt.errorbar(halo["M500_truth"].value,halo["LW_HSE_M500"].value/halo["M500_truth"].value, yerr = np.reshape(halo["LW_HSE_M500_uncertainty"].value, (2,1)) /halo["M500_truth"].value, capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)

        if plot_running_median:
            plt.sca(frame6)
            x_data = np.array([halo["M500_truth"].value for halo in self.sample_arr])
            y_data = np.array([halo["LW_HSE_M500"].value for halo in self.sample_arr])
            print(f"\nMax LW bias: {max(1-np.array(y_data/x_data))}")
            print(f"Min LW bias: {min(1-np.array(y_data/x_data))}")

            sort_idxs = np.argsort(x_data)
            x_data = x_data[sort_idxs]
            y_data = y_data[sort_idxs]
            running_median = median_filter(y_data/x_data, size = median_kernel_size)
            running_perc1 = percentile_filter(y_data/x_data,percentile = min(plot_running_percentiles), size = median_kernel_size)
            running_perc2 = percentile_filter(y_data/x_data,percentile = max(plot_running_percentiles), size = median_kernel_size)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)  
            plt.sca(frame5)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median*x_data)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)    
            if plot_running_percentiles != False:
                    plt.sca(frame6)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = running_perc1[median_lims[0]:median_lims[1]],y2 = running_perc2[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)   
                    plt.sca(frame5)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = (running_perc1*x_data)[median_lims[0]:median_lims[1]],y2 = (running_perc2*x_data)[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)  


        frame7 = fig1.add_axes((.9,.3,.8,.6))
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,Xray}}{\mathtt{M}_\odot}$")
        frame8 = fig1.add_axes((.9,.1,.8,.2))
        plt.xlabel(r"$\frac{\mathtt{M}_\mathtt{500,SO}}{\mathtt{M}_\odot}$")
        plt.ylabel(r"$\frac{\mathtt{M}_\mathtt{500,Xray}}{\mathtt{M}_\mathtt{500,SO}}$")
        plt.sca(frame7)
        for halo in self.sample_arr:
            plt.sca(frame7)
            plt.errorbar(halo["M500_truth"].value,halo["Xray_HSE_M500"].value, yerr = np.reshape(halo["Xray_HSE_M500_uncertainty"].value, (2,1)) , capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
            plt.scatter(halo["M500_truth"],halo["Xray_HSE_M500"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100)
            plt.sca(frame8)
            plt.scatter(halo["M500_truth"],halo["Xray_HSE_M500"]/halo["M500_truth"], facecolors='none', edgecolors=halo["marker color"], marker = "^", s = 100)
            plt.errorbar(halo["M500_truth"].value,halo["Xray_HSE_M500"].value/halo["M500_truth"].value, yerr = np.reshape(halo["Xray_HSE_M500_uncertainty"].value, (2,1)) /halo["M500_truth"].value, capsize = 5, fmt = 'None', color = halo["marker color"], elinewidth = 0.3)
        
        if plot_running_median:
            plt.sca(frame8)
            x_data = np.array([halo["M500_truth"].value for halo in self.sample_arr])
            y_data = np.array([halo["Xray_HSE_M500"].value for halo in self.sample_arr])
            print(f"\nMax Xray bias: {max(1-np.array(y_data/x_data))}")
            print(f"Min Xray bias: {min(1-np.array(y_data/x_data))}")


            sort_idxs = np.argsort(x_data)
            x_data = x_data[sort_idxs]
            y_data = y_data[sort_idxs]
            running_median = median_filter(y_data/x_data, size = median_kernel_size)
            running_perc1 = percentile_filter(y_data/x_data,percentile = min(plot_running_percentiles), size = median_kernel_size)
            running_perc2 = percentile_filter(y_data/x_data,percentile = max(plot_running_percentiles), size = median_kernel_size)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)  
            plt.sca(frame7)
            plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median*x_data)[median_lims[0]:median_lims[1]], color = "black", lw = 3, ls = "solid", alpha = median_alpha)    
            if plot_running_percentiles != False:
                    plt.sca(frame8)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = running_perc1[median_lims[0]:median_lims[1]],y2 = running_perc2[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)   
                    plt.sca(frame7)
                    plt.fill_between(x_data[median_lims[0]:median_lims[1]], y1 = (running_perc1*x_data)[median_lims[0]:median_lims[1]],y2 = (running_perc2*x_data)[median_lims[0]:median_lims[1]], color = "grey", alpha = 0.5*median_alpha)  
            if plot_ODR_median:
                plt.sca(frame8)
                x_data = np.array([halo["M500_truth"].value for halo in self.sample_arr if int(halo["idx"]) not in bad_ODR_idxs])
                y_data = np.array([halo["Xray_ODR_HSE_M500"].value for halo in self.sample_arr if int(halo["idx"]) not in bad_ODR_idxs])
                sort_idxs = np.argsort(x_data)
                x_data = x_data[sort_idxs]
                y_data = y_data[sort_idxs]
                running_median = median_filter(y_data/x_data, size = median_kernel_size)
                running_perc1 = percentile_filter(y_data/x_data,percentile = min(plot_running_percentiles), size = median_kernel_size)
                running_perc2 = percentile_filter(y_data/x_data,percentile = max(plot_running_percentiles), size = median_kernel_size)
                plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median)[median_lims[0]:median_lims[1]], color = "red", lw = 3, ls = "dashed", alpha = median_alpha)  
                plt.sca(frame7)
                plt.plot(x_data[median_lims[0]:median_lims[1]], (running_median*x_data)[median_lims[0]:median_lims[1]], color = "red", lw = 3, ls = "dashed", alpha = median_alpha)    

        frame1.xaxis.set_ticks_position('top')
        frame1.xaxis.set_label_position('top')        
        frame2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        frame3.xaxis.set_ticks_position('top')
        frame3.yaxis.set_ticks_position('right')
        frame3.xaxis.set_label_position('top')
        frame3.yaxis.set_ticks_position('right')
        frame3.xaxis.set_label_position('top')
        frame3.yaxis.set_label_position('right')
        frame4.yaxis.set_ticks_position('right')
        frame4.yaxis.set_label_position('right')
        frame4.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        frame5.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        frame6.xaxis.set_label_position('bottom')
        frame7.yaxis.set_ticks_position('right')
        frame7.yaxis.set_label_position('right')
        frame7.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        frame8.yaxis.set_ticks_position('right')
        frame8.yaxis.set_label_position('right')
        frame8.xaxis.set_label_position('bottom')
        
        
        
        
        for frame in [frame2, frame4, frame6, frame8]:
            plt.sca(frame)
            plt.hlines(y = 1, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "green", lw = 2, ls = "solid", alpha = bias_lines_alpha)
            plt.hlines(y = 0.9, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "orange", lw = 2, ls = "solid", alpha = bias_lines_alpha)
            plt.hlines(y = 0.7, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "sienna", lw = 2, ls = "solid", alpha = bias_lines_alpha)
            plt.hlines(y = 0.4, xmin = xmin_plt_lim, xmax = xmax_plt_lim, color = "teal", lw = 2, ls = "solid", alpha = bias_lines_alpha)
            plt.ylim(bottom = 0,top = 2.5)       
            plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
            plt.xscale("log")
            plt.yscale("linear")
            for label in (frame.get_xticklabels() + frame.get_yticklabels()):
                label.set_fontname('monospace')
            ml = MultipleLocator(0.1)
            frame.yaxis.set_minor_locator(ml)
            ml = MultipleLocator(1)
            frame.yaxis.set_major_locator(ml)
            
        for frame in [frame1, frame3, frame5, frame7]:
            plt.sca(frame)
            plt.xlim(left = xmin_plt_lim, right = xmax_plt_lim)
            plt.ylim(bottom = xmin_plt_lim, top = xmax_plt_lim)
            plt.yscale("log")
            plt.xscale('log')
            for label in (frame.get_xticklabels() + frame.get_yticklabels()):
                label.set_fontname('monospace')
            plt.plot(10**np.arange(10,20,step = 0.5), 10**np.arange(10,20,step = 0.5),  color = "green", label = "b =  0", lw = 2, ls = "solid", alpha = bias_lines_alpha)
            plt.plot(10**np.arange(10,20,step = 0.5), 0.9 * 10**np.arange(10,20,step = 0.5),  color = "orange", label = "b =  0.1", lw = 2, ls = "solid", alpha = bias_lines_alpha)
            plt.plot(10**np.arange(10,20,step = 0.5), 0.7 * 10**np.arange(10,20,step = 0.5),  color = "sienna", label = "b =  0.3", lw = 2, ls = "solid", alpha = bias_lines_alpha)
            plt.plot(10**np.arange(10,20,step = 0.5), 0.4 * 10**np.arange(10,20,step = 0.5),  color = "teal", label = "b =  0.6", lw = 2, ls = "solid", alpha = bias_lines_alpha)
        print(f"Saving at {self.save_dir}/mass_bias_panel_4x4.png")    
        plt.savefig(f"{self.save_dir}/mass_bias_panel_4x4.png", bbox_inches='tight')
        plt.clf()
        plt.close()      
        
        
        
        
    def calculate_2D_ks_test_for_bias(self, key1, key2, bias_offset = 0, min_mass = 0, max_mass = 1e30, mass_key = None, ks_kwargs = {}, min_num = 10):
        # print("min/max mass", min_mass, max_mass)
        if mass_key == None:
            mass_key = key1
        X1 = np.array([ 1 -(halo[key1].value/halo["M500_truth"].value) for halo in self.sample_arr if key1 in halo.keys() and key2 in halo.keys() and halo[mass_key].value < max_mass and halo[mass_key].value >= min_mass])
        if mass_key == None:
            mass_key = key2
        X2 = np.array([ 1 -(halo[key2].value/halo["M500_truth"].value) for halo in self.sample_arr if key1 in halo.keys() and key2 in halo.keys() and halo[mass_key].value < max_mass and halo[mass_key].value >= min_mass])
        if len(X1) < min_num or len(X2) < min_num:
            return None
        if len(bias_offset + X1) != len(X1):
            print("Error!")
            return
        return scipy.stats.ks_2samp(bias_offset + X1, X2, **ks_kwargs)
    
    def generate_2D_ks_test_table(self, keys,  min_mass = 0, significance = 0.05, max_mass = 1e30, mass_key = None, ks_kwargs = {}, min_num = 10):
        # print("min/max mass", min_mass, max_mass)
        bias_offset = 0
        table_string = r"\begin{tabular}{||c| " + "c"*len(keys) +  r"||}" + "\n" + r"\hline" + "\n"
        table_string += " & " + " & ".join([key.split('_')[0] for key in list(reversed(keys))]) + r"\\ \hline\hline" + "\n"
        for key1 in keys:
            table_string += key1.split('_')[0] + " & "
            for key2 in list(reversed(keys)):
                if mass_key == None:
                    mass_key = key1
                X1 = np.array([ 1 -(halo[key1].value/halo["M500_truth"].value) for halo in self.sample_arr if key1 in halo.keys() and key2 in halo.keys() and halo[mass_key].value < max_mass and halo[mass_key].value >= min_mass])
                if mass_key == None:
                    mass_key = key2
                X2 = np.array([ 1 -(halo[key2].value/halo["M500_truth"].value) for halo in self.sample_arr if key1 in halo.keys() and key2 in halo.keys() and halo[mass_key].value < max_mass and halo[mass_key].value >= min_mass])
                if len(X1) < min_num or len(X2) < min_num:
                    return None
                if len(bias_offset + X1) != len(X1):
                    print("Error!")
                    return
                ks_stats = scipy.stats.ks_2samp(bias_offset + X1, X2, **ks_kwargs)
                # print(key1,key2, round(ks_stats[1],3))
                if key1 == key2 or keys.index(key1) > keys.index(key2):
                    table_string += "- & "
                else:
                    if ks_stats[1] > significance:
                        table_string += "\cellcolor{green} "
                    table_string += f"({round(ks_stats[0],3)}, {'{:.2e}'.format(ks_stats[1])})" + " & "
            table_string = table_string[:-2] ## Remove trailing ampersand
            table_string += r"\\" + "\n"
        table_string += r" \hline \hline" + "\n" + "\end{tabular}"
        print(table_string)
        
        
        
    def generate_2D_ks_test_table_all_masses(self, keys,   significance = 0.05, ks_kwargs = {}, ):
        # print("min/max mass", min_mass, max_mass)
        table_string = r"\begin{tabular}{||c| " + "c"*len(keys) +  r"||}" + "\n" + r"\hline" + "\n"
        table_string += " & " + " & ".join([key.split('_')[0] for key in list(reversed(keys))]) + r"\\ \hline\hline" + "\n"
        for key1 in keys:
            table_string += key1.split('_')[0] + " & "
            for key2 in list(reversed(keys)):
                X1 = np.array([ 1 -(halo[key1].value/halo["M500_truth"].value) for halo in self.sample_arr if key1 in halo.keys() and key2 in halo.keys() ])
                X2 = np.array([ 1 -(halo[key2].value/halo["M500_truth"].value) for halo in self.sample_arr if key1 in halo.keys() and key2 in halo.keys() ])

                ks_stats = scipy.stats.ks_2samp(X1, X2, **ks_kwargs)
                # print(key1,key2, round(ks_stats[1],3))
                if key1 == key2 or keys.index(key1) > keys.index(key2):
                    table_string += "- & "
                else:
                    if ks_stats[1] > significance:
                        table_string += "\cellcolor{green} "
                    table_string += f"({round(ks_stats[0],3)}, {'{:.2e}'.format(ks_stats[1])})" + " & "
            table_string = table_string[:-2] ## Remove trailing ampersand
            table_string += r"\\" + "\n"
        table_string += r" \hline \hline" + "\n" + "\end{tabular}"
        print(table_string)

    
    
    
    def plot_2D_ks_shifting_test_panel(self,key_pairs, min_delta_b = -0.2, max_delta_b = 0.5, significance = 0.05,  min_mass = 0, max_mass = 1e30, ylims = (-5,10), xlims = None, mass_key = None, min_num = 10, plot_max = False, plot_significance_range=True, legend_fontsize=20, plot_all_on_one = False, colors = None, save_label = ""): 
        
        if not plot_all_on_one:
            fig = plt.figure(figsize=(len(key_pairs)*5,5))
        else:
            fig = plt.figure(figsize=(8,8))
        for j,key_pair in enumerate(key_pairs):
            if not plot_all_on_one:
                frame = fig.add_axes((.1 + 0.25*j,1.1,1/len(key_pairs),.9))
                plt.sca(frame)
            else:
                frame = plt.gca()
                
            b_offsets = []
            log_ks_p = []
            for i in np.linspace(min_delta_b,max_delta_b,1000):
                ks_p = self.calculate_2D_ks_test_for_bias(key1 = key_pair[0], key2 = key_pair[1], min_mass = min_mass, bias_offset = i)
                if ks_p == None: 
                    continue
                b_offsets.append(i)
                log_ks_p.append(np.log10(ks_p[1]))
            
            try:
                print(key_pair, round(min(np.array(b_offsets)[np.where(log_ks_p > np.log10(significance))[0]]),3),round(b_offsets[np.argmax(np.array(log_ks_p))],3), round(max(np.array(b_offsets)[np.where(log_ks_p > np.log10(significance))[0]]),3),
                 round(max(np.array(b_offsets)[np.where(log_ks_p > np.log10(significance))[0]]) - min(np.array(b_offsets)[np.where(log_ks_p > np.log10(significance))[0]]), 3))
            except:
                pass
            frame.plot(b_offsets,np.array(log_ks_p), label = f"{key_pair[0].split('_')[0]}, {key_pair[1].split('_')[0]}", color = colors[j] )                    
            frame.hlines(y = np.log10(significance), xmin = -1, xmax = 3, color = "grey")

            if plot_max:
                frame.vlines(b_offsets[np.argmax(np.array(log_ks_p))], ymin = -10, ymax = 10, color = colors[j], ls = "dashed")
            if plot_significance_range:
                try:
                    frame.axvspan(min(np.array(b_offsets)[np.where(log_ks_p > np.log10(significance))[0]]),max(np.array(b_offsets)[np.where(log_ks_p > np.log10(significance))[0]]), ymin = -10, ymax = 10, color = colors[j], alpha = 0.15)
                except:
                    pass

            if not plot_all_on_one:
                ml = MultipleLocator(0.5)
                frame.yaxis.set_minor_locator(ml)
                ml = MultipleLocator(1)
                frame.yaxis.set_major_locator(ml)
                ml = MultipleLocator(0.1)
                frame.xaxis.set_minor_locator(ml)
                ml = MultipleLocator(0.5)
                frame.xaxis.set_major_locator(ml)
                
                if ylims != None:
                    frame.set_ylim(ylims)
                if xlims != None:
                    frame.set_xlim(xlims)
                frame.set_xlabel(r"$\Delta \mathtt{b}$")
                if key_pair == key_pairs[0]:
                    frame.tick_params(axis='y', which='both', left = True, right=False, labelleft=True, labelright=False)
                    frame.set_ylabel(r"$\mathtt{log p}$")
                elif key_pair == key_pairs[-1]:
                    frame.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
                    frame.tick_params(axis='y', which='both', left = False, right=True, labelleft=False, labelright=False)
                    frame.yaxis.set_label_position('right')
                else:
                    frame.tick_params(axis='y', which='both', left = False, right=False, labelleft=False, labelright=False)
                frame.legend(fontsize=legend_fontsize)  
                
        if plot_all_on_one:
            plt.legend(fontsize = legend_fontsize)
            plt.xlim(xlims)
            plt.ylim(ylims)
            plt.xlabel(r"$\Delta \mathtt{b}$")
            plt.ylabel(r"$\mathtt{log \ \ p}_{\Delta \mathtt{b}}$")
            ml = MultipleLocator(0.5)
            plt.gca().yaxis.set_minor_locator(ml)
            ml = MultipleLocator(1)
            plt.gca().yaxis.set_major_locator(ml)
            ml = MultipleLocator(0.1)
            plt.gca().xaxis.set_minor_locator(ml)
            ml = MultipleLocator(0.5)
            plt.gca().xaxis.set_major_locator(ml)
        if not plot_all_on_one:
            plt.savefig(f"{self.save_dir}/shifting_2d_ks_panel_1x4_{save_label}.png", bbox_inches='tight')
        else:
            plt.savefig(f"{self.save_dir}/shifting_2d_ks_{save_label}.png", bbox_inches='tight')
        plt.clf()
        plt.close()    

            

    
    
    
    
    
    
    def calculate_2D_ks_test_for_bias_as_func_of_mass(self, key, pivot_mass, mass_key = None, ks_kwargs = {}, print_data_vals = False, min_num = 10):
        if mass_key == None:
            mass_key = key
        X1 = np.array([ 1 -(halo[key].value/halo["M500_truth"].value) for halo in self.sample_arr if key in halo.keys() and halo[mass_key].value < pivot_mass ])
        X2 = np.array([ 1 -(halo[key].value/halo["M500_truth"].value) for halo in self.sample_arr if key in halo.keys() and halo[mass_key].value >= pivot_mass ])
        if len(X1) < min_num or len(X2) < min_num:
            return None
        if print_data_vals:
            print(np.log10(np.array(sorted([halo[key].value for halo in self.sample_arr if key in halo.keys() and halo[mass_key].value < pivot_mass ]))))
            print(np.log10(np.array(sorted([halo[key].value for halo in self.sample_arr if key in halo.keys() and halo[mass_key].value >= pivot_mass ]))))
        return scipy.stats.ks_2samp(X1, X2, **ks_kwargs)
    
    
    
    
    
    
    
    
    
    
        
    def plot_bias_hist(self, key, min_mass = 0, max_mass = 1e30, mass_key = None, bins = 20, plot_multiple_medians = True, colors = None): 
        fig = plt.figure(figsize=(7,5))

        if isinstance(key, str):
            if mass_key == None:
                mass_key = key
            X1 = np.array([ 1 -(halo[key].value/halo["M500_truth"].value) for halo in self.sample_arr if key in halo.keys() and halo[mass_key].value < max_mass and halo[mass_key].value >= min_mass])
            hist_plot = plt.hist(X1, bins = bins, histtype = "bar",  lw=3, ec="navy", fc="powderblue", alpha=0.5,)
            ml = MultipleLocator(2)
            plt.gca().yaxis.set_minor_locator(ml)
            ml = MultipleLocator(10)
            plt.gca().yaxis.set_major_locator(ml)
            ml = MultipleLocator(0.1)
            plt.gca().xaxis.set_minor_locator(ml)
            ml = MultipleLocator(1)
            plt.gca().xaxis.set_major_locator(ml)
            return fig, np.median(X1)
        if isinstance(key, list):
            for i,keyi in enumerate(key):
                if mass_key == None:
                    mass_key = keyi
                X1 = np.array([ 1 -(halo[keyi].value/halo["M500_truth"].value) for halo in self.sample_arr if keyi in halo.keys() and halo[mass_key].value < max_mass and halo[mass_key].value >= min_mass])
                plt.hist(X1, bins = bins,  lw=3, ec=colors[i], histtype = "step", alpha=0.5, label = keyi.split("_HSE")[0])
                if plot_multiple_medians:
                    plt.vlines(x = np.median(X1), ymin = 0, ymax = 200, color = colors[i], ls = "dashed")
                ml = MultipleLocator(2)
                plt.gca().yaxis.set_minor_locator(ml)
                ml = MultipleLocator(10)
                plt.gca().yaxis.set_major_locator(ml)
                ml = MultipleLocator(0.1)
                plt.gca().xaxis.set_minor_locator(ml)
                ml = MultipleLocator(1)
                plt.gca().xaxis.set_major_locator(ml)
            return fig
        
    def plot_bias_hist_panel(self,key_pairs, min_mass = 0, max_mass = 1e30, ylims = (0,100), xlims = None, mass_key = None, normalise = True, bins = 20, plot_multiple_medians = True, legend_fontsize=20, colors = None, bad_ODR_idxs = []): 
        original_mass_key = mass_key
        masstag = ""
        fig = plt.figure(figsize=(len(key_pairs)*5,5))
        for j,key_pair in enumerate(key_pairs):
            frame = fig.add_axes((.1 + 0.25*j,1.1,1/len(key_pairs),.9))
            plt.sca(frame)
            for i,keyi in enumerate(key_pair):
                if mass_key == None:
                    mass_key = keyi
                # print(f"Mass Key = {mass_key}")
                if keyi == "Xray_ODR_HSE_M500":
                    X1 = np.array([ 1 -(halo[keyi].value/halo["M500_truth"].value) for halo in self.sample_arr if keyi in halo.keys() and halo[mass_key].value < max_mass and halo[mass_key].value >= min_mass and int(halo["idx"]) not in bad_ODR_idxs])
                else:
                    X1 = np.array([ 1 -(halo[keyi].value/halo["M500_truth"].value) for halo in self.sample_arr if keyi in halo.keys() and halo[mass_key].value < max_mass and halo[mass_key].value >= min_mass])
                answer = np.percentile(X1, [16, 50, 84])
                q = np.diff(answer)
                quant_label = "${0:.2f}_{{-{1:.2f}}}^{{+{2:.2f}}}$"
                quant_label = quant_label.format(answer[1], q[0], q[1])
                label = f"{keyi.split('_HSE')[0]} ({quant_label})"
                frame.hist(X1, bins = bins, density=normalise,  lw=3, ec=colors[keyi], histtype = "step", alpha=0.5, label = label)
                if plot_multiple_medians:
                    frame.vlines(x = np.median(X1), ymin = 0, ymax = 200, color = colors[keyi], ls = "dashed")
            if normalise:
                ml = MultipleLocator(0.2)
            else:
                ml = MultipleLocator(2)
            frame.yaxis.set_minor_locator(ml)
            if normalise:
                ml = MultipleLocator(1)
            else:
                ml = MultipleLocator(10)                
            frame.yaxis.set_major_locator(ml)

            ml = MultipleLocator(0.1)
            frame.xaxis.set_minor_locator(ml)
            ml = MultipleLocator(0.5)
            frame.xaxis.set_major_locator(ml)
            if ylims != None:
                frame.set_ylim(ylims)
            if xlims != None:
                frame.set_xlim(xlims)
            frame.set_xlabel(r"$\mathtt{b}$")
            if key_pair == key_pairs[0]:
                frame.tick_params(axis='y', which='both', left = True, right=False, labelleft=True, labelright=False)
                if not normalise: frame.set_ylabel(r"$\mathtt{N}$")
            elif key_pair == key_pairs[-1]:
                frame.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
                frame.tick_params(axis='y', which='both', left = False, right=True, labelleft=False, labelright=False)
                frame.yaxis.set_label_position('right')
            else:
                frame.tick_params(axis='y', which='both', left = False, right=False, labelleft=False, labelright=False)
            if j == 0 and min_mass > 0:
                frame.legend(fontsize=legend_fontsize,title_fontsize=legend_fontsize*1.2, title = r'log M/M$_\odot$ >' + f" {np.log10(min_mass)}")
                if original_mass_key != None: 
                    masstag += f"_{original_mass_key}_>{np.log10(min_mass)}"
                else:
                    masstag += f"_individualmasses_>{np.log10(min_mass)}"
            else:
                frame.legend(fontsize=legend_fontsize)    
                
        plt.savefig(f"{self.save_dir}/mass_bias_hist{masstag}_panel_1x4.png", bbox_inches='tight')
        plt.clf()
        plt.close()  


        
        
        
 
    
    
    
    
    
    
    def _loglog_scaling_bces(self, x_property_key, y_property_key, xerr_key, yerr_key, Ez_power, self_sim_scaling_power,save_name,min_length, mass_key, hubble_correction = True,nboot=10000,  best_fit_min_mass = 0, best_fit_max_mass = 1e30, natural_mass_key = None):
        import bces.bces as BCES
        import nmmn.stats

 
        halo_samples = np.array([halo for halo in self.sample_arr if y_property_key in halo.keys() and x_property_key in halo.keys()])
        halo_samples = np.array([halo for halo in halo_samples if halo[y_property_key].value > 0])
        x_data = np.array([halo[x_property_key].value for halo in halo_samples])
        mass_data = np.array([halo[mass_key].value for halo in halo_samples])
        if len(mass_data) == 0:
            raise RuntimeError(f'No Halos found! Check mass and property keys! Mass key = {mass_key}, x key = {x_property_key}, y key = {y_property_key}')
        if xerr_key != None:
            x_data_err = np.array([halo[xerr_key].value for halo in halo_samples])
        else:
            x_data_err = np.zeros_like(x_data)
        
        if hubble_correction:
            y_data = np.array([halo[y_property_key].value * (self.cosmo.H(z=halo["redshift"])/self.cosmo.H(z=0))**(Ez_power) for halo in halo_samples])
            if yerr_key != None:
                y_data_err = np.array([halo[yerr_key].value * (self.cosmo.H(z=halo["redshift"])/self.cosmo.H(z=0))**(Ez_power) for halo in halo_samples])
            else:
                y_data_err = np.zeros_like(x_data)
        elif not hubble_correction:
            y_data = np.array([halo[y_property_key].value for halo in halo_samples])
            if yerr_key != None:
                y_data_err = np.array([halo[yerr_key].value for halo in halo_samples])
            else:
                y_data_err = np.zeros_like(x_data)
            Ez_correction = 1

        def best_fit_slope(log_x,b,m):
            return m*log_x + b

        cov = np.zeros_like(x_data)
        errx = np.array( [x_data_err[i]/(float(x_data[i])*np.log(10)) for i in range(len(x_data_err))])
        erry = np.array( [y_data_err[i]/(float(y_data[i])*np.log(10)) for i in range(len(y_data_err))])
        errx = np.array([np.sum(err)/2 for err in errx])
        erry = np.array([np.sum(err)/2 for err in erry])
        
        sort_idxs = np.argsort(mass_data)
        
        bces_x_data = np.array(x_data[sort_idxs])
        bces_y_data = np.array(y_data[sort_idxs])
        bces_m_data = np.array(mass_data[sort_idxs])
        bces_errx = np.array(errx[sort_idxs])
        bces_erry = np.array(erry[sort_idxs])


        mass_range_idxs = np.where( (bces_m_data >= best_fit_min_mass) & (bces_m_data <= best_fit_max_mass) )[0]
        print(f"{len(mass_range_idxs)} out of {len(bces_m_data)} fall in the required mass range ({mass_key} > 10$^{{{np.log10(best_fit_min_mass)}}}$)")
        if len(mass_range_idxs) == 0:
            print("Masses:", np.log10(bces_m_data))
        bces_x_data = np.array(bces_x_data[mass_range_idxs])
        bces_y_data = np.array(bces_y_data[mass_range_idxs])
        bces_m_data = np.array(bces_m_data[mass_range_idxs])
        bces_errx = np.array(bces_errx[mass_range_idxs])
        bces_erry = np.array(bces_erry[mass_range_idxs])
        
        if len(bces_x_data) < min_length:
            return [None,], None, None, len(bces_x_data)
        
        bces_x_data = np.log10(bces_x_data)
        bces_y_data = np.log10(bces_y_data)
        #print("bces_x_data", bces_x_data)
        print(f"BCES with {nboot} bootstrap samples")
        a,b,erra,errb,covab=BCES.bcesp(y1= bces_x_data,y1err = bces_errx,y2 = bces_y_data,y2err=bces_erry,cerr=cov,nsim=nboot)        
        return a, b, erra, len(bces_x_data)

        
    def generate_scaling_table(self,emin, emax, yT_emin, yT_emax, bces_methods, mass_key = None, min_masses= [None,], min_length = 5, flagged_length = 10, nboot = 100000):

        user_mass_key = mass_key
        
        
        M500_Xray_vs_weighted_kT_Xray_dict = {"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "Xray_HSE_M500",
        "y_property_key" : "weighted_kT_Xray",
        "xerr_key" : "Xray_HSE_M500_uncertainty",
        "yerr_key" : "weighted_kT_Xray_spread", 
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$kT_X-M_{500,X}$"}        
        
        M500_Xray_vs_kTR500_Xray_dict = {"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "Xray_HSE_M500",
        "y_property_key" : "kT_Xray_at_Xray_HSE_R500",
        "xerr_key" : "Xray_HSE_M500_uncertainty",
        "yerr_key" : "kT_Xray_at_Xray_HSE_R500_spread", 
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$kT_X(R500_X)-M_{500,X}$"}   

        weighted_kT_Xray_vs_M500_Xray_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "weighted_kT_Xray",
        "y_property_key" : "Xray_HSE_M500",
        "xerr_key" : "weighted_kT_Xray_spread",
        "yerr_key" : "Xray_HSE_M500_uncertainty",
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$M_{500,X}-kT_X$"}

        kT_Xray_at_Xray_HSE_R500_vs_M500_Xray_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "kT_Xray_at_Xray_HSE_R500",
        "y_property_key" : "Xray_HSE_M500",
        "xerr_key" : "kT_Xray_at_Xray_HSE_R500_spread",
        "yerr_key" : "Xray_HSE_M500_uncertainty",
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$M_{500,X}-kT_X(R500_X)$"}
        
        
        weighted_S_Xray_vs_M500_Xray_dict={"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "Xray_HSE_M500",
        "y_property_key" : "weighted_S_Xray",
        "xerr_key" : "Xray_HSE_M500_uncertainty",
        "yerr_key" : "weighted_S_Xray_spread",
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$S_X-M_{500,X}$"}  
        
        S_Xray_at_Xray_R500_vs_M500_Xray_dict={"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "Xray_HSE_M500",
        "y_property_key" : "S_Xray_at_Xray_HSE_R500",
        "xerr_key" : "Xray_HSE_M500_uncertainty",
        "yerr_key" : "S_Xray_at_Xray_HSE_R500_spread",
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$S_X(R_{500,X})-M_{500,X}$"}  

        M500_Xray_vs_Lx_in_R500x_dict={"Ez_power" : -7/3,
        "self_sim_scaling_power" : 4/3,
        "x_property_key" : "Xray_HSE_M500",
        "y_property_key" : f"Xray_L_in_Xray_HSE_R500_{emin}-{emax}_RF_keV",
        "xerr_key" : "Xray_HSE_M500_uncertainty",
        "yerr_key" : f"Xray_L_in_Xray_HSE_R500_spread_{emin}-{emax}_RF_keV",
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$L_X-M_{500,X}$"}
        
        
        M500_SO_vs_Lx_truth_in_R500_SO_dict={"Ez_power" : -7/3,
        "self_sim_scaling_power" : 4/3,
        "x_property_key" : "M500_truth",
        "y_property_key" : f"Lx_in_R500_truth_{yT_emin}_{yT_emax}_keV",
        "xerr_key" : None,
        "yerr_key" : None,
        "natural_mass_key" : "M500_truth",
        "save_name" : r"$L_\text{truth}-M_{500,SO}$"}


        weighted_kT_Xray_vs_Lx_in_R500x_dict={"Ez_power" : -1,
        "self_sim_scaling_power" : 2,
        "x_property_key" : "weighted_kT_Xray",
        "y_property_key" : f"Xray_L_in_Xray_HSE_R500_{emin}-{emax}_RF_keV",
        "xerr_key" : "weighted_kT_Xray_spread",
        "yerr_key" : f"Xray_L_in_Xray_HSE_R500_spread_{emin}-{emax}_RF_keV",
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$L_X-kT_X$"}
        
        kTR500_Xray_vs_Lx_in_R500x_dict={"Ez_power" : -1,
        "self_sim_scaling_power" : 2,
        "x_property_key" : "kT_Xray_at_Xray_HSE_R500",
        "y_property_key" : f"Xray_L_in_Xray_HSE_R500_{emin}-{emax}_RF_keV",
        "xerr_key" : "kT_Xray_at_Xray_HSE_R500_spread",
        "yerr_key" : f"Xray_L_in_Xray_HSE_R500_spread_{emin}-{emax}_RF_keV",
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$L_X-kT_X(R_{500,X})$"}

        M500_Xray_vs_Y_Xray_dict={"Ez_power" : -2/3,
        "self_sim_scaling_power" : 5/3,
        "x_property_key" : "Xray_HSE_M500",
        "y_property_key" : "Xray_Y",
        "xerr_key" : "Xray_HSE_M500_uncertainty",
        "yerr_key" : "Xray_Y_spread",
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$Y_X-M_{500,X}$"}
        
        M500_MW_vs_weighted_kT_MW_dict = {"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "MW_HSE_M500",
        "y_property_key" : "weighted_kT_MW",
        "xerr_key" : "MW_HSE_M500_uncertainty",
        "yerr_key" : "weighted_kT_MW_spread",
        "natural_mass_key" : "MW_HSE_M500",
        "save_name" : r"$kT_{MW}-M_{500,MW}$"}      
        
        M500_MW_vs_kTR500_MW_dict = {"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "MW_HSE_M500",
        "y_property_key" : "kT_MW_at_MW_HSE_R500",
        "xerr_key" : "MW_HSE_M500_uncertainty",
        "yerr_key" : "kT_MW_at_MW_HSE_R500_spread", 
        "natural_mass_key" : "MW_HSE_M500",
        "save_name" : r"$kT_{MW}(R_{500,MW})-M_{500,MW}$"} 

        weighted_kT_MW_vs_M500_MW_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "weighted_kT_MW",
        "y_property_key" : "MW_HSE_M500",
        "xerr_key" : "weighted_kT_MW_spread",
        "yerr_key" : "MW_HSE_M500_uncertainty",
        "natural_mass_key" : "MW_HSE_M500",
        "save_name" : r"$M_{500,MW}-kT_{MW}$"}
        
        kT_MW_at_MW_HSE_R500_vs_M500_MW_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "kT_MW_at_MW_HSE_R500",
        "y_property_key" : "MW_HSE_M500",
        "xerr_key" : "kT_MW_at_MW_HSE_R500_spread",
        "yerr_key" : "MW_HSE_M500_uncertainty",
        "natural_mass_key" : "MW_HSE_M500",
        "save_name" : r"$M_{500,MW}-kT_{MW}(R_{500,MW)}$"}

        weighted_S_MW_vs_M500_MW_dict={"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "MW_HSE_M500",
        "y_property_key" : "weighted_S_MW",
        "xerr_key" : "MW_HSE_M500_uncertainty",
        "yerr_key" : "weighted_S_MW_spread",
        "natural_mass_key" : "MW_HSE_M500",
        "save_name" : r"$S_{MW}-M_{500,MW}$"}  
        
        S_MW_at_MW_R500_vs_M500_MW_dict={"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "MW_HSE_M500",
        "y_property_key" : "S_MW_at_MW_HSE_R500",
        "xerr_key" : "MW_HSE_M500_uncertainty",
        "yerr_key" : "S_MW_at_MW_HSE_R500_spread",
        "natural_mass_key" : "MW_HSE_M500",
        "save_name" : r"$S_{MW}(R_{500,MW})-M_{500,MW}$"}  

        M500_MW_vs_Y_MW_dict={"Ez_power" : -2/3,
        "self_sim_scaling_power" : 5/3,
        "x_property_key" : "MW_HSE_M500",
        "y_property_key" : "MW_Y",
        "xerr_key" : "MW_HSE_M500_uncertainty",
        "yerr_key" : "MW_Y_spread",
        "natural_mass_key" : "MW_HSE_M500",
        "save_name" : r"$Y_{MW}-M_{500,MW}$"}
        
        M500_EW_vs_weighted_kT_EW_dict = {"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "EW_HSE_M500",
        "y_property_key" : "weighted_kT_EW",
        "xerr_key" : "EW_HSE_M500_uncertainty",
        "yerr_key" : "weighted_kT_EW_spread",
        "natural_mass_key" : "EW_HSE_M500",
        "save_name" : r"$kT_{EW}-M_{500,EW}$"}    

        M500_EW_vs_kTR500_EW_dict = {"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "EW_HSE_M500",
        "y_property_key" : "kT_EW_at_EW_HSE_R500",
        "xerr_key" : "EW_HSE_M500_uncertainty",
        "yerr_key" : "kT_EW_at_EW_HSE_R500_spread", 
        "natural_mass_key" : "EW_HSE_M500",
        "save_name" : r"$kT_{EW}(R_{500,EW})-M_{500,EW}$"} 

        weighted_kT_EW_vs_M500_EW_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "weighted_kT_EW",
        "y_property_key" : "EW_HSE_M500",
        "xerr_key" : "weighted_kT_EW_spread",
        "yerr_key" : "EW_HSE_M500_uncertainty",
        "natural_mass_key" : "EW_HSE_M500",
        "save_name" : r"$M_{500,EW}-kT_{EW}$"}
        
        kT_EW_at_EW_HSE_R500_vs_M500_EW_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "kT_EW_at_EW_HSE_R500",
        "y_property_key" : "EW_HSE_M500",
        "xerr_key" : "kT_EW_at_EW_HSE_R500_spread",
        "yerr_key" : "EW_HSE_M500_uncertainty",
        "natural_mass_key" : "EW_HSE_M500",
        "save_name" : r"$M_{500,EW}-kT_{EW}(R_{500,EW)}$"}

        weighted_S_EW_vs_M500_EW_dict={"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "EW_HSE_M500",
        "y_property_key" : "weighted_S_EW",
        "xerr_key" : "EW_HSE_M500_uncertainty",
        "yerr_key" : "weighted_S_EW_spread",
        "natural_mass_key" : "EW_HSE_M500",
        "save_name" : r"$S_{EW}-M_{500,EW}$"}  
        
        S_EW_at_EW_R500_vs_M500_EW_dict={"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "EW_HSE_M500",
        "y_property_key" : "S_EW_at_EW_HSE_R500",
        "xerr_key" : "EW_HSE_M500_uncertainty",
        "yerr_key" : "S_EW_at_EW_HSE_R500_spread",
        "natural_mass_key" : "EW_HSE_M500",
        "save_name" : r"$S_{EW}(R_{500,EW})-M_{500,EW}$"}  

        M500_EW_vs_Y_EW_dict={"Ez_power" : -2/3,
        "self_sim_scaling_power" : 5/3,
        "x_property_key" : "EW_HSE_M500",
        "y_property_key" : "EW_Y",
        "xerr_key" : "EW_HSE_M500_uncertainty",
        "yerr_key" : "EW_Y_spread",
        "natural_mass_key" : "EW_HSE_M500",
        "save_name" : r"$Y_{EW}-M_{500,EW}$"}
        
        M500_LW_vs_weighted_kT_LW_dict = {"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "LW_HSE_M500",
        "y_property_key" : "weighted_kT_LW",
        "xerr_key" : "LW_HSE_M500_uncertainty",
        "yerr_key" : "weighted_kT_LW_spread",
        "natural_mass_key" : "LW_HSE_M500",
        "save_name" : r"$kT_{LW}-M_{500,LW}$"}        
        
        M500_LW_vs_kTR500_LW_dict = {"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "LW_HSE_M500",
        "y_property_key" : "kT_LW_at_LW_HSE_R500",
        "xerr_key" : "LW_HSE_M500_uncertainty",
        "yerr_key" : "kT_LW_at_LW_HSE_R500_spread", 
        "natural_mass_key" : "LW_HSE_M500",
        "save_name" : r"$kT_{LW}(R_{500,LW})-M_{500,LW}$"} 

        weighted_kT_LW_vs_M500_LW_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "weighted_kT_LW",
        "y_property_key" : "LW_HSE_M500",
        "xerr_key" : "weighted_kT_LW_spread",
        "yerr_key" : "LW_HSE_M500_uncertainty",
        "natural_mass_key" : "LW_HSE_M500",
        "save_name" : r"$M_{500,LW}-kT_{LW}$"}
        
        kT_LW_at_LW_HSE_R500_vs_M500_LW_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "kT_LW_at_LW_HSE_R500",
        "y_property_key" : "LW_HSE_M500",
        "xerr_key" : "kT_LW_at_LW_HSE_R500_spread",
        "yerr_key" : "LW_HSE_M500_uncertainty",
        "natural_mass_key" : "LW_HSE_M500",
        "save_name" : r"$M_{500,LW}-kT_{LW}(R_{500,LW)}$"}

        weighted_S_LW_vs_M500_LW_dict={"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "LW_HSE_M500",
        "y_property_key" : "weighted_S_LW",
        "xerr_key" : "LW_HSE_M500_uncertainty",
        "yerr_key" : "weighted_S_LW_spread",
        "natural_mass_key" : "LW_HSE_M500",
        "save_name" : r"$S_{LW}-M_{500,LW}$"}  
        
        S_LW_at_LW_R500_vs_M500_LW_dict={"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "LW_HSE_M500",
        "y_property_key" : "S_LW_at_LW_HSE_R500",
        "xerr_key" : "LW_HSE_M500_uncertainty",
        "yerr_key" : "S_LW_at_LW_HSE_R500_spread",
        "natural_mass_key" : "LW_HSE_M500",
        "save_name" : r"$S_{LW}(R_{500,LW})-M_{500,LW}$"}  

        M500_LW_vs_Y_LW_dict={"Ez_power" : -2/3,
        "self_sim_scaling_power" : 5/3,
        "x_property_key" : "LW_HSE_M500",
        "y_property_key" : "LW_Y",
        "xerr_key" : "LW_HSE_M500_uncertainty",
        "yerr_key" : "LW_Y_spread",
        "natural_mass_key" : "LW_HSE_M500",
        "save_name" : r"$Y_{LW}-M_{500,LW}$"}

        LW_dicts = [M500_LW_vs_weighted_kT_LW_dict,M500_LW_vs_kTR500_LW_dict, weighted_kT_LW_vs_M500_LW_dict, kT_LW_at_LW_HSE_R500_vs_M500_LW_dict, S_LW_at_LW_R500_vs_M500_LW_dict, M500_LW_vs_Y_LW_dict]
        EW_dicts = [M500_EW_vs_weighted_kT_EW_dict,M500_EW_vs_kTR500_EW_dict, weighted_kT_EW_vs_M500_EW_dict, kT_EW_at_EW_HSE_R500_vs_M500_EW_dict, S_EW_at_EW_R500_vs_M500_EW_dict, M500_EW_vs_Y_EW_dict]
        MW_dicts = [M500_MW_vs_weighted_kT_MW_dict,M500_MW_vs_kTR500_MW_dict, weighted_kT_MW_vs_M500_MW_dict, kT_MW_at_MW_HSE_R500_vs_M500_MW_dict, S_MW_at_MW_R500_vs_M500_MW_dict, M500_MW_vs_Y_MW_dict]
        Xray_dicts = [M500_Xray_vs_weighted_kT_Xray_dict,M500_Xray_vs_kTR500_Xray_dict, weighted_kT_Xray_vs_M500_Xray_dict, kT_Xray_at_Xray_HSE_R500_vs_M500_Xray_dict, S_Xray_at_Xray_R500_vs_M500_Xray_dict, M500_Xray_vs_Y_Xray_dict]
        
        
        scaling_dicts = np.array(list(zip(MW_dicts,EW_dicts,LW_dicts,Xray_dicts))).flatten()
        scaling_dicts = np.concatenate( (np.array([M500_Xray_vs_Lx_in_R500x_dict,M500_SO_vs_Lx_truth_in_R500_SO_dict, weighted_kT_Xray_vs_Lx_in_R500x_dict, kTR500_Xray_vs_Lx_in_R500x_dict]),scaling_dicts) , )
        
        table_string = r"\begin{tabular}{||c| " + "p{15mm}"*(len(min_masses)+1) +  r"||}" + "\n" + r"\hline" + "\n" 
        table_string += " & "  + "Self-Similar"
        for min_mass in min_masses:
            if min_mass == 0:
                mass_label = "All"
            else:
                mass_label = f" M > 10$^{{{np.log10(min_mass)}}}$"
            table_string += " & "  + mass_label 
        table_string += r"\\ \hline\hline" + "\n"        
                
        for scaling_dict in scaling_dicts:
            
            if user_mass_key == None:
                mass_key = scaling_dict["natural_mass_key"]
            
            table_string += scaling_dict["save_name"] + " & "
            if scaling_dict["natural_mass_key"] == "MW_HSE_M500" or scaling_dict in [M500_Xray_vs_Lx_in_R500x_dict, weighted_kT_Xray_vs_Lx_in_R500x_dict]:
                table_string += r"\textbf{" + str(round(scaling_dict["self_sim_scaling_power"],2)) + "} & "
            else:
                table_string +=  " & "                
            for min_mass in min_masses:
                print(f"{scaling_dict['save_name']} for M > 10$^{{{np.log10(min_mass)}}}$")
                # print(f" M > 10$^{{{np.log10(min_mass)}}}$)", scaling_dict["save_name"])
                print("Mass Key", mass_key)
                m,b, merr, num_data = self._loglog_scaling_bces(**scaling_dict, nboot = nboot, mass_key = mass_key, best_fit_min_mass = min_mass, best_fit_max_mass = 1e30,  hubble_correction = True, min_length = min_length,)
                
                if num_data < min_length:
                    table_string += " -- & "
                else:
                    for i,bcesMethod in enumerate(bces_methods):
                        m_rounded = str(round(m[bcesMethod],2)).ljust(4, '0')
                        merr_rounded = str(round(merr[bcesMethod],2)).ljust(4, '0')
                        if  np.isnan(m[bcesMethod]) or np.isnan(merr[bcesMethod]):
                            table_string += " --  " +f"\\newline "
                        if float(merr[bcesMethod]) > float(m[bcesMethod]):
                            print(f"Error {merr_rounded} too large compared to value {m_rounded} for {scaling_dict['save_name']} for M > 10$^{{{np.log10(min_mass)}}}$")
                            table_string += " --  " +f"\\newline "
                        elif num_data <= flagged_length:
                            table_string += f"{m_rounded} $\pm$ {merr_rounded}" + r"$^{*}$ " +f"\\newline "
                            print(f"Error {merr_rounded} acceptable compared to value {m_rounded} for {scaling_dict['save_name']} for M > 10$^{{{np.log10(min_mass)}}}$")
                        else:
                            table_string += f"{m_rounded} $\pm$ {merr_rounded}  \\newline "
                            print(f"Error {merr_rounded} acceptable compared to value {m_rounded} for {scaling_dict['save_name']} for M > 10$^{{{np.log10(min_mass)}}}$")
                            
                    table_string = table_string[:-9]
                    table_string += " & "
                    print("\n")
            print("\n", "-"*10)

            table_string = table_string[:-2] ## Remove trailing ampersand
            table_string += r"\\" + "\n" + r"\hline" "\n"
            if scaling_dict["natural_mass_key"] == "Xray_HSE_M500" and scaling_dict not in [weighted_kT_Xray_vs_Lx_in_R500x_dict,]:
                table_string += r"\hline" "\n"
        table_string += r" \hline \hline" + "\n" + "\end{tabular}"
        with open("./Halo_Sample/scaling_relations/scaling_table.txt", "w") as f:
            f.write(f"BCES Methods = {', '.join([str(x) for x in bces_methods])} \n")
            f.write(table_string)


        
        
        
    def mass_evolution_of_scaling_mass_bins(self,emin, emax, yT_emin, yT_emax, bces_methods, mass_key, min_masses= [None,], min_length = 10):
        M500x_vs_weighted_kT_Xray_dict = {"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "Xray_HSE_M500",
        "y_property_key" : "weighted_kT_Xray",
        "xerr_key" : "Xray_HSE_M500_uncertainty",
        "yerr_key" : "weighted_kT_Xray_spread",
        "save_name" : f"kT-M"}        

        weighted_kT_Xray_vs_M500x_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "weighted_kT_Xray",
        "y_property_key" : "Xray_HSE_M500",
        "xerr_key" : "weighted_kT_Xray_spread",
        "yerr_key" : "Xray_HSE_M500_uncertainty",
        "save_name" : f"M-kT"}

        weighted_S_Xray_vs_M500x_dict={"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "Xray_HSE_M500",
        "y_property_key" : "weighted_S_Xray",
        "xerr_key" : "Xray_HSE_M500_uncertainty",
        "yerr_key" : "weighted_S_Xray_spread",
        "save_name" : f"S_X-M"}  

        M500x_vs_Lx_in_R500x_dict={"Ez_power" : -7/3,
        "self_sim_scaling_power" : 4/3,
        "x_property_key" : "Xray_HSE_M500",
        "y_property_key" : f"Xray_L_in_Xray_HSE_R500_{emin}-{emax}_RF_keV",
        "xerr_key" : "Xray_HSE_M500_uncertainty",
        "yerr_key" : f"Xray_L_in_Xray_HSE_R500_spread_{emin}-{emax}_RF_keV",
        "save_name" : f"L_X-M"}
 
        M500_truth_vs_L_in_R500_truth_dict={"Ez_power" : -7/3,
        "self_sim_scaling_power" : 4/3,
        "x_property_key" : "M500_truth",
        "y_property_key" : f"Lx_in_R500_truth_{yT_emin}-{yT_emax}_keV",
        "xerr_key" : None,
        "yerr_key" : None,
        "save_name" : f"L_truth_Ez_corrected"}

        weighted_kT_Xray_vs_Lx_in_R500x_dict={"Ez_power" : -1,
        "self_sim_scaling_power" : 2,
        "x_property_key" : "weighted_kT_Xray",
        "y_property_key" : f"Xray_L_in_Xray_HSE_R500_{emin}-{emax}_RF_keV",
        "xerr_key" : "weighted_kT_Xray_spread",
        "yerr_key" : f"Xray_L_in_Xray_HSE_R500_spread_{emin}-{emax}_RF_keV",
        "save_name" : f"L_X-kT"}

        M500x_vs_Y_dict={"Ez_power" : -2/3,
        "self_sim_scaling_power" : 5/3,
        "x_property_key" : "Xray_HSE_M500",
        "y_property_key" : "Xray_Y",
        "xerr_key" : "Xray_HSE_M500_uncertainty",
        "yerr_key" : "Xray_Y_spread",
        "save_name" : f"Y_X-M"}
        
        scaling_dicts = [M500x_vs_weighted_kT_Xray_dict, weighted_kT_Xray_vs_M500x_dict,  M500x_vs_Lx_in_R500x_dict, weighted_kT_Xray_vs_Lx_in_R500x_dict, M500x_vs_Y_dict]
        fig = plt.figure(figsize = (10,10), facecolor = 'w')
        
        for i,scaling_dict in enumerate(scaling_dicts):
            print("\n")
            print(scaling_dict["save_name"])
            scaling_power = scaling_dict["self_sim_scaling_power"]
            frame = fig.add_axes((.1,.1+ 0.2*i,.8,.2))
            frame.set_xlim(left = np.log10(3e12), right = np.log10(10**14.5))
            # frame.set_ylim(scaling_power -5, scaling_power+5)
            x_property_key = scaling_dict["x_property_key"]
            y_property_key = scaling_dict["y_property_key"]
            halo_samples = np.array([halo for halo in self.sample_arr if y_property_key in halo.keys() and x_property_key in halo.keys()])
            halo_samples = np.array([halo for halo in halo_samples if halo[y_property_key].value > 0])
            mass_data = np.array([halo[mass_key].value for halo in halo_samples])
            sort_idxs = np.argsort(mass_data)
            all_masses = mass_data[sort_idxs]
            mass_bins = np.array_split(all_masses,len(all_masses)//min_length)
            # mass_bins_simplified = [  [mass_bins[i][0] , (mass_bins[i+1][0] + mass_bins[i][-1])/2] for i in range(0,len(mass_bins)-1)]
            # mass_bins_simplified += [[mass_bins[-1][0], 1e40]]  
            masses_arr = []
            m_arr = []
            merr_arr = []
            for mass_bin in mass_bins:
                m,b, merr, num_data = self._loglog_scaling_bces(**scaling_dict, mass_key = mass_key, best_fit_min_mass = mass_bin[0], best_fit_max_mass = mass_bin[-1],  hubble_correction = True, min_length = min_length,)
                print(round(np.log10(mass_bin[0]),3),round(np.log10(mass_bin[1]),3),num_data)
                if num_data != len(mass_bin):
                    print("Problem with bin numbers!")
                    print(num_data,len(mass_bin))
                    return
                masses_arr.append(0.5*(np.log10(mass_bin[0])+np.log10(mass_bin[-1])))
                m_arr.append(m)
                merr_arr.append(merr)
                print(m[0])
                print(merr[0])
            frame.plot(masses_arr,np.array(m_arr)[:,0], color = "mediumaquamarine")
            plt.fill_between(masses_arr,np.array(m_arr)[:,0]+np.array(merr_arr)[:,0],   np.array(m_arr)[:,0]-np.array(merr_arr)[:,0],      alpha = 0.4, color = "seagreen")
            # frame.plot(masses_arr,np.array(m_arr)[:,3])
            frame.scatter(masses_arr,np.array(m_arr)[:,0], color = "seagreen")
            # frame.scatter(masses_arr,np.array(m_arr)[:,3], color = "black")
            frame.hlines(y = scaling_power, xmin = 12, xmax = 16, alpha = 0.6, color = "black", ls = "dashed", lw = 2)
            if i != 0:
                frame.tick_params(axis='x', which='both', bottom = False, top=False, labelbottom=False, labeltop=False)
                frame.set_xlabel("")
            else:
                frame.set_xlabel(r"$\mathtt{M}_\mathtt{500,X}$  [$\mathtt{M}_\odot$]")
            frame.set_ylabel(scaling_dict["save_name"], fontsize = 15)
            max_dev = max(1.2*max(abs(np.array(m_arr)[:,0] - scaling_power)),4)
            frame.set_ylim(scaling_power -max_dev, scaling_power+max_dev)
        plt.show()
        
        
        
    def mass_evolution_of_scaling_func_vs_min_mass(self,emin, emax, yT_emin, yT_emax, bces_methods, mass_key=None, min_masses= [None,], jump_size = 5, min_length = 10, nboot=10000):
        user_mass_key = mass_key
        
        M500_Xray_vs_weighted_kT_Xray_dict = {"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "Xray_HSE_M500",
        "y_property_key" : "weighted_kT_Xray",
        "xerr_key" : "Xray_HSE_M500_uncertainty",
        "yerr_key" : "weighted_kT_Xray_spread", 
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$kT_X-M_{500,X}$"}        

        weighted_kT_Xray_vs_M500_Xray_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "weighted_kT_Xray",
        "y_property_key" : "Xray_HSE_M500",
        "xerr_key" : "weighted_kT_Xray_spread",
        "yerr_key" : "Xray_HSE_M500_uncertainty",
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$M_{500,X}-kT_X$"}

        kT_Xray_at_Xray_HSE_R500_vs_M500_Xray_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "kT_Xray_at_Xray_HSE_R500",
        "y_property_key" : "Xray_HSE_M500",
        "xerr_key" : "kT_Xray_at_Xray_HSE_R500_spread",
        "yerr_key" : "Xray_HSE_M500_uncertainty",
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$M_{500,X}-kT_X(R_{500,X})$"}
        
        
        weighted_S_Xray_vs_M500_Xray_dict={"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "Xray_HSE_M500",
        "y_property_key" : "weighted_S_Xray",
        "xerr_key" : "Xray_HSE_M500_uncertainty",
        "yerr_key" : "weighted_S_Xray_spread",
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$S_X-M_{500,X}$"}  

        M500_Xray_vs_Lx_in_R500x_dict={"Ez_power" : -7/3,
        "self_sim_scaling_power" : 4/3,
        "x_property_key" : "Xray_HSE_M500",
        "y_property_key" : f"Xray_L_in_Xray_HSE_R500_{emin}-{emax}_RF_keV",
        "xerr_key" : "Xray_HSE_M500_uncertainty",
        "yerr_key" : f"Xray_L_in_Xray_HSE_R500_spread_{emin}-{emax}_RF_keV",
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$L_X-M_{500,X}$"}
 
        M500_truth_vs_L_in_R500_truth_dict={"Ez_power" : -7/3,
        "self_sim_scaling_power" : 4/3,
        "x_property_key" : "M500_truth",
        "y_property_key" : f"Lx_in_R500_truth_{yT_emin}-{yT_emax}_keV",
        "xerr_key" : None,
        "yerr_key" : None,
        "save_name" : f"L_truth_Ez_corrected"}

        weighted_kT_Xray_vs_Lx_in_R500x_dict={"Ez_power" : -1,
        "self_sim_scaling_power" : 2,
        "x_property_key" : "weighted_kT_Xray",
        "y_property_key" : f"Xray_L_in_Xray_HSE_R500_{emin}-{emax}_RF_keV",
        "xerr_key" : "weighted_kT_Xray_spread",
        "yerr_key" : f"Xray_L_in_Xray_HSE_R500_spread_{emin}-{emax}_RF_keV",
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$L_X-kT_X$"}

        M500_Xray_vs_Y_Xray_dict={"Ez_power" : -2/3,
        "self_sim_scaling_power" : 5/3,
        "x_property_key" : "Xray_HSE_M500",
        "y_property_key" : "Xray_Y",
        "xerr_key" : "Xray_HSE_M500_uncertainty",
        "yerr_key" : "Xray_Y_spread",
        "natural_mass_key" : "Xray_HSE_M500",
        "save_name" : r"$Y_X-M_{500,X}$"}
        
        M500_MW_vs_weighted_kT_MW_dict = {"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "MW_HSE_M500",
        "y_property_key" : "weighted_kT_MW",
        "xerr_key" : "MW_HSE_M500_uncertainty",
        "yerr_key" : "weighted_kT_MW_spread",
        "natural_mass_key" : "MW_HSE_M500",
        "save_name" : r"$kT_{MW}-M_{500,MW}$"}        

        weighted_kT_MW_vs_M500_MW_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "weighted_kT_MW",
        "y_property_key" : "MW_HSE_M500",
        "xerr_key" : "weighted_kT_MW_spread",
        "yerr_key" : "MW_HSE_M500_uncertainty",
        "natural_mass_key" : "MW_HSE_M500",
        "save_name" : r"$M_{500,MW}-kT_{MW}$"}
        
        kT_MW_at_MW_HSE_R500_vs_M500_MW_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "kT_MW_at_MW_HSE_R500",
        "y_property_key" : "MW_HSE_M500",
        "xerr_key" : "kT_MW_at_MW_HSE_R500_spread",
        "yerr_key" : "MW_HSE_M500_uncertainty",
        "natural_mass_key" : "MW_HSE_M500",
        "save_name" : r"$M_{500,MW}-kT_{MW}(R_{500,MW)}$"}

        weighted_S_MW_vs_M500_MW_dict={"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "MW_HSE_M500",
        "y_property_key" : "weighted_S_MW",
        "xerr_key" : "MW_HSE_M500_uncertainty",
        "yerr_key" : "weighted_S_MW_spread",
        "natural_mass_key" : "MW_HSE_M500",
        "save_name" : r"$S_{MW}-M_{500,MW}$"}  

        M500_MW_vs_Y_MW_dict={"Ez_power" : -2/3,
        "self_sim_scaling_power" : 5/3,
        "x_property_key" : "MW_HSE_M500",
        "y_property_key" : "MW_Y",
        "xerr_key" : "MW_HSE_M500_uncertainty",
        "yerr_key" : "MW_Y_spread",
        "natural_mass_key" : "MW_HSE_M500",
        "save_name" : r"$Y_{MW}-M_{500,MW}$"}
        
        M500_EW_vs_weighted_kT_EW_dict = {"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "EW_HSE_M500",
        "y_property_key" : "weighted_kT_EW",
        "xerr_key" : "EW_HSE_M500_uncertainty",
        "yerr_key" : "weighted_kT_EW_spread",
        "natural_mass_key" : "EW_HSE_M500",
        "save_name" : r"$kT_{EW}-M_{500,EW}$"}        

        weighted_kT_EW_vs_M500_EW_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "weighted_kT_EW",
        "y_property_key" : "EW_HSE_M500",
        "xerr_key" : "weighted_kT_EW_spread",
        "yerr_key" : "EW_HSE_M500_uncertainty",
        "natural_mass_key" : "EW_HSE_M500",
        "save_name" : r"$M_{500,EW}-kT_{EW}$"}
        
        kT_EW_at_EW_HSE_R500_vs_M500_EW_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "kT_EW_at_EW_HSE_R500",
        "y_property_key" : "EW_HSE_M500",
        "xerr_key" : "kT_EW_at_EW_HSE_R500_spread",
        "yerr_key" : "EW_HSE_M500_uncertainty",
        "natural_mass_key" : "EW_HSE_M500",
        "save_name" : r"$M_{500,EW}-kT_{EW}(R_{500,EW)}$"}

        weighted_S_EW_vs_M500_EW_dict={"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "EW_HSE_M500",
        "y_property_key" : "weighted_S_EW",
        "xerr_key" : "EW_HSE_M500_uncertainty",
        "yerr_key" : "weighted_S_EW_spread",
        "natural_mass_key" : "EW_HSE_M500",
        "save_name" : r"$S_{EW}-M_{500,EW}$"}  

        M500_EW_vs_Y_EW_dict={"Ez_power" : -2/3,
        "self_sim_scaling_power" : 5/3,
        "x_property_key" : "EW_HSE_M500",
        "y_property_key" : "EW_Y",
        "xerr_key" : "EW_HSE_M500_uncertainty",
        "yerr_key" : "EW_Y_spread",
        "natural_mass_key" : "EW_HSE_M500",
        "save_name" : r"$Y_{EW}-M_{500,EW}$"}
        
        M500_LW_vs_weighted_kT_LW_dict = {"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "LW_HSE_M500",
        "y_property_key" : "weighted_kT_LW",
        "xerr_key" : "LW_HSE_M500_uncertainty",
        "yerr_key" : "weighted_kT_LW_spread",
        "natural_mass_key" : "LW_HSE_M500",
        "save_name" : r"$kT_{LW}-M_{500,LW}$"}        

        weighted_kT_LW_vs_M500_LW_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "weighted_kT_LW",
        "y_property_key" : "LW_HSE_M500",
        "xerr_key" : "weighted_kT_LW_spread",
        "yerr_key" : "LW_HSE_M500_uncertainty",
        "natural_mass_key" : "LW_HSE_M500",
        "save_name" : r"$M_{500,LW}-kT_{LW}$"}
        
        kT_LW_at_LW_HSE_R500_vs_M500_LW_dict = {"Ez_power" : 1,
        "self_sim_scaling_power" : 3/2,
        "x_property_key" : "kT_LW_at_LW_HSE_R500",
        "y_property_key" : "LW_HSE_M500",
        "xerr_key" : "kT_LW_at_LW_HSE_R500_spread",
        "yerr_key" : "LW_HSE_M500_uncertainty",
        "natural_mass_key" : "LW_HSE_M500",
        "save_name" : r"$M_{500,LW}-kT_{LW}(R_{500,LW)}$"}

        weighted_S_LW_vs_M500_LW_dict={"Ez_power" : 2/3,
        "self_sim_scaling_power" : 2/3,
        "x_property_key" : "LW_HSE_M500",
        "y_property_key" : "weighted_S_LW",
        "xerr_key" : "LW_HSE_M500_uncertainty",
        "yerr_key" : "weighted_S_LW_spread",
        "natural_mass_key" : "LW_HSE_M500",
        "save_name" : r"$S_{LW}-M_{500,LW}$"}  

        M500_LW_vs_Y_LW_dict={"Ez_power" : -2/3,
        "self_sim_scaling_power" : 5/3,
        "x_property_key" : "LW_HSE_M500",
        "y_property_key" : "LW_Y",
        "xerr_key" : "LW_HSE_M500_uncertainty",
        "yerr_key" : "LW_Y_spread",
        "natural_mass_key" : "LW_HSE_M500",
        "save_name" : r"$Y_{LW}-M_{500,LW}$"}

        LW_dicts = [M500_LW_vs_weighted_kT_LW_dict, weighted_kT_LW_vs_M500_LW_dict, kT_LW_at_LW_HSE_R500_vs_M500_LW_dict, weighted_S_LW_vs_M500_LW_dict, M500_LW_vs_Y_LW_dict]
        EW_dicts = [M500_EW_vs_weighted_kT_EW_dict, weighted_kT_EW_vs_M500_EW_dict, kT_EW_at_EW_HSE_R500_vs_M500_EW_dict, weighted_S_EW_vs_M500_EW_dict, M500_EW_vs_Y_EW_dict]
        MW_dicts = [M500_MW_vs_weighted_kT_MW_dict, weighted_kT_MW_vs_M500_MW_dict, M500_MW_vs_Y_MW_dict]
        Xray_dicts = [M500_Xray_vs_weighted_kT_Xray_dict, weighted_kT_Xray_vs_M500_Xray_dict,  M500_Xray_vs_Lx_in_R500x_dict, weighted_kT_Xray_vs_Lx_in_R500x_dict, M500_Xray_vs_Y_Xray_dict]
        
        save_names = ["Xray", "MW", "EW"]
        for k,scaling_dicts in enumerate([Xray_dicts, MW_dicts, EW_dicts]):

            fig = plt.figure(figsize = (10,10), facecolor = 'w') 
            for i,scaling_dict in enumerate(scaling_dicts):
                print("\n")
                print(scaling_dict["save_name"])
                if user_mass_key == None:
                    mass_key = scaling_dict["natural_mass_key"]
                print("mass_key", mass_key)
                scaling_power = scaling_dict["self_sim_scaling_power"]
                frame = fig.add_axes((.1,.1+ 0.2*i,.8,.2))
                frame.set_xlim(left = np.log10(3e12), right = np.log10(10**14.1))
                # frame.set_ylim(scaling_power -5, scaling_power+5)
                x_property_key = scaling_dict["x_property_key"]
                y_property_key = scaling_dict["y_property_key"]
                halo_samples = np.array([halo for halo in self.sample_arr if y_property_key in halo.keys() and x_property_key in halo.keys()])
                halo_samples = np.array([halo for halo in halo_samples if halo[y_property_key].value > 0])
                mass_data = np.array([halo[mass_key].value for halo in halo_samples])
                sort_idxs = np.argsort(mass_data)
                all_masses = np.array(mass_data[sort_idxs])
                min_mass_idx = len(all_masses) - min_length
                masses_arr_0 = []
                m_arr_0 = []
                merr_arr_0 = []
                masses_arr_3 = []
                m_arr_3 = []
                merr_arr_3 = []
                while True:
                    min_mass = all_masses[min_mass_idx]
                    m,b, merr, num_data = self._loglog_scaling_bces(**scaling_dict, nboot = nboot, mass_key = mass_key, best_fit_min_mass = min_mass, best_fit_max_mass = 1e40,  hubble_correction = True, min_length = min_length,)

                    if abs(merr[0]/m[0]) <=  0.25: 
                        masses_arr_0.append(np.log10(min_mass))
                        m_arr_0.append(m[0])
                        merr_arr_0.append(merr[0])
                    if abs(merr[3]/m[3]) <=  0.25: 
                        masses_arr_3.append(np.log10(min_mass))
                        m_arr_3.append(m[3])
                        merr_arr_3.append(merr[3])

                    if min_mass_idx == 0: break
                    min_mass_idx = max(0, min_mass_idx - jump_size)

                if i == len(scaling_dicts) - 1:

                    frame.plot(masses_arr_0,np.array(m_arr_0), color = "mediumaquamarine", label = "Y|X")
                    plt.fill_between(masses_arr_0,np.array(m_arr_0)+np.array(merr_arr_0),   np.array(m_arr_0)-np.array(merr_arr_0),      alpha = 0.4, color = "seagreen")
                    frame.scatter(masses_arr_0,np.array(m_arr_0), color = "seagreen")
                    frame.hlines(y = scaling_power, xmin = 12, xmax = 16, alpha = 0.6, color = "black", ls = "dashed", lw = 2)

                    frame.plot(masses_arr_3,np.array(m_arr_3), color = "indianred", label = "Orthogonal")
                    plt.fill_between(masses_arr_3,np.array(m_arr_3)+np.array(merr_arr_3),   np.array(m_arr_3)-np.array(merr_arr_3),      alpha = 0.4, color = "coral")
                    frame.scatter(masses_arr_3,np.array(m_arr_3), color = "coral")
                    frame.hlines(y = scaling_power, xmin = 12, xmax = 16, alpha = 0.6, color = "black", ls = "dashed", lw = 2)
                    frame.legend(fontsize = 18) 
                else:
                    frame.plot(masses_arr_0,np.array(m_arr_0), color = "mediumaquamarine")
                    plt.fill_between(masses_arr_0,np.array(m_arr_0)+np.array(merr_arr_0),   np.array(m_arr_0)-np.array(merr_arr_0),      alpha = 0.4, color = "seagreen")
                    frame.scatter(masses_arr_0,np.array(m_arr_0), color = "seagreen")
                    frame.hlines(y = scaling_power, xmin = 12, xmax = 16, alpha = 0.6, color = "black", ls = "dashed", lw = 2)

                    frame.plot(masses_arr_3,np.array(m_arr_3), color = "indianred")
                    plt.fill_between(masses_arr_3,np.array(m_arr_3)+np.array(merr_arr_3),   np.array(m_arr_3)-np.array(merr_arr_3),      alpha = 0.4, color = "coral")
                    frame.scatter(masses_arr_3,np.array(m_arr_3), color = "coral")
                    frame.hlines(y = scaling_power, xmin = 12, xmax = 16, alpha = 0.6, color = "black", ls = "dashed", lw = 2)                

                if i != 0:
                    frame.tick_params(axis='x', which='both', bottom = False, top=False, labelbottom=False, labeltop=False)
                    frame.set_xlabel("")
                else:
                    frame.tick_params(axis='x', which='both', bottom = True, top=False, labelbottom=True, labeltop=False)
                    mass_tag = mass_key.split("_")[0]
                    frame.set_xlabel(f"$\lfloor \mathtt{{M}}_\mathtt{{500,{mass_tag}}}$  [$\mathtt{{M}}_\odot$]")
                    ml = MultipleLocator(0.5)
                    frame.xaxis.set_major_locator(ml)
                    ml = MultipleLocator(0.1)
                    frame.xaxis.set_minor_locator(ml)
                if scaling_dict["save_name"] != f"L_X-kT":
                    ml = MultipleLocator(1)
                    frame.yaxis.set_major_locator(ml)
                    ml = MultipleLocator(0.5)
                    frame.yaxis.set_minor_locator(ml)
                else:
                    ml = MultipleLocator(1)
                    frame.yaxis.set_major_locator(ml)
                    ml = MultipleLocator(0.5)
                    frame.yaxis.set_minor_locator(ml)


                frame.set_ylabel(f"$\mathtt{{{scaling_dict['save_name'].replace(r'$','')}}}$", fontsize = 20)
                try:
                    max_dev = min(1.4*max(max(abs(np.array(m_arr_0) - scaling_power)), max(abs(np.array(m_arr_3) - scaling_power))) , 4)
                except:
                    max_dev = 4
                frame.set_ylim(scaling_power -max_dev, scaling_power+max_dev)
                plt.savefig(f"{self.save_dir}/scalings_vs_Mmin_panel_{save_names[k]}.png", bbox_inches='tight')
