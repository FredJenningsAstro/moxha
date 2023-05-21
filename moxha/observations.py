import yt
import numpy as np
import sys
from pathlib import Path
import os
import datetime
from soxs.instrument_registry import instrument_registry
from soxs.instrument import RedistributionMatrixFile
import shutil
import unyt 
import astropy
import math
from matplotlib import cm as mplcm
import matplotlib.pyplot as plt
import logging

from .tools import *
from astropy.io import fits
# yt.set_log_level(40)
import soxs
import imp

print(f"setting soxs config loc to " + f"{str(imp.find_module('moxha')[1])}/instr_files/")
soxs.set_soxs_config("soxs_data_dir", f"{str(imp.find_module('moxha')[1])}/instr_files/")

class Observation:
    
    '''
    Class for making Observations of Halos and also for making corresponding Blank-Sky Observations for Noise Estimations.
    Observation files will be saved under {save_dir}/{run_ID}/instrument_name/OBS
    ------------------------------------------------
    Constructor Positional Arguments:
                    box_path: Path to the Simulation Box. Should be a dataset format loadable by yT (https://yt-project.org/doc/examining/loading_data.html)
                    snap_num: Snapshot Number of the Box. Will be used for filenames.
                    run_ID: Identifier string for the current suite.
                    save_dir: Top-most directory name under which everything will be saved. A given suite will be found under save_dir/run_ID/
                    emin: Min energy for observations in keV
                    emax: Max energy for observations in keV
                    
    Constructor Keyword Arguments:
                    overwrite:
                    redshift: Specify if you want the observation redshift to be different from that read from the box. Default = "from_box"
                    h: Specify if you want the hubble constant (small h) to be different from that read from the box. Default = "from_box"
                    
    Returns: Observation object
        
    '''
    
    def __init__(self, box_path: str,snap_num: int, save_dir: str, run_ID: str, emin: float, emax: float, emin_for_EW_values:float, emax_for_EW_values:float, energies_for_Lx_tot:list, overwrite = False, redshift = "from_box", h = "from_box", test_align = False, enable_parallelism = False,):
        

        self._logger = logging.getLogger("MOXHA")
        if (self._logger.hasHandlers()):
            self._logger.handlers.clear()     
        c_handler = logging.StreamHandler()
        c_handler.setLevel(level = logging.INFO)
        self._logger.setLevel(logging.INFO)
        c_format = logging.Formatter('%(name)s - [%(levelname)-8s] --- %(asctime)s  - %(message)s')
        c_handler.setFormatter(c_format)
        self._logger.addHandler(c_handler)
        
        self.enable_parallelism =  enable_parallelism
        if self.enable_parallelism:
            yt.enable_parallelism()
         
        import pyxsim
        self._logger.info(f"Using pyXSIM version {pyxsim.__version__}")
        self._logger.info(f"Using soxs version {soxs.__version__}")
        self._logger.info(f"Using yt version {yt.__version__}")
        # if str(yt.__version__) != "4.1.1":
        #     print("yT version is not 4.1.1. Exiting...")
        #     sys.exit()
        self._logger.info(f"setting soxs config loc to " + f"{str(imp.find_module('moxha')[1])}/instr_files/")
        soxs.set_soxs_config("soxs_data_dir", f"{str(imp.find_module('moxha')[1])}/instr_files/")
        self._logger.info("Observation Initialised")

        assert(isinstance(run_ID,str) & len(run_ID)<10)   ## Long run_IDs can cause SOXS to fail silently when reading some filetypes   
        self._run_ID = f"{run_ID}_sn{str(snap_num).zfill(3)}"
        self.box_path = Path(box_path)
        assert(isinstance(save_dir,str))
        self._top_save_path = Path(f"{save_dir}/{run_ID}")
        assert(isinstance(redshift, float) or redshift == "from_box")
        self.redshift = redshift
        assert(isinstance(h, float) or h == "from_box")
        self.hubble = h
        self._logger.info(f"Making dir at {self._top_save_path}")
        os.makedirs(self._top_save_path, exist_ok = True)
        self.dataset_cuts = []
        self.emin = emin
        self.emax = emax 
        self.emin_for_EW_values = emin_for_EW_values
        self.emax_for_EW_values = emax_for_EW_values
        
        energies_for_Lx_tot = np.array(energies_for_Lx_tot)
        if (energies_for_Lx_tot.ndim) == 1:
            self.energies_for_Lx_tot = np.array([energies_for_Lx_tot,]).astype(float)
        elif (energies_for_Lx_tot.ndim) == 2:
            self.energies_for_Lx_tot = np.array(energies_for_Lx_tot).astype(float)

        
        athena_chip_ctr_arcsec = 632.37
        chip_width_arcsec = 1144.08213373        
        
        try:
            if test_align:
                self._instrument_defaults = [
                    {"Name":"athena_wfi", "aim_shift": [-125 +  athena_chip_ctr_arcsec - chip_width_arcsec/2, 118 -  athena_chip_ctr_arcsec + chip_width_arcsec/2], "chip_width":250 },
                ]
            else:
                self._instrument_defaults = [
                    {"Name":"athena_wfi", "aim_shift": [-125 +  athena_chip_ctr_arcsec, 118 -  athena_chip_ctr_arcsec ], "chip_width":250 },
                    {"Name":"lem_0.9eV", "aim_shift": [0.0, 0.0], "chip_width":64, 'image_width':1.08 },
                    {"Name":"lem_2eV", "aim_shift": [0.0, 0.0], "chip_width":64, 'image_width':1.08 },
                    {"Name":"chandra_acisi_cy0", "aim_shift": [218, -283], "chip_width":1050, 'image_width':0.25 },
                    {"Name":"chandra_acisi_cy22", "aim_shift": [218, -283], "chip_width":1050, 'image_width':0.25 }
                ]   
        except:
            raise RuntimeError('Default instrument values could not be loaded. Make sure instrument name is one of: athena_wfi, lem_0.9eV, lem_2eV')        
            
            
            
        self.active_instruments = []
        self.active_halos = []
        
     
    
    
    def add_instrument(self, Name: str, exp_time: float, ID = None, reblock=1, aim_shift=None, chip_width=None):
        '''
        Function for adding instruments to the Observation
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
    
        try:
            defaults = [x for x in self._instrument_defaults if x["Name"] == Name ][0]
        except:
            self._logger.error("Could Not Find instrument in MOXHA's defined instrument list!") 
        if len(defaults) == 0:
            self._logger.error("Could Not Find instrument in MOXHA's defined instrument list!")     
        if len(self.active_instruments) == 0:
            self.active_instruments = []
        if aim_shift == None:
            aim_shift = defaults["aim_shift"]
        if chip_width == None:
            chip_width = defaults["chip_width"]
        if ID == None:
            ID = f"{Name}_{exp_time}ks" 
        image_width = defaults.get('image_width' , 1)
        self.active_instruments.append({"Name":Name,"ID":ID, "exp_time": exp_time, "reblock":reblock, "aim_shift":aim_shift, "chip_width":chip_width, "image_width":image_width })
      
    def clear_instruments(self):
        '''
        Clear all currently active Observation instruments.
        ------------------------------------------------
        Returns:
        '''
        self.active_instruments = []
        
        
        
    def set_active_halos(self, halos):
        '''
        Set active halos for the observation. On calling MakePhotons() and ObservePhotons() these halos will be used.
        ------------------------------------------------
        Positional Arguments:
                    halos: Dictionary or list of dictionaries where each dictionary should contain
                    { "index": halo_idx, "center": halo_center, "R500": halo_R500}
        Returns:
        '''
        if isinstance(halos, dict):
            self.active_halos = [halos,]
        elif (isinstance(halos,list)&all([isinstance(x,dict) for x in halos])):
            self.active_halos = halos
        else:
            self._logger.error(f"Could not set active halo/s. Input must be dict or list of dicts")
        


    def add_cut(self, field, gtr_than = None, less_than = None, equals = None):
        '''
        Convenient function to add a filter cut on the dataset before creating photons. Of course you could just filter Observation.ds object in the front-end yourself if preferred.
        ------------------------------------------------
        Positional Arguments:
                    field: yT-readable field upon which to add the filter. e.g. ('gas','density')
                    gtr_than: Astropy Quantity or Unyt_array to enforce as a lower bound on the dataset. Default = None
                    less_than: Astropy Quantity or Unyt_array to enforce as an upper bound on the dataset. Default = None
                    equals: Astropy Quantity or Unyt_array to set the field value at. Default = None
        Returns:
        '''
        # print("NEED TO SANITIZE TO ASTROPY/UNYT UNITS")
        self.dataset_cuts.append({"field":field, ">":gtr_than, "<":less_than, "=": equals})
        
        
    def _write_log(self, message_type, message):
        now = datetime.datetime.now()
        if message_type == "obs":
            f = open(self._obs_log, 'a')
        if message_type == "deproj":
            f = open(self._deproj_log, 'a')
        f.write("[{0}] ".format(now.strftime("%Y-%m-%d %H:%M:%S")) + str(message))
        f.write('\n')
        f.close()


    def _write_warning(self, message_type, message):
        now = datetime.datetime.now()
        if message_type == "obs":
            f = open(self._obs_log, 'a')
        if message_type == "deproj":
            f = open(self._deproj_log, 'a')
        f.write("[{0}] >>>>> WARNING! >>>> ".format(now.strftime("%Y-%m-%d %H:%M:%S")) + str(message))
        f.write('\n')
        f.close()



    def load_ds(self, **ds_load_kwargs):  
        '''
        Load the dataset in through yT, using the box_name attribute of the Observation instance to specify the path to the box. Sets default_species_fields="ionized".
        ------------------------------------------------
        Keyword Arguments:
                ds_load_kwargs as accepted by yT project's yt.load() (https://yt-project.org/doc/examining/loading_data.html)
        Returns:
        '''
        
        self._logger.info(f"Loading dataset {self.box_path}...")
        self.ds = yt.load(self.box_path, default_species_fields="ionized", **ds_load_kwargs)
        if self.redshift == "from_box":
            self.redshift = float(self.ds.current_redshift)
        if self.hubble == "from_box":
            self.hubble = float(self.ds.hubble_constant)
        try:
            del self.ds['filtered_gas']
        except Exception as e:
            self._logger.warning(e)
        self._logger.info(f"Box at {self.box_path} successfully loaded.")
        

            
    def calculate_simba_pressurization_density(self):
        '''
        Calculate the pressurization density for SIMBA. Can be used in a dataset filter to remove artificially-pressurized gas e.g. galaxies.
        ------------------------------------------------
        Returns: Density unyt_array 
        '''        
        mu = 0.59
        Nngb = 64
        T0 = 1e4 * unyt.K
        m_gas_el = 1.82e7 * unyt.M_sun
        nth = (3/(4 * math.pi * mu * unyt.mp))*((5*unyt.kb * T0)/(unyt.G * mu * unyt.mp))**3 * (1/(Nngb * m_gas_el))**2
        simba_rho_th = nth.to(unyt.cm**-3) * 0.6*unyt.mp
        return simba_rho_th

    
    def MakePhotons(self, area = (2.5, "m**2"), nbins = 6000,  metals = None, photons_emin = 0.0, photons_emax = 10.0, model = "CIE APEC", nH_val = 0.018, absorb_model="tbabs", sphere_R500s = 5, const_metals = False, thermal_broad=True, orient_vec = (0,0,1), north_vector=None, generator_field = None, photon_sample_exp = 1200, only_profiles = False, make_profiles = True, make_phaseplots = True, overwrite = False):
        '''
        Function to use pyXSIM to generate photon lists for the active halos and then project the photons. We use pyXSIM's CIE Source Model. 
        ------------------------------------------------
        Keyword Arguments:
                        area: Collecting area for photons. Should be larger than the area of all instruments being used. Default = (2.5, "m**2")
                        nbins:  The number of channels in the spectrum. Default = 6000
                        metals: A list of metal fields to use in the source model. Default = None
                        photons_emin: Minimum energy to generate the photons for in keV. If smaller than self.emin, will be set to self.emin. Default = 0
                        photons_emax: Maximum energy to generate the photons for in keV. If larger than self.emax, will be set to self.emax. Default = 10
                        model: pyXSIM model. Options: "CIE APEC", "IGM".  Default = "CIE APEC". 
                        nh_val: Foreground galactic column density in 10^22/cm^2. Default = 0.018
                        absorb_model: foreground absorption model, as accepted by pyXSIM. Default = "tbabs"
                        sphere_R500s: A sphere of radius sphere_R500s * halo R500 will be cut out from the dataset and used to generate the photons. Default = 3
                        const_metals: Set to true if you want a contant metallicity with zmet = 0.3. Default = False
                        thermal_broad: Thermal Broadening of the source model. Default = True
                        orient_vec: Normal vector to the plane of photon projection. Default = (0,0,1)
                        generator_field: yT field to use for the photon generation. Default = None, in which case is set to 'gas'.
                        photon_sample_exp: Exposure time in ks for photon generation. Should be at least a few times longer than the specific intrumental exposure times so that a good photon sample can be used. Default = 1000.
                        only_profiles: Set to true if you just want to measure yT profiles and bypass the generation of photons. Default = False
                        
        More Info on some of the pyXSIM arguments can be found here: 
        Making Photons: https://hea-www.cfa.harvard.edu/~jzuhone/pyxsim/api/photon_list.html#pyxsim.photon_list.make_photons
        Projecting Photons: https://hea-www.cfa.harvard.edu/~jzuhone/pyxsim/api/photon_list.html#pyxsim.photon_list.project_photons
        Source Model: https://hea-www.cfa.harvard.edu/~jzuhone/pyxsim/api/source_models.html#pyxsim.source_models.sources.SourceModel
        
        Returns: 

        '''        

        if self.enable_parallelism:
            yt.enable_parallelism()
        import pyxsim
        
        min_Lx_tot_e = min([x[0] for x in self.energies_for_Lx_tot])
        max_Lx_tot_e = max([x[1] for x in self.energies_for_Lx_tot])
        self._photons_emin = min(photons_emin, self.emin, min_Lx_tot_e, self.emin_for_EW_values)
        self._photons_emax = max(photons_emax, self.emax, max_Lx_tot_e, self.emax_for_EW_values)
        self.generator_field = generator_field
        self._photon_exp_time = (photon_sample_exp,'ks')
        self._area = area
        os.makedirs(self._top_save_path/"LOGS", exist_ok = True)     
        self._pyxsim_source_model = model   
        self._thermal_broad = thermal_broad
        self._const_metals = const_metals
        self._absorb_model = absorb_model
        self._nH_val = nH_val
        self._orient_vec = orient_vec
        self._north_vector = north_vector
        if self.generator_field == None:
            self._logger.warning("No generating field specified for X-rays specified. Will default to gas")
            self.generator_field = "gas"
            
        

        
        if len(self.dataset_cuts) > 0:    
            self._cut_dataset()
        else:
            self._logger.info("No Cut being made on the data before observation")

        
        
        if self._pyxsim_source_model == "CIE APEC":
            print(f"Making CIE Source model with emin = {self._photons_emin}, emax = {self._photons_emax}, nbins = {nbins}")
            if self._const_metals:
                self._logger.warning(f"const_metals set to {self._const_metals}. Zmet will be set to constant 0.3")
                self._source_model = pyxsim.CIESourceModel("apec", emin = self._photons_emin, emax = self._photons_emax, nbins = nbins, 
                            Zmet = 0.3, temperature_field = ("filtered_gas","temperature"), 
                            emission_measure_field= ('filtered_gas', 'emission_measure'),
                            thermal_broad=self._thermal_broad)
                self._logger.info(f'''CIE Model Pars:
                "apec", emin = {self._photons_emin}, emax = {self._photons_emax}, nbins = {nbins}, 
                            Zmet = 0.3, temperature_field = ("filtered_gas","temperature"), 
                            emission_measure_field= ('filtered_gas', 'emission_measure'),
                            thermal_broad={self._thermal_broad}''') 
            else:
                var_elem = {elem.split("_")[0]: ("filtered_gas", "{0}".format(elem)) for elem in metals}       
                self._source_model = pyxsim.CIESourceModel("apec", emin = self._photons_emin, emax = self._photons_emax, nbins = nbins, 
                            Zmet = ("filtered_gas", "metallicity"), temperature_field = ("filtered_gas","temperature"), 
                            emission_measure_field= ('filtered_gas', 'emission_measure'),
                            var_elem=var_elem, thermal_broad=self._thermal_broad)    
                
                self._logger.info(f'''CIE Model Pars:
                "apec", emin = {self._photons_emin}, emax = {self._photons_emax}, nbins = {nbins}, 
                            Zmet = ("filtered_gas", "metallicity"), temperature_field = ("filtered_gas","temperature"), 
                            emission_measure_field= ('filtered_gas', 'emission_measure'),
                            var_elem={var_elem}, thermal_broad={self._thermal_broad}''') 
            
        if self._pyxsim_source_model == "IGM for Gerrit 18052023":   
            var_elem = {elem.split("_")[0]: ("filtered_gas", "{0}".format(elem)) for elem in metals}  
            print(f"Using LEM mock settings 18052023 with emin = {self._photons_emin}, emax = {self._photons_emax}, nbins = {nbins}")
            self._photons_emin = 0.2
            self._photons_emax = 3.0
            
            self._source_model = pyxsim.IGMSourceModel(
            0.2,
            3.0,
            nbins = nbins,
            binscale="linear",
            resonant_scattering=True,
            cxb_factor=0.5,
            kT_max=30.0,
            Zmet = ("filtered_gas", "metallicity"), 
            nh_field=("filtered_gas","H_nuclei_density"),
            temperature_field = ("filtered_gas","temperature"), 
            emission_measure_field= ('filtered_gas', 'emission_measure'),
            var_elem=var_elem,          
        )
            self._logger.info(f'''IGM Model Pars:
            0.2,
            3.0,
            nbins = {nbins},
            binscale="linear",
            resonant_scattering=True,
            cxb_factor=0.5,
            kT_max=30.0,
            Zmet = ("filtered_gas", "metallicity"), 
            nh_field=("filtered_gas","H_nuclei_density"),
            temperature_field = ("filtered_gas","temperature"), 
            emission_measure_field= ('filtered_gas', 'emission_measure'),
            var_elem={var_elem},''')
            
            
        # yt.add_xray_emissivity_field(self.ds, self.emin_for_EW_values, self.emax_for_EW_values, table_type="apec", metallicity = (self.generator_field, "metallicity") , redshift=self.redshift, cosmology=self.ds.cosmology, data_dir="./CODE/instr_files/")
        # for emin_for_Lx_tot, emax_for_Lx_tot in self.energies_for_Lx_tot:
        #     yt.add_xray_emissivity_field(self.ds, emin_for_Lx_tot, emax_for_Lx_tot, table_type="apec", metallicity = (self.generator_field, "metallicity") , redshift=self.redshift, cosmology=self.ds.cosmology, data_dir="./CODE/instr_files/")
    
        self._source_model.make_source_fields(self.ds, self.emin_for_EW_values, self.emax_for_EW_values)   
        
        for emin_for_Lx_tot, emax_for_Lx_tot in self.energies_for_Lx_tot:
            self._source_model.make_source_fields(self.ds, emin_for_Lx_tot, emax_for_Lx_tot)
             
            
            
        for i,halo in enumerate(self.active_halos):
            halo_idx = halo["index"]
            self.R500 = halo["R500"]
            self.R200 = halo.get("R200", None)
            halo_center = halo["center"]
            self._logger.info(f"Making photons for Halo {halo_idx}")
            self._idx_tag = f"{self._run_ID}_h{str(halo_idx).zfill(3)}"
            photons_path = Path(self._top_save_path/"PHOTONS"/self._idx_tag)
            
            if os.path.exists(f"{photons_path}/{self._idx_tag}_photons.h5") and os.path.exists(f"{photons_path}/{self._idx_tag}_halo_phlist.fits") and not overwrite and not only_profiles:
                self._logger.info(f"{photons_path}/{self._idx_tag}_photons.h5 and {photons_path}/{self._idx_tag}_halo_phlist.fits already exist and overwrite == False, so we will skip making photons for this halo index.")
                continue            

            os.makedirs(photons_path, exist_ok = True)            
            
            self._obs_log = self._top_save_path/"LOGS"/f"obs_log_{self._idx_tag}.log"
            self._write_log("obs",f" Using pyXSIM version {pyxsim.__version__}")
            self._write_log("obs",f" Using soxs version {soxs.__version__}")
            self._write_log("obs",f" Using yt version {yt.__version__}")
            self._write_log("obs",f" \n \n Halo {self._idx_tag} \n")
            self._write_log("obs", f"R500 = {self.R500}")
            self._write_log("obs", f"Halo Center = {halo_center}")
            
            self.sp = self.ds.sphere(halo_center, sphere_R500s*self.R500)
            self.sp_of_R500 = self.ds.sphere(halo_center, self.R500)
            if self.R200 != None:
                self.sp_of_R200 = self.ds.sphere(halo_center, self.R200)
            
            
            if make_phaseplots:
                self._yT_phaseplots()

            if make_profiles:
                self._yT_profiles()
            if only_profiles:
                self._logger.info(f"only_profiles set to true, so we are done for this halo...")
                continue
            
            

            # slc = yt.SlicePlot(self.ds,normal = "z", fields = [("filtered_gas","temperature")], width = self.sp.radius, center = self.sp.center)
            # slc.show()
            # slc = yt.SlicePlot(self.ds,normal = "z", fields = [("filtered_gas","density")], width = self.sp.radius, center = self.sp.center)
            # slc.show()
            # lumin_field = str(f"xray_luminosity_{self.emin}_{self.emax}_keV")
            # slc = yt.SlicePlot(self.ds,normal = "z", fields = [("filtered_gas",lumin_field)], width = self.sp.radius, center = self.sp.center)
            # slc.show()
            # emis_field = str(f"xray_photon_emissivity_{self.emin}_{self.emax}_keV")
            # slc = yt.SlicePlot(self.ds,normal = "z", fields = [("filtered_gas",emis_field)], width = self.sp.radius, center = self.sp.center)
            # slc.show()            
            self._logger.info(f"Generating photons with exp time of {(float(self._photon_exp_time[0]), self._photon_exp_time[1])}, collecting area = {self._area}, redshift = {self.redshift}")
            n_photons, n_cells = pyxsim.make_photons(f"{photons_path}/{self._idx_tag}_photons", self.sp, self.redshift, self._area, (float(self._photon_exp_time[0]), self._photon_exp_time[1]), self._source_model, point_sources= False, center = self.sp.center, )
            
            self._logger.info(f'''Make_photons Pars:
            {photons_path}/{self._idx_tag}_photons, self.sp, redshift= {self.redshift}, area= {self._area}, photon_exp_time= {(float(self._photon_exp_time[0]), self._photon_exp_time[1])}, {self._source_model}, point_sources= False, center = {self.sp.center}''')  
            
            self._log_make_photons()
            self._write_log("obs",f"{n_photons}_Photons Generated")
            if n_photons == 0:
                continue

            '''Generate an event file of projected photons'''
            self._log_project_photons()
            n_events = pyxsim.project_photons(f"{photons_path}/{self._idx_tag}_photons", f"{photons_path}/{self._idx_tag}_events", sky_center = (30.0, 45.0), absorb_model=self._absorb_model, nH=self._nH_val, normal = self._orient_vec, north_vector = self._north_vector)
            self._logger.info(f'''Project_photons Pars:
            {photons_path}/{self._idx_tag}_photons, {photons_path}/{self._idx_tag}_events, sky_center = (30.0, 45.0), absorb_model={self._absorb_model}, nH={self._nH_val}, normal = {self._orient_vec}, north_vector = {self._north_vector}''')            
            
            
            
            self._write_log("obs",f"{n_events}_Photons Projected.")
            if n_events == 0:
                self._logger.warning("No projected events detected within pyXSIM!")
                continue      
                

            if self.enable_parallelism:
                self._logger.info(f"yt.enable_parallelism = True so we must merge the phton and event lists profuced by pyxsim at this stage...")
                files_to_merge = []
                for file in os.listdir(f"{photons_path}/"):
                    if f"{self._idx_tag}_events" in file and "events.000" not in file and "events.h5" not in file:
                        files_to_merge.append(f"{photons_path}/{file}")       
                for file in files_to_merge:
                    print("file to merge:",file)
                pyxsim.merge_files(files_to_merge, f"{photons_path}/{self._idx_tag}_events.h5",overwrite=True, add_exposure_times=False)
                for file in files_to_merge:
                    os.remove(f"{photons_path}/{file}")
                files_to_merge = []
                for file in os.listdir(f"{photons_path}/"):
                    if f"{self._idx_tag}_photons" in file and "photons.000" not in file and "photons.h5" not in file:
                        files_to_merge.append(f"{photons_path}/{file}")       
                for file in files_to_merge:
                    print("file to merge:",file)
                pyxsim.merge_files(files_to_merge, f"{photons_path}/{self._idx_tag}_photons.h5",overwrite=True, add_exposure_times=False)
                for file in files_to_merge:
                    os.remove(f"{photons_path}/{file}")
                
                

            
            events = pyxsim.EventList(f"{photons_path}/{self._idx_tag}_events.h5")
            events.write_to_simput(f"{photons_path}/{self._idx_tag}_halo", overwrite=True)      
            
            
            
            
    def ObservePhotons(self, image_energies = None, instr_bkgnd = True, foreground = False, ptsrc_bkgnd = True, no_dither = False, overwrite = False, write_foreground_spectrum = False, spectrum_plot_emin = 0.0, spectrum_plot_emax = 3.0, delete_photon_files = False, calibration_markers = False, soxs_stretch = "log", soxs_cmap = "cubehelix"):
        '''
        Function to use SOXS to observe the event lists generated by pyXSIM. See https://hea-www.cfa.harvard.edu/soxs/index.html.
        ------------------------------------------------
        Keyword Arguments:
                    instr_bkgnd: Include instrumental Background. Default = True
                    foreground: Include foregrounds. Default = True
                    ptsrc_bkgnd: Include point-source background. Default = True
                        
        More Info on SOXS can be found here: https://hea-www.cfa.harvard.edu/soxs/index.html
        Returns: 

        '''    
        # if spectrum_plot_emin == None: spectrum_plot_emin = 0.8 * self.emin
        # if spectrum_plot_emax == None: spectrum_plot_emax = 1.05 * self.emax
        
         
        self._image_energies = image_energies
        self._instr_bkgnd = instr_bkgnd
        self._ptsrc_bkgnd = ptsrc_bkgnd
        self._foreground = foreground
        self.instruments = self.active_instruments
        for instrument in self.instruments:
            os.makedirs(self._top_save_path / instrument["Name"], exist_ok = True)
            
        for i,halo in enumerate(self.active_halos):
            halo_idx = halo["index"]
            self.R500 = halo["R500"]
            self.R200 = halo.get("R200", None)
            halo_center = halo["center"]
            self._idx_tag = f"{self._run_ID}_h{str(halo_idx).zfill(3)}"
            self._obs_log = self._top_save_path/"LOGS"/f"obs_log_{self._idx_tag}.log"
            photons_path = Path(self._top_save_path/"PHOTONS"/self._idx_tag)
            '''for some reason the phlist files must be under ./ for instrument_simulator to work. Lets generate them locally then move them after'''
            try:
                shutil.copy( f"{photons_path}/{self._idx_tag}_halo_phlist.fits", f"{self._idx_tag}_halo_phlist.fits" )
                shutil.copy( f"{photons_path}/{self._idx_tag}_halo_simput.fits", f"{self._idx_tag}_halo_simput.fits" ) 
            except Exception as e:
                self._logger.info(e)
                continue
            
            for instrument in self.active_instruments:
                instrument_name = instrument["Name"]
                aim_shift = instrument["aim_shift"]
                obs_exp_time = instrument["exp_time"]
                if "ID" not in list(instrument.keys()):
                    instrument_ID = instrument_name
                else:
                    instrument_ID = instrument["ID"]
                idx_instr_tag = f"{self._idx_tag}_{instrument_ID}"
                evts_path = Path(self._top_save_path/instrument_name/"OBS"/idx_instr_tag)
                
                if os.path.exists(f"{evts_path}/{idx_instr_tag}_evt.fits") and not overwrite:
                    self._logger.info(f"{evts_path}/{idx_instr_tag}_evt.fits already exists and overwrite == False, so we will skip making photons for this halo index.")
                    continue                    

                os.makedirs(evts_path, exist_ok = True)
                self._logger.info(f"Observing Halo {halo_idx} with {instrument_name}")
                
                self._write_log("obs",f"-------------------------------")
                self._write_log("obs",f"Observing Photons:")
                self._write_log("obs",f"Instrument    = {instrument_name}")
                self._write_log("obs",f"Exposure Time = {obs_exp_time}")
                self._write_log("obs",f"Aimpt shift = {aim_shift} arcseconds")
                self._write_log("obs",f"Instrument Background = {self._instr_bkgnd}")
                self._write_log("obs",f"Point Source Background = {self._ptsrc_bkgnd}")
                self._write_log("obs",f"Foreground = {self._foreground}")
                self._write_log("obs",f"no_dither = {no_dither}")
                self._write_log("obs",f"-------------------------------")                          

                self._logger.info(f"Observering with {instrument_name} with exp time of {obs_exp_time}")
                soxs.instrument_simulator(f"{self._idx_tag}_halo_simput.fits", f"{evts_path}/{idx_instr_tag}_evt.fits", obs_exp_time,
                                          instrument_name, [30., 45.], overwrite=True, instr_bkgnd = self._instr_bkgnd, foreground=self._foreground, ptsrc_bkgnd =self._ptsrc_bkgnd, aimpt_shift = aim_shift, no_dither = no_dither)
                
                
                self._write_log("obs",f"Instrument Simulation Complete")
                
                self._logger.info(f'''instrument_simulator Pars:{self._idx_tag}_halo_simput.fits", {evts_path}/{idx_instr_tag}_evt.fits", {obs_exp_time}, {instrument_name}, [30., 45.], overwrite=True, instr_bkgnd = {self._instr_bkgnd}, foreground={self._foreground}, ptsrc_bkgnd ={self._ptsrc_bkgnd}, aimpt_shift = {aim_shift}, no_dither = {no_dither}''')
                self._logger.info(f"Current SOXS Instrument Registry for instr: {soxs.instrument_registry[instrument_name]}")
                
                
                
                soxs.write_spectrum(f"{evts_path}/{idx_instr_tag}_evt.fits", f"{evts_path}/{idx_instr_tag}_evt.pha", overwrite=True)
                fig, ax = soxs.plot_spectrum(f"{evts_path}/{idx_instr_tag}_evt.pha", xmin=spectrum_plot_emin, xmax=spectrum_plot_emax, lw = 0.5, color = "black" )
                fig.set_size_inches((15,5))
                plt.xscale("linear")
                plt.tight_layout()
                # plt.ylim(100,5000000)
                cmap = mplcm.get_cmap('Spectral')
                if self._image_energies != None:
                    ax = plt.gca()
                    for energy_dict in self._image_energies:
                        ax.axvspan(energy_dict['emin'], energy_dict['emax'], alpha = 0.6, label = energy_dict['name'], color = cmap(energy_dict['emax']/self.emax))
                plt.legend()  
                plt.savefig(f"{evts_path}/{idx_instr_tag}_spectrum.png", dpi=400)    
                
                
                        
                soxs.write_image(f"{evts_path}/{idx_instr_tag}_evt.fits", f"{evts_path}/{idx_instr_tag}_img.fits",  emin=self.emin, emax=self.emax, overwrite=True)
                fig, ax = soxs.plot_image(f"{evts_path}/{idx_instr_tag}_img.fits", stretch='log', cmap='cubehelix', width = instrument['image_width'])
                with astropy.io.fits.open(f"{evts_path}/{idx_instr_tag}_img.fits") as hdul:
                    center = np.array([float(hdul[0].header['CRPIX1']),float(hdul[0].header['CRPIX2'])] )
                    if calibration_markers:ax.scatter(center[0],center[1], c = "yellow", marker = "+", s = 1000000, linewidths= 0.5)
                    instrument_spec = instrument_registry[instrument_name]

                    if calibration_markers:
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
                plt.savefig(f"{evts_path}/{idx_instr_tag}_img.png")
                fig.clear() 
                plt.close(fig)

                if self._image_energies != None:   
                    for energy_dict in self._image_energies:
                        soxs.write_image(f"{evts_path}/{idx_instr_tag}_evt.fits", f"{evts_path}/{idx_instr_tag}_img_{energy_dict['name']}.fits", emin=energy_dict['emin'], emax=energy_dict['emax'], overwrite=True,)
                        try: 
                            fig, ax = soxs.plot_image(f"{evts_path}/{idx_instr_tag}_img_{energy_dict['name']}.fits", stretch= soxs_stretch, cmap=soxs_cmap , width = instrument['image_width'])
                        except:
                            fig, ax = soxs.plot_image(f"{evts_path}/{idx_instr_tag}_img_{energy_dict['name']}.fits", stretch='sqrt', cmap=soxs_cmap , width = instrument['image_width'])
                            
                        with astropy.io.fits.open(f"{evts_path}/{idx_instr_tag}_img_{energy_dict['name']}.fits") as hdul:
                            center = np.array([float(hdul[0].header['CRPIX1']),float(hdul[0].header['CRPIX2'])] )
                            if calibration_markers:ax.scatter(center[0],center[1], c = "yellow", marker = "+", s = 1000000, linewidths= 0.5)
                            instrument_spec = instrument_registry[instrument_name]

                            if calibration_markers:
                                try:
                                    print("Using chip 1")
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
                        plt.savefig(f"{evts_path}/{idx_instr_tag}_img_{energy_dict['name']}.png")
                        fig.clear() 
                        plt.close(fig)              
                        
                        
                if write_foreground_spectrum and foreground:
                    self._logger.info(f"Making foregrond plots for {instrument_name}")
                    soxs.make_background_file(f"{evts_path}/{idx_instr_tag}_evt_foregrounds.fits", obs_exp_time,
                        instrument_name, [30., 45.], overwrite=True, foreground=True, instr_bkgnd=False,
                        ptsrc_bkgnd=False)     
                    soxs.write_image(f"{evts_path}/{idx_instr_tag}_evt_foregrounds.fits", f"{evts_path}/{idx_instr_tag}_img_foregrounds.fits",  emin=self.emin, emax=self.emax, overwrite=True)
                    fig, ax = soxs.plot_image(f"{evts_path}/{idx_instr_tag}_img_foregrounds.fits", stretch=soxs_stretch, cmap=soxs_cmap , width = instrument['image_width'])
                    with astropy.io.fits.open(f"{evts_path}/{idx_instr_tag}_img_foregrounds.fits") as hdul:
                        center = np.array([float(hdul[0].header['CRPIX1']),float(hdul[0].header['CRPIX2'])] )
                        if calibration_markers:ax.scatter(center[0],center[1], c = "yellow", marker = "+", s = 1000000, linewidths= 0.5)
                        instrument_spec = instrument_registry[instrument_name]   
                    
                    if calibration_markers:    
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
                        
                    plt.savefig(f"{evts_path}/{idx_instr_tag}_img_foregrounds.png")
                    fig.clear() 
                    plt.close(fig)
                    
                    if self._image_energies != None:   
                        for energy_dict in self._image_energies:
                            soxs.write_image(f"{evts_path}/{idx_instr_tag}_evt_foregrounds.fits", f"{evts_path}/{idx_instr_tag}_img_{energy_dict['name']}_foregrounds.fits", emin=energy_dict['emin'], emax=energy_dict['emax'], overwrite=True,)
                            try:
                                fig, ax = soxs.plot_image(f"{evts_path}/{idx_instr_tag}_img_{energy_dict['name']}_foregrounds.fits", stretch= soxs_stretch, cmap=soxs_cmap , width = instrument['image_width'])
                            except:
                                continue
                            with astropy.io.fits.open(f"{evts_path}/{idx_instr_tag}_img_{energy_dict['name']}_foregrounds.fits") as hdul:
                                center = np.array([float(hdul[0].header['CRPIX1']),float(hdul[0].header['CRPIX2'])] )
                                if calibration_markers:ax.scatter(center[0],center[1], c = "yellow", marker = "+", s = 1000000, linewidths= 0.5)
                                instrument_spec = instrument_registry[instrument_name]
                                
                                if calibration_markers:
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
                                    
                            plt.savefig(f"{evts_path}/{idx_instr_tag}_img_{energy_dict['name']}_foregrounds.png")
                            fig.clear() 
                            plt.close(fig)      
                    
                    soxs.write_spectrum(f"{evts_path}/{idx_instr_tag}_evt_foregrounds.fits", f"{evts_path}/{idx_instr_tag}_evt_foregrounds.pha", overwrite=True)         
                    with fits.open(f"{evts_path}/{idx_instr_tag}_evt.pha") as hdul:
                            total_rates = hdul["SPECTRUM"].data.field('COUNT_RATE')      
                            rmf = hdul["SPECTRUM"].header.get("RESPFILE", None)
                            rmf = RedistributionMatrixFile(rmf)
                            energy_bins = 0.5*(rmf.ebounds_data["E_MIN"]+rmf.ebounds_data["E_MAX"])  
                            xerr = 0.5 * (rmf.ebounds_data["E_MAX"] - rmf.ebounds_data["E_MIN"])  
                            
                    with fits.open(f"{evts_path}/{idx_instr_tag}_evt_foregrounds.pha") as hdul:
                            foreground_rates = hdul["SPECTRUM"].data.field('COUNT_RATE')        
                            rmf = hdul["SPECTRUM"].header.get("RESPFILE", None)
                            rmf = RedistributionMatrixFile(rmf)
                            if not np.array_equal(energy_bins, 0.5*(rmf.ebounds_data["E_MIN"]+rmf.ebounds_data["E_MAX"])) :
                                self._logger.error("Energy bins don't match!")
                                sys.exit() 
                    
                    
                    font = {'family' : 'monospace',
                    'weight' : 'normal',
                    'size'   : 25}
                    plt.rc("font", **font)    
                    plt.rcParams['xtick.major.size'] = 15
                    plt.rcParams['xtick.major.width'] = 2
                    plt.rcParams['xtick.minor.size'] = 10
                    plt.rcParams['xtick.minor.width'] = 2  
                    plt.rcParams['ytick.major.size'] = 15
                    plt.rcParams['ytick.major.width'] = 2
                    plt.rcParams['ytick.minor.size'] = 10
                    plt.rcParams['ytick.minor.width'] = 2       
                    fig = plt.figure()
                    fig.set_size_inches((15,5))
                    plt.xscale("linear")
                    plt.tight_layout()
                    # plt.ylim(100,5000000)
                    used = total_rates > 0 ## min y_value for including point in plot
                    energy_bins = energy_bins[used]
                    total_rates = total_rates[used]
                    foreground_rates = foreground_rates[used]
                    xerr = xerr[used]
                    plt.plot(energy_bins, total_rates/(2*xerr), label = "Observed", color = "darkred", lw = 0.5)
                    plt.plot(energy_bins, foreground_rates/(2*xerr), label = "MW-Foreground", color = "teal", lw = 0.5)
                    
                    used = total_rates - foreground_rates > 0 ## min y_value for including point in plot
                    energy_bins = energy_bins[used]
                    total_rates = total_rates[used]
                    foreground_rates = foreground_rates[used]
                    xerr = xerr[used]                   
                    plt.plot(energy_bins, (total_rates-foreground_rates)/(2*xerr), label = "MW-Subtracted", color = "orchid", lw = 0.5)
                    plt.yscale('log')
                    plt.xlabel("E [keV]")
                    plt.ylabel("Cts/s/keV")
                    
                    plt.xlim(left=spectrum_plot_emin, right=spectrum_plot_emax,)
                    cmap = mplcm.get_cmap('tab20')
                    if self._image_energies != None:
                        ax = plt.gca()
                        max_line_E = max([k['emax'] for k in self._image_energies])
                        for energy_dict in self._image_energies:
                            ax.axvspan(energy_dict['emin'], energy_dict['emax'], alpha = 0.25, label = energy_dict['name'], color = cmap(energy_dict['emax']/spectrum_plot_emax))
                    plt.legend(fontsize = 16, loc = "upper right") 
                    plt.tight_layout() 
                    plt.savefig(f"{evts_path}/{idx_instr_tag}_spectrum_w_foregrounds.png", dpi=400)                               
                    self._logger.info(f"Observation made for Halo {halo_idx} with {instrument_name}")                
                
            os.remove(f"{self._idx_tag}_halo_phlist.fits"  )
            os.remove(f"{self._idx_tag}_halo_simput.fits" )
            
            if delete_photon_files:
                self._logger.info(f"Asked to delete photon files, so removing dir {photons_path}")
                shutil.rmtree(photons_path) 
        
            
            

        
                       
                  
    def _cut_dataset(self):  
        ''' Finer control over cuts can of cource be obtained by just adding cuts onto the Observation.ds object before observing if desired'''
        def _filtered_gas(pfilter, data):
            pfilter = True
            for i, cut in enumerate(self.dataset_cuts):
                if cut["="] != None:
                    pfilter &= data[cut["field"]] == cut["="]
                else:
                    if cut[">"] != None:
                        pfilter &= data[cut["field"]] > cut[">"]
                    if cut["<"] != None:
                        pfilter &= data[cut["field"]] < cut["<"]  
                              
            return pfilter
        required_fields = [x["field"][1] for x in self.dataset_cuts if  x["field"][0] == self.generator_field ]
        yt.add_particle_filter("filtered_gas", function=_filtered_gas, filtered_type=self.generator_field, requires=required_fields)
        if len(required_fields) == 0:
            self._logger.error("No required fields specified in the particle filter!")
            sys.exit()
        self.ds.add_particle_filter("filtered_gas")     
        self._logger.info("Finished Filtering on Dataset")
                           
     
        
    def BlankSkyBackgrounds(self,N, calibration_markers = False):
        '''
        Function to use SOXS to generate mock blank-sky observations for the active instruments using the same set-up as used for ObservePhotons().
        ------------------------------------------------
        Keyword Arguments:
                    N: The number of blank-sky observationss to create. If more than one, can use several blank-skys to get an average noise spectrum further down the line.
                        
        More Info on SOXS can be found here: https://hea-www.cfa.harvard.edu/soxs/index.html
        Returns: 

        '''     
        self._num_backgrounds = N
        self._photon_exp_time = (1000,'ks')
        for i in range(self._num_backgrounds):
            self._logger.info(f"Making Blank Sky Observation {i}")
            self._idx_tag = f"{self._run_ID}_blanksky{str(i).zfill(2)}"
            self._obs_log = self._top_save_path/"LOGS"/f"obs_log_{self._idx_tag}.log"
            for instrument in self.active_instruments:
                instrument_name = instrument["Name"]
                if "ID" not in list(instrument.keys()):
                    instrument_ID = instrument_name
                else:
                    instrument_ID = instrument["ID"]
                obs_exp_time = instrument["exp_time"]
                aim_shift = instrument["aim_shift"]
                idx_instr_tag = f"{self._idx_tag}_{instrument_ID}"
                evts_path = Path(self._top_save_path/instrument_name/"OBS"/idx_instr_tag)
                os.makedirs(evts_path, exist_ok = True)
                
                self._write_log("obs",f"-------------------------------")
                self._write_log("obs",f"Observing Photons:")
                self._write_log("obs",f"Instrument    = {instrument_name}")
                self._write_log("obs",f"Exposure Time = {obs_exp_time}")
                self._write_log("obs",f"Instrument Background = {self._instr_bkgnd}")
                self._write_log("obs",f"Point Source Background = {self._ptsrc_bkgnd}")
                self._write_log("obs",f"Foreground = {self._foreground}")
                self._write_log("obs",f"-------------------------------")                          

                if os.path.exists(f"{evts_path}/{idx_instr_tag}_evt.fits"):
                    self._logger.info("Background File {evts_path}/{idx_instr_tag}_evt.fits already exists. Won't make another one...")
                    
                else:
                    make_background_file_with_shift(f"{evts_path}/{idx_instr_tag}_evt.fits", exp_time = obs_exp_time, instrument = instrument_name, sky_center = [30., 45.], overwrite=True, 
                                                         foreground=self._foreground, instr_bkgnd=self._instr_bkgnd, ptsrc_bkgnd=self._ptsrc_bkgnd, aimpt_shift = aim_shift)
                    soxs.write_spectrum(f"{evts_path}/{idx_instr_tag}_evt.fits", f"{evts_path}/{idx_instr_tag}_evt.pha", overwrite=True)        
                    
                    soxs.write_image(f"{evts_path}/{idx_instr_tag}_evt.fits", f"{evts_path}/{idx_instr_tag}_img.fits",  emin=self.emin, emax=self.emax, overwrite=True)
                    fig, ax = soxs.plot_image(f"{evts_path}/{idx_instr_tag}_img.fits", stretch='log', cmap='cubehelix',)
                    with astropy.io.fits.open(f"{evts_path}/{idx_instr_tag}_img.fits") as hdul:
                        center = np.array([float(hdul[0].header['CRPIX1']),float(hdul[0].header['CRPIX2'])] )
                        if calibration_markers:ax.scatter(center[0],center[1], c = "yellow", marker = "+", s = 1000000, linewidths= 0.5)
                        instrument_spec = instrument_registry[instrument_name]

                        if calibration_markers:
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
                            
                    plt.savefig(f"{evts_path}/{idx_instr_tag}_img.png")
                    fig.clear() 
                    plt.close(fig)
                        
                    if self._image_energies != None:
                        for energy_dict in self._image_energies:
                            soxs.write_image(f"{evts_path}/{idx_instr_tag}_evt.fits", f"{evts_path}/{idx_instr_tag}_img_{energy_dict['name']}.fits",  emin=energy_dict['emin'], emax=energy_dict['emax'], overwrite=True)
                            fig, ax = soxs.plot_image(f"{evts_path}/{idx_instr_tag}_img_{energy_dict['name']}.fits", stretch='log', cmap='cubehelix',)
                            with astropy.io.fits.open(f"{evts_path}/{idx_instr_tag}_img_{energy_dict['name']}.fits") as hdul:
                                center = np.array([float(hdul[0].header['CRPIX1']),float(hdul[0].header['CRPIX2'])] )
                                if calibration_markers:ax.scatter(center[0],center[1], c = "yellow", marker = "+", s = 1000000, linewidths= 0.5)
                                instrument_spec = instrument_registry[instrument_name]
                                
                                if calibration_markers:
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
                                    
                            plt.savefig(f"{evts_path}/{idx_instr_tag}_img_{energy_dict['name']}.png")
                            fig.clear() 
                            plt.close(fig)
                            
                    

                self._logger.info(f"Observation made for Blank Sky Region {i} with {instrument_name}")
        
        
        
        


    def _yT_profiles(self):
        self._logger.info(f"Generating yT Profiles")
        yt_data_path = Path(self._top_save_path/"YT_DATA"/self._idx_tag)
        os.makedirs(yt_data_path, exist_ok = True)   
        n_bins = 50
        
        ptype = "filtered_gas" # ["PartType0","PartType1","PartType4","PartType5"]
        lumin_field = str(f"xray_luminosity_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV")
        emis_field = str(f"xray_emissivity_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV")
        radius = (ptype,"radius")
        
        # for field in self.ds.derived_field_list:
        #     # if field[0] == ptype:
        #         print(field)
        self._logger.info(f"Emmission-Weighted quantities being calculated with {lumin_field} and {emis_field}")
        rp_total_mass = yt.create_profile(self.sp,("all",'particle_position_spherical_radius'), extrema = {("all",'particle_position_spherical_radius'):(0.05*self.R500, 1.2*self.R500)},
                               fields = [("all", 'Masses')],
                               units={("all",'particle_position_spherical_radius'): "kpc", ("all", 'Masses'):"Msun"},logs={("all",'particle_position_spherical_radius'): False},
                               weight_field = None,
                               accumulation = True,
                               n_bins = n_bins)
        self._logger.info(f"Halo {self._idx_tag}: Successfully read True Total Mass Profile from Dataset") 
        
        rp_kT_EW = yt.create_profile(self.sp,radius, extrema = {radius:(0.05*self.R500, 1.2*self.R500)},
                               fields=[(ptype, 'temperature')],
                               units={radius: "kpc", (ptype, 'temperature'):"K"},logs={radius: False},
                               weight_field = (ptype, emis_field) ,
                               accumulation = False,
                               n_bins = n_bins)    
        self._logger.info(f"Halo {self._idx_tag}: Successfully read Emmissivity-Weighted kT Profile from Dataset")
        
        rp_ne_EW = yt.create_profile(self.sp,radius, extrema = {radius:(0.05*self.R500, 1.2*self.R500)},
                               fields=[(ptype, 'El_number_density')],
                               units={radius: "kpc", (ptype, 'El_number_density'):"cm**-3"},logs={radius: False},
                               weight_field = (ptype, emis_field),
                               accumulation = False,
                               n_bins = n_bins)  
        self._logger.info(f"Halo {self._idx_tag}: Successfully read Emmissivity-Weighted ne Profile from Dataset")
        
        rp_kT_LuminW = yt.create_profile(self.sp,radius, extrema = {radius:(0.05*self.R500, 1.2*self.R500)},
                               fields=[(ptype, 'temperature')],
                               units={radius: "kpc", (ptype, 'temperature'):"K"},logs={radius: False},
                               weight_field = (ptype, lumin_field) ,
                               accumulation = False,
                               n_bins = n_bins)    
        self._logger.info(f"Halo {self._idx_tag}: Successfully read Luminosity-Weighted kT Profile from Dataset")
        
        rp_ne_LuminW = yt.create_profile(self.sp,radius, extrema = {radius:(0.05*self.R500, 1.2*self.R500)},
                               fields=[(ptype, 'El_number_density')],
                               units={radius: "kpc", (ptype, 'El_number_density'):"cm**-3"},logs={radius: False},
                               weight_field = (ptype, lumin_field),
                               accumulation = False,
                               n_bins = n_bins)  
        self._logger.info(f"Halo {self._idx_tag}: Successfully read Luminosity-Weighted ne Profile from Dataset")
        
        rp_kT_filtGasMassW = yt.create_profile(self.sp,radius, extrema = {radius:(0.05*self.R500, 1.2*self.R500)},
                               fields=[(ptype, 'temperature')],
                               units={radius: "kpc", (ptype, 'temperature'):"K"},logs={radius: False},
                               weight_field = (ptype, "mass") ,
                               accumulation = False,
                               n_bins = n_bins)    
        self._logger.info(f"Halo {self._idx_tag}: Successfully read Mass-Weighted kT Profile from Dataset")
        
        rp_ne_filtGasMassW = yt.create_profile(self.sp,radius, extrema = {radius:(0.05*self.R500, 1.2*self.R500)},
                               fields=[(ptype, 'El_number_density')],
                               units={radius: "kpc", (ptype, 'El_number_density'):"cm**-3"},logs={radius: False},
                               weight_field = (ptype, "mass"),
                               accumulation = False,
                               n_bins = n_bins)  
        self._logger.info(f"Halo {self._idx_tag}: Successfully read Mass-Weighted ne Profile from Dataset")
        
        rp_Lx_RAW = yt.create_profile(self.sp,radius, extrema = {radius:(0.05*self.R500, 1.2*self.R500)},
                               fields=[(ptype, lumin_field)],
                               units={radius: "kpc", (ptype, lumin_field) :"erg/s"},logs={radius: False},
                               weight_field = None,
                               accumulation = False,
                               n_bins = n_bins)   
        self._logger.info(f"Halo {self._idx_tag}: Successfully read True Luminosity Profile from Dataset")       
        

        rp_data = []
        
        for emin_for_Lx_tot, emax_for_Lx_tot in self.energies_for_Lx_tot:
            self._logger.info(f"Total Lx in R500 being calculated with xray_luminosity_{emin_for_Lx_tot}_{emax_for_Lx_tot}_keV field")
            lumin_field_for_Lx_tot = str(f"xray_luminosity_{emin_for_Lx_tot}_{emax_for_Lx_tot}_keV")
            total_Lx_in_R500 = self.sp_of_R500.quantities.total_quantity([(ptype, lumin_field_for_Lx_tot),])
            rp_data.append(  {"Name":f"total_Lx_in_R500_{emin_for_Lx_tot}_{emax_for_Lx_tot}_keV", "value": total_Lx_in_R500.to_astropy()  }) 
            if self.R200 != None:
                self._logger.info(f"Total Lx in R200 being calculated with xray_luminosity_{emin_for_Lx_tot}_{emax_for_Lx_tot}_keV field")
                total_Lx_in_R200 = self.sp_of_R200.quantities.total_quantity([(ptype, lumin_field_for_Lx_tot),])
                rp_data.append(  {"Name":f"total_Lx_in_R200_{emin_for_Lx_tot}_{emax_for_Lx_tot}_keV", "value": total_Lx_in_R200.to_astropy()  }) 
                lx_r = self.R200
            else:
                lx_r = self.R500    
            Lx_profile = yt.create_profile(self.sp,radius, extrema = {radius:(0.01*self.R500, 1.2*lx_r)},
                        fields=[(ptype, lumin_field_for_Lx_tot)],
                        units={radius: "kpc", (ptype, lumin_field_for_Lx_tot) :"erg/s"},logs={radius: False},
                        weight_field = None,
                        accumulation = False,
                        n_bins = n_bins)   
            print(f"Saving radial profile for the luminosity with xray_luminosity_{emin_for_Lx_tot}_{emax_for_Lx_tot}_keV")
            rp_data.append(  {"Name":f"Lx_profile_{emin_for_Lx_tot}_{emax_for_Lx_tot}_keV","radius":Lx_profile.x.to_astropy(), "values":Lx_profile[(ptype, lumin_field_for_Lx_tot)].to_astropy()  })
                
        
        
        
        
        rp_data.append(  {"Name":f"kT_EW_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV","radius":rp_kT_EW.x.to_astropy(), "values":rp_kT_EW[(ptype,'temperature')].to_astropy() })
        rp_data.append(  {"Name":f"ne_EW_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV","radius":rp_ne_EW.x.to_astropy(), "values":rp_ne_EW[(ptype,'El_number_density')].to_astropy()  })
        
        rp_data.append(  {"Name":f"kT_LuminW_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV","radius":rp_kT_LuminW.x.to_astropy(), "values":rp_kT_LuminW[(ptype,'temperature')].to_astropy() })
        rp_data.append(  {"Name":f"ne_LuminW_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV","radius":rp_ne_LuminW.x.to_astropy(), "values":rp_ne_LuminW[(ptype,'El_number_density')].to_astropy()  })
        
        rp_data.append(  {"Name":f"kT_filtGasMassW","radius":rp_kT_filtGasMassW.x.to_astropy(), "values":rp_kT_filtGasMassW[(ptype,'temperature')].to_astropy() })
        rp_data.append(  {"Name":f"ne_filtGasMassW","radius":rp_ne_filtGasMassW.x.to_astropy(), "values":rp_ne_filtGasMassW[(ptype,'El_number_density')].to_astropy()  })
        
        rp_data.append(  {"Name":"Lx_raw","radius":rp_Lx_RAW.x.to_astropy(), "values":rp_Lx_RAW[(ptype, lumin_field)].to_astropy()  })
        rp_data.append(  {"Name":"Total_Mass","radius":rp_total_mass.x.to_astropy(), "values": rp_total_mass[("all", 'Masses')].to_astropy()  }) 
        
        
        self._logger.info("yT Data Successfully Taken")
        np.save(f"{yt_data_path}/{self._idx_tag}_yt_data_pyxsim.npy",rp_data)  





    def _yT_phaseplots(self,):
        self._logger.info(f"Generating yT Phaseplots")
        yt_data_path = Path(self._top_save_path/"YT_DATA"/self._idx_tag)
        os.makedirs(yt_data_path, exist_ok = True)   
        
        
        ptype = "filtered_gas" # ["PartType0","PartType1","PartType4","PartType5"]
        lumin_field = str(f"xray_luminosity_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV")
        emis_field = str(f"xray_emissivity_{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV")
        
        try:        
            plot = yt.PhasePlot(self.sp, (ptype, "density"), (ptype, "temperature"), [(ptype, "mass")], weight_field=None)
            plot.save(str(yt_data_path) + f"/{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV__density_vs_T_vs_mass_phaseplot.png")
        except Exception as e:
            print(e)
        try:
            plot = yt.PhasePlot(self.sp, (ptype, "density"), (ptype, "temperature"), [(ptype, lumin_field)], weight_field=None)
            plot.set_colorbar_label((ptype, lumin_field), lumin_field)
            plot.save(str(yt_data_path) + f"/{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV__density_vs_T_vs_luminosity_phaseplot.png")
        except Exception as e:
            print(e)
        try:
            plot = yt.PhasePlot(self.sp, ('PartType0', "density"), ('PartType0', "temperature"), [('PartType0', 'StarFormationRate')], weight_field=None)
            plot.set_colorbar_label((ptype, lumin_field), lumin_field)
            plot.save(str(yt_data_path) + f"/{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV__density_vs_T_vs_sfr_phaseplot_UNFILTERED_GAS.png")
        except Exception as e:
            print(e)
        try:        
            plot = yt.PhasePlot(self.sp, (ptype, "density"), (ptype, "temperature"), [(ptype, emis_field)], weight_field=None)
            plot.save(str(yt_data_path) + f"/{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV__density_vs_T_vs_emissivity_phaseplot.png")
        except Exception as e:
            print(e)
        try:        
            plot = yt.PhasePlot(self.sp, (ptype, emis_field), (ptype, lumin_field), [(ptype, "density")], weight_field=None)
            plot.set_ylabel(lumin_field)
            plot.save(str(yt_data_path) + f"/{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV__emis_vs_luminosity_vs_density_phaseplot.png")
        except Exception as e:
            print(e)
        try:        
            plot = yt.PhasePlot(self.sp, (ptype, "density"), (ptype, "metallicity"), [(ptype, lumin_field)], weight_field=None)
            plot.set_colorbar_label((ptype, lumin_field), lumin_field)
            plot.save(str(yt_data_path) + f"/{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV__density_vs_metallicity_vs_luminosity_phaseplot.png")
        except Exception as e:
            print(e)
        try:        
            plot = yt.PhasePlot(self.sp, (ptype, "density"), (ptype, "metal_mass"), [(ptype, lumin_field)], weight_field=None)
            plot.set_colorbar_label((ptype, lumin_field), lumin_field)
            plot.save(str(yt_data_path) + f"/{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV__density_vs_metal_mass_vs_luminosity_phaseplot.png")
        except Exception as e:
            print(e)
        try:        
            plot = yt.PhasePlot(self.sp, (ptype, "x"), (ptype, "y"), [(ptype, lumin_field)], weight_field=None)
            plot.set_colorbar_label((ptype, lumin_field), lumin_field)
            plot.save(str(yt_data_path) + f"/{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV__x_vs_y_vs_luminosity_phaseplot.png")
        except Exception as e:
            print(e)
        try:        
            plot = yt.PhasePlot(self.sp, (ptype, "x"), (ptype, "y"), [(ptype, "density")], weight_field=None)
            plot.save(str(yt_data_path) + f"/{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV__x_vs_y_vs_density_phaseplot.png")
        except Exception as e:
            print(e)
        try:        
            plot = yt.PhasePlot(self.sp, (ptype, "x"), (ptype, "y"), [(ptype, "temperature")], weight_field=None)
            plot.save(str(yt_data_path) + f"/{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV__x_vs_y_vs_T_phaseplot.png")
        except Exception as e:
            print(e)
        try:        
            plot = yt.PhasePlot(self.sp, (ptype, "x"), (ptype, "y"), [(ptype, "metallicity")], weight_field=None)
            plot.save(str(yt_data_path) + f"/{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV__x_vs_y_vs_metallicity_phaseplot.png")
        except Exception as e:
            print(e)
                  
        try:        
            plot = yt.PhasePlot(self.sp, ('PartType0', "x"), ('PartType0', "y"), [('PartType0', 'StarFormationRate')], weight_field=None)
            plot.save(str(yt_data_path) + f"/{self.emin_for_EW_values}_{self.emax_for_EW_values}_keV__x_vs_y_vs_sfr_phaseplot_UNFILTERED_GAS.png")
        except Exception as e:
            print(e)
            
            
            
        






        
        
        
    def _log_make_photons(self):
        self._write_log("obs",f"-------------------------------")
        self._write_log("obs",f"Making Photons:")
        self._write_log("obs",f"Snap = {self.box_path}")
        self._write_log("obs",f"Redshift = {self.redshift}")
        self._write_log("obs",f"Exp Time = { float(self._photon_exp_time[0]), self._photon_exp_time[1]} ")
        self._write_log("obs",f"Area     = {self._area}")
        self._write_log("obs",f"photons emin     = {self._photons_emin} keV")
        self._write_log("obs",f"photons emax     = {self._photons_emax} keV")
        self._write_log("obs",f"Model    = {self._pyxsim_source_model}")
        self._write_log("obs",f"Bounding Sphere Radius    = {self.sp.radius}")
        self._write_log("obs",f"-------------------------------")
        self._write_log("obs",f"Cuts on data:")
        self._write_log("obs",f"Data Cuts     = {self.dataset_cuts} ")
        for cut in self.dataset_cuts:
            if ("filtered_gas", cut["field"][1]) in self.ds.derived_field_list:
                try:
                    cut_field = cut["field"]
                    print("cut field", cut_field)
                    self._write_log("obs",f"ds original gas field min {(cut_field[0],cut_field[1])} = {self.sp.min([(cut_field[0],cut_field[1])])}")
                    self._write_log("obs",f"ds original gas field max {(cut_field[0],cut_field[1])} = {self.sp.max([(cut_field[0],cut_field[1])])}")
                    self._write_log("obs",f"ds filtered field min {('filtered_gas',cut_field[1])} = {self.sp.min(  [  ( 'filtered_gas',cut_field[1] )  ]  )}")
                    self._write_log("obs",f"ds filtered field max {('filtered_gas',cut_field[1])} = {self.sp.max(  [  ( 'filtered_gas',cut_field[1] )  ]  )}")
                except Exception as e:
                    print("error in printing dataset min and maxes. Re-run, since this can ruin photon production due to this bug: https://github.com/jzuhone/pyxsim/issues/44", e)
                    # sys.exit()
        self._write_log("obs",f"-------------------------------")
        self._write_log("obs",f"Constant Metals = {self._const_metals}")
        self._write_log("obs",f"Thermal Broadening = {self._thermal_broad}")
        self._write_log("obs",f"-------------------------------")
        
        
    def _log_project_photons(self):
        self._write_log("obs", f"-------------------------------")
        self._write_log("obs", f"Projecting Photons:")
        self._write_log("obs", f"Absorbtion Model = {self._absorb_model}")
        self._write_log("obs", f"nH               = {self._nH_val}")
        self._write_log("obs", f"Normal Vector    = {self._orient_vec}")
        self._write_log("obs", f"-------------------------------")   