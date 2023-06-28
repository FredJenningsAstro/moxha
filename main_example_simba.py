from moxha.observations import Observation
import unyt
import caesar
import sys

######################################################
####################              ####################
####################  Edit these  ####################
####################              ####################
######################################################

snapnum         = 151                                                                                       ### Box Snapnum
halo_idxs       = [int(sys.argv[1]),]                                                                       ### Indexes into the Caesar list of halos i.e. obj.halos[i]
CSR_infile      = f"/home/b/babul/rennehan/project/simba_data/caesar_{str(snapnum).zfill(3)}.hdf5"          ### Path to Caesar File
SNAPFILE        = f"/home/b/babul/rennehan/project/simba_data/snapshot_{str(snapnum).zfill(3)}.hdf5"        ### Path to box
save_dir        = f"renier_xrays"                                                                           ### Top-level directory name that will appear in the running directory
run_ID          = f"simba_c"                                                                                ### Identifier for this suite of mocks
redshift        = 0.01                                                                                      ### Redshift that Moxha will observe at. To use the box redshift, set = "from_box"

ebins           = [[0.2,3.0],[0.5,2.0],[0.5,2.4]]                                                           ### For making the maps and measuring total luminosity, these are the energies that will be used. Ensure these are between 0.05-10 keV
weighting_e     = [0.2,3.0]                                                                                 ### For the weighted profiles, weight using the emission in this band

### These are the cuts to make on the dataset before the X-rays are generated.
cuts_dict       =  [{'field':('gas','density'), 'less_than':(0.01*unyt.physical_constants.proton_mass)*unyt.cm**-3},
                    {'field':("gas", "temperature"), 'gtr_than': 2e5*unyt.K},
                    {'field':("PartType0", 'Density'), 'less_than':(0.01*unyt.physical_constants.proton_mass)*unyt.cm**-3},
                    {'field':("PartType0", "temperature"), 'gtr_than': 2e5*unyt.K},
                    {'field':("PartType0", "DelayTime"), 'equals':0},
                    {'field':("PartType0", "StarFormationRate"), 'equals':0}]

### These are the instruments you want to observe with, along with an Identifier that will be used in filenames, and the exposure time            
instruments_dict = [{"Name":"chandra_acisi_cy0","ID":"chandra_acisi_cy0_1000ks", "exp_time": (1000,'ks'), "reblock":1},
                    {"Name":"chandra_acisi_cy0_nogap","ID":"chandra_acisi_cy0_nogap_1000ks", "exp_time": (1000,'ks'), "reblock":1},]

### For info on backgrounds/foregrounds, consult https://hea-www.cfa.harvard.edu/soxs/users_guide/background.html
instrument_background   = False                                                                             ### Include instrumental noise in the observations
point_source_background = False                                                                             ### Include CXB sources (e.g. from binaries) in the observations
milky_way_foreground    = False                                                                             ### Include Milky Way CGM emission in the observations




##########################################################################
####################                                  ####################
####################  Don't edit anything below this  ####################
####################                                  ####################
##########################################################################

image_energies = []
for ebin_min, ebin_max in ebins:
    image_energies.append({'name':f"{ebin_min}_{ebin_max}_keV",'emin': ebin_min , 'emax':ebin_max})
halos = []
obj = caesar.load(CSR_infile)
for halo_idx in halo_idxs:
    halo = obj.halos[halo_idx]
    halos.append( { "index": halo_idx,"center": halo.minpotpos,"R500": halo.virial_quantities["r500c"], "M500": halo.virial_quantities["m500c"]})    
    
    
metals = ["He_metallicity", "C_metallicity", "N_metallicity", "O_metallicity", 
          "Ne_metallicity", "Mg_metallicity", "Si_metallicity", "S_metallicity", 
          "Ca_metallicity", "Fe_metallicity"] 

obs = Observation(SNAPFILE, snapnum, save_dir = save_dir, run_ID= run_ID, emin = 0.05, emax = 10, emin_for_EW_values=weighting_e[0], emax_for_EW_values=weighting_e[1], redshift = redshift, energies_for_Lx_tot = ebins)
for instr in instruments_dict: obs.add_instrument(**instr )
obs.load_ds()
for cut in cuts_dict:obs.add_cut(**cut )
    
photon_sample_exp = 1.5*max([x["exp_time"][0] for x in instruments_dict])
for halo in halos:  
    print("Halo", halo["index"])
    obs.set_active_halos(halo)
    obs.MakePhotons(overwrite = True,metals = metals, photon_sample_exp=photon_sample_exp, area = (2.5, "m**2"), model = "CIE APEC", make_profiles=True, make_phaseplots=True) 
    obs.ObservePhotons(overwrite = True, image_energies = image_energies, instr_bkgnd = instrument_background, foreground = milky_way_foreground, ptsrc_bkgnd = point_source_background, no_dither = False,)   