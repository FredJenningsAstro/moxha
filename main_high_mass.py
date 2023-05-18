from moxha.observations import Observation
from mpi4py import MPI
import unyt
import caesar



redshift = 0.035


snapnum = 151
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

halo_idxs = [793,]
halo_idxs = np.array_split(halo_idxs,size)[rank]




ebins = [[0.2,3.0],[0.5,2.0],[0.5,2.4]]
image_energies = []
for ebin_min, ebin_max in ebins:
    image_energies.append({'name':f"{ebin_min}_{ebin_max}_keV",'emin': ebin_min , 'emax':ebin_max})


print("rank:", rank, ", halo_idxs:", halo_idxs)

halos = []
CSR_infile = f"/home/b/babul/rennehan/project/simba_data/caesar_{str(snapnum).zfill(3)}.hdf5"
SNAPFILE   = f"/home/b/babul/rennehan/project/simba_data/snapshot_{str(snapnum).zfill(3)}.hdf5"

obj = caesar.load(CSR_infile)
for halo_idx in halo_idxs:
    halo = obj.halos[halo_idx]
    halos.append( { "index": halo_idx,"center": halo.minpotpos,"R500": halo.virial_quantities["r500c"]})    
  
    
    

metals = ["He_metallicity", "C_metallicity", "N_metallicity", "O_metallicity", 
          "Ne_metallicity", "Mg_metallicity", "Si_metallicity", "S_metallicity", 
          "Ca_metallicity", "Fe_metallicity"] 
### Simba pressurization density = 0.13 cm**-3
cuts_dict = [{'field':('gas','H_nuclei_density'), 'less_than':(0.1)*unyt.cm**-3},
            {'field':("gas", "temperature"), 'gtr_than': 2e5*unyt.K},
            {'field':("PartType0", 'H_nuclei_density'), 'less_than':(0.1)*unyt.cm**-3},
            {'field':("PartType0", "temperature"), 'gtr_than': 2e5*unyt.K},
            {'field':("PartType0", "DelayTime"), 'equals':0},
            {'field':("PartType0", "StarFormationRate"), 'equals':0}]


instruments_dict = [ {"Name":"lem_2.0eV","ID":"lem_2eV_1Ms", "exp_time": (1000,'ks'), "reblock":1},]
obs = Observation(SNAPFILE, snapnum, save_dir = "./08052023_LEM/", run_ID= f"m100s50", emin = 0.05, emax = 10, emin_for_EW_values=0.2, emax_for_EW_values=3.0, redshift = redshift)
for instr in instruments_dict: obs.add_instrument(**instr )
obs.load_ds()
for cut in cuts_dict:obs.add_cut(**cut )
    
    
    
for halo in halos:  
    print("Halo", halo["index"])
    obs.set_active_halos(halo)
    obs.MakePhotons(metals = metals, photon_sample_exp=2000, area = (2.5, "m**2"), model = "IGM for Gerrit 18052023", make_profiles = False) 
    obs.ObservePhotons(overwrite = True, image_energies = image_energies, instr_bkgnd = False, foreground = False, ptsrc_bkgnd = False, no_dither = False,)    
    
    
#    obs.BlankSkyBackgrounds(1)
    # pp = PostProcess(obs)
    # pp.clear_instruments()
    # pp.add_instrument(**instruments_dict[0])    
    # pp.generate_annuli(S2N_threshold = 200**0.5, S2N_range = (1,2), S2N_good_fraction = 0.7, overwrite = False)
    # sf = FitRadialSpectra(obs)
    # sf.fit_spectra(fit_emin = 0.1, fit_emax = 2, n_max = 100, overwrite = False)    
    # vt = VoronoiTesselation(obs, reblock = 1, median_cleaning_thresh = 70, needs_cleaning = False, needs_tesselating=False, needs_cutouts=False)

    
    
    
    
    
    
    
    
    
    
    
    
