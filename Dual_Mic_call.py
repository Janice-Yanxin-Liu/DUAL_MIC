import numpy as np
import pandas as pd
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from Dual_Mic_func import DUAL_MIC_SA

## -------------------------------------------------------------------------------------------------------- ##
##                            Inputs (2024-01-10 12:00:00 - 2025-04-09 18:00:00)                            ##
## -------------------------------------------------------------------------------------------------------- ##
# 1, Soil layers information
dzsoil = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])            # Unit: m

z_cum = np.cumsum(dzsoil)

z_lower = np.concatenate((np.array([0.0]), z_cum[:-1]))
z_upper = z_cum

z_center = []

for i in range(len(dzsoil)-1):
    center_i = 1/2 * dzsoil[i+1] + z_cum[i]
    z_center.append(center_i)

z_center = np.concatenate((np.array([0.1]), np.array(z_center)))         # shape(z,)

# 2, Construct WFPS (ti, zi)  np.array     Unit: unitless           2024-01-10 - 2025-04-09
WTD = pd.read_csv('/Users/y.liu13/Library/CloudStorage/OneDrive-UniversityofExeter/TropMethane/Data/'
                  'Nottingham_data/WTD/Hardwood/Hardwood_WTD_3H.csv',\
                    parse_dates = [0], index_col = [0], header = 0, dayfirst = True)          # Unit: m, >0 when below water table

WTD = WTD.loc[:'2025-01-10 06:00:00']

Dist_to_WTD = []             # > 0, WT below a layer (unsaturated layer); <= 0, WT above a layer (saturated layer).  DataFrame,shape(ti, zi)
for i in range(len(z_center)):
    Dist_to_WTD_Li = WTD - z_center[i]
    Dist_to_WTD_Li = Dist_to_WTD_Li.rename(columns = {'Average (m)': f'L{i}'})    # DataFrame, shape(t,1)
    Dist_to_WTD.append(Dist_to_WTD_Li)
Dist_to_WTD = pd.concat(Dist_to_WTD, axis = 1)  
Dist_to_WTD = Dist_to_WTD.to_numpy()            

Soil_props = pd.read_csv('/Users/y.liu13/Library/CloudStorage/OneDrive-UniversityofExeter/TropMethane/Data/'
                         'Panama/Ancillaries/Changuinola_BrooksCorey_truncated.csv',\
                            header = 0, index_col = 0)

def Brooks(x, params):       # x: the distance of layer center depth to water table location (m); params: SWRC parameters of this layer
    theta_s, theta_r, sai_b, lambda_ = params
    return np.piecewise(x, [(x <= sai_b), (x > sai_b)], [lambda x: theta_s, lambda x: theta_r + (theta_s - theta_r)*(sai_b/x)**lambda_])

SWC_inferred = []
for i in range(len(z_center)):
    Dist_i = Dist_to_WTD[:, i]           # shape(t,)
    params_i = Soil_props.iloc[i, :].values        # shape(z,)
    SWC_Li = np.where(Dist_i > 0, Brooks(Dist_i, params_i), params_i[0])                        # shape(t,)
    SWC_inferred.append(SWC_Li)

SWC_inferred = np.stack(SWC_inferred, axis = 1)                           # shape (t, z)
SWC_sat = Soil_props['theta_s (m^3/m^3)'].values                          # shape (z,)

WFPS = SWC_inferred/SWC_sat                                     # shape(t, z)

# 3, SoilT (t, z)  np.array   Unit: °C        2024-01-10 - 2025-04-09 
SoilT_obs = pd.read_csv('/Users/y.liu13/Library/CloudStorage/OneDrive-UniversityofExeter/TropMethane/Data/'
                        'Nottingham_data/tomst/Hardwood/Hardwood_soilT_3H.csv',\
                            usecols = [0, 1], parse_dates = [0], index_col = 0)

SoilT_obs = SoilT_obs.loc[:'2025-01-10 06:00:00']

SoilT = np.tile(SoilT_obs.values, (1, 6))            # numpy array

# 4, LAI (ti,) --> f_grow(ti,) np.array  2024-01-10 - 2025-04-09 
f_grow = np.ones(SoilT_obs.shape[0])

# 5, Lit_input (ti, zi)   np.array  2024-01-10 - 2025-04-09
z_root = 1.1
C_root = 1.818

def f_root(z, z_root, C_root):
    return C_root * max((z_root - z) / z_root, 0)

z_root_prop = []

for low, up in zip(z_lower, z_upper):

    z_root_prop_i = max(integrate.quad(f_root, low, up, args=(z_root, C_root))[0], 0)

    z_root_prop.append(z_root_prop_i)

z_root_prop = np.asarray(z_root_prop)    

Lit_flux = pd.read_csv('/Users/y.liu13/Library/CloudStorage/OneDrive-UniversityofExeter/TropMethane/Results/'
                       'Nottingham/JULES/sim5/2000_peatsp_litter/Litter_3h_avg_df.csv', \
                        parse_dates = [0], index_col = [0])                          # Unit: mg/m^2/h, shape(t,1)

Lit_flux = Lit_flux.loc[:'2025-01-10 06:00:00']

Lit_input = []

for i in range(len(dzsoil)):
    Lit_input.append(Lit_flux.values.flatten() * z_root_prop[i]/dzsoil[i])

Lit_input = np.stack(Lit_input, axis = 1)              # Unit: mg/m^3/h, shape(t,z)   

# All the inputs required
inputs = {

    # (1) Soil layers info: 
        'dzsoil': dzsoil,
        'z_center': z_center,
    # (2) WFPS
        'Dist_to_WTD': Dist_to_WTD,
        'SWC_sat': SWC_sat,
        'WFPS': WFPS,
    # (3) SoilT: numpy array, shape(t,z)
        'SoilT': SoilT,
    # (4) LAI --> f_grow
        'f_grow': f_grow,
    # (5) Litter input
        'z_root': z_root,
        'z_root_prop': z_root_prop,
        'Lit_input': Lit_input
}        

## -------------------------------------------------------------------------------------------------------- ##
##                                              Fixed Parameters                                            ##
## -------------------------------------------------------------------------------------------------------- ##
fixed_params = {
    # Oxygen proxy
    'k_oxy': 35,
    'W_crit_ana': 0.82,

    # Water modifier
    'k_water': 20,
    'W_crit_water': 0.45,

    # SOC_F and SOC_S Decomposition
    'Ea_decomp_SOC_F': 47,                       # kJ/mol
    'Ea_decomp_SOC_S': 47,

    # Uptake
    'Ea_uptake_aero': 47,
    'Ea_uptake_ana': 55,

    # Production of CH4 and CO2
    'CUE_aero': 0.27,                               # unitless
    'CUE_ana': 0.15,                                # unitless

    # Methanotrophy
    'Ea_oxid': 49
}

## -------------------------------------------------------------------------------------------------------- ##
##                                Valid Parameters extracted from SA results                                ##
## -------------------------------------------------------------------------------------------------------- ##
params_SA = {
    'root_oxy': 0.518,
    'V_max_decomp_SOC_F': 3.3e12,
    'K_decomp_SOC_F': 9.3e9,
    'V_max_decomp_SOC_S': 2.5e12,
    'K_decomp_SOC_S': 4.7e11,
    'V_max_uptake_aero': 5e10,
    'K_uptake_aero': 4.8e8,
    'V_max_uptake_ana': 1.6e11,
    'K_uptake_ana': 6.5e7,
    'r_EnzProd_aero': 0.00032,
    'r_EnzProd_ana': 0.00092, 
    'r_EnzLoss': 0.0836,
    'r_death_aero': 0.00048,  
    'r_death_ana': 0.00034,
    'Eta': 0.004,
    'V_max_oxid': 3.3e8,
    'K_CH4': 9816,
    'V_diff': 130,
    'lambda_ox': 0.004,
    'lambda_PMT': 0.007,
    'z_u': 3.58,
    'r_CH4': 0.18,
    'V_ebull': 0.05,
    'V_PMT': 0.03,
    'P_ox': 0.7,
    'resp': 0.65
}

# params_SA = np.array(list(params_SA.values()))

## -------------------------------------------------------------------------------------------------------- ##
##                                        Function Call & Spin                                              ##
## -------------------------------------------------------------------------------------------------------- ##
timesteps = SoilT.shape[0]
dt = 3

# Initial values
SOC_F0 = np.array([3e5, 3e5, 3e5, 3e5, 3e5, 2e5], dtype = float)
SOC_S0 = np.array([2e7, 2e7, 2e7, 2e7, 2e7, 2e7], dtype = float)                                           # mg/m^3
DOC0 = np.array([5000, 7000, 9000, 1e4, 2e4, 1e4], dtype = float)
MIC_aero0 = np.array([1e5, 8e4, 6e4, 4e4, 1e2, 1e2], dtype = float)
MIC_ana0 = np.array([1e2, 1e2, 1e2, 1e2, 5e3, 2e3], dtype = float)
Enz0 = np.array([500, 300, 200, 200, 50, 50], dtype = float)
CH40 = np.array([100, 300, 300, 700, 1000, 500], dtype = float)
CO20 = np.array([1e4, 1e4, 5e3, 5e3, 1e3, 1e3], dtype = float)

# Spin
nspin = 100

SOC_F0_spin = [SOC_F0]
SOC_S0_spin = [SOC_S0]                                             
DOC0_spin = [DOC0]
MIC_aero0_spin = [MIC_aero0]
MIC_ana0_spin = [MIC_ana0]
Enz0_spin = [Enz0]
CH40_spin = [CH40]
CO20_spin = [CO20]

for i in range(nspin):

    SVs, Fluxes = DUAL_MIC_SA(timesteps, dt, inputs, \
                           SOC_F0_spin[-1], SOC_S0_spin[-1], DOC0_spin[-1], MIC_aero0_spin[-1], MIC_ana0_spin[-1], Enz0_spin[-1], CH40_spin[-1], CO20_spin[-1],\
                           fixed_params, params_SA)
                                                                         
    SOC_F0_spin.append(SVs[0][-1, :])
    SOC_S0_spin.append(SVs[1][-1, :])                                           
    DOC0_spin.append(SVs[2][-1, :]) 
    MIC_aero0_spin.append(SVs[3][-1, :]) 
    MIC_ana0_spin.append(SVs[4][-1, :]) 
    Enz0_spin.append(SVs[5][-1, :]) 
    CH40_spin.append(SVs[6][-1, :]) 
    CO20_spin.append(SVs[7][-1, :]) 

SOC_F0_spin = np.stack(SOC_F0_spin, axis = 0)                # Shape(nspin, z)
SOC_S0_spin = np.stack(SOC_S0_spin, axis = 0)                                       
DOC0_spin = np.stack(DOC0_spin, axis = 0)
MIC_aero0_spin = np.stack(MIC_aero0_spin, axis = 0)
MIC_ana0_spin = np.stack(MIC_ana0_spin, axis = 0)
Enz0_spin = np.stack(Enz0_spin, axis = 0)
CH40_spin = np.stack(CH40_spin, axis = 0)
CO20_spin = np.stack(CO20_spin, axis = 0)

# DEBUG
SVs, Fluxes = DUAL_MIC_SA(timesteps, dt, inputs, \
                        SOC_F0_spin[-1], SOC_S0_spin[-1], DOC0_spin[-1], MIC_aero0_spin[-1], MIC_ana0_spin[-1], Enz0_spin[-1], CH40_spin[-1], CO20_spin[-1],\
                        fixed_params, params_SA)

# ------ Plot S.V.s ------ #
# SOC_F
fig1, ax1 = plt.subplots()

for i in range(len(z_center)):
    ax1.plot(
        np.arange(nspin + 1),
        SOC_F0_spin[:, i],
        'o-',
        markersize=2,
        label=f'Layer {i}'
    )

ax1.set_xlabel('number of spins')
ax1.set_ylabel('mg/m^3')
ax1.legend()
ax1.set_title('SOC_F spin-up')

plt.show()

# SOC_S
fig2, ax2 = plt.subplots()

for i in range(len(z_center)):
    ax2.plot(
        np.arange(nspin + 1),
        SOC_S0_spin[:, i],
        'o-',
        markersize=2,
        label=f'Layer {i}'
    )

ax2.set_xlabel('number of spins')
ax2.set_ylabel('mg/m^3')
ax2.legend()
ax2.set_title('SOC_S spin-up')

plt.show()

# DOC
fig3, ax3 = plt.subplots()

for i in range(len(z_center)):
    ax3.plot(
        np.arange(nspin + 1),
        DOC0_spin[:, i],
        'o-',
        markersize=2,
        label=f'Layer {i}'
    )

ax3.set_xlabel('number of spins')
ax3.set_ylabel('mg/m^3')
ax3.legend()
ax3.set_title('DOC spin-up')

plt.show()

# MIC_aero
fig4, ax4 = plt.subplots()

for i in range(len(z_center)):
    ax4.plot(
        np.arange(nspin + 1),
        MIC_aero0_spin[:, i],
        'o-',
        markersize=2,
        label=f'Layer {i}'
    )

ax4.set_xlabel('number of spins')
ax4.set_ylabel('mg/m^3')
ax4.legend()
ax4.set_title('MIC_aero spin-up')

plt.show()

# MIC_ana
fig5, ax5 = plt.subplots()

for i in range(len(z_center)):
    ax5.plot(
        np.arange(nspin + 1),
        MIC_ana0_spin[:, i],
        'o-',
        markersize=2,
        label=f'Layer {i}'
    )

ax5.set_xlabel('number of spins')
ax5.set_ylabel('mg/m^3')
ax5.legend()
ax5.set_title('MIC_ana spin-up')

plt.show()

# Enz
fig6, ax6 = plt.subplots()

for i in range(len(z_center)):
    ax6.plot(
        np.arange(nspin + 1),
        Enz0_spin[:, i],
        'o-',
        markersize=2,
        label=f'Layer {i}'
    )

ax6.set_xlabel('number of spins')
ax6.set_ylabel('mg/m^3')
ax6.legend()
ax6.set_title('Enz spin-up')

plt.show()

# CH4
fig7, ax7 = plt.subplots()

for i in range(len(z_center)):
    ax7.plot(
        np.arange(nspin + 1),
        CH40_spin[:, i],
        'o-',
        markersize=2,
        label=f'Layer {i}'
    )

ax7.set_xlabel('number of spins')
ax7.set_ylabel('mg/m^3')
ax7.legend()
ax7.set_title('CH4 spin-up')

plt.show()

# CO2
fig8, ax8 = plt.subplots()

for i in range(len(z_center)):
    ax8.plot(
        np.arange(nspin + 1),
        CO20_spin[:, i],
        'o-',
        markersize=2,
        label=f'Layer {i}'
    )

ax8.set_xlabel('number of spins')
ax8.set_ylabel('mg/m^3')
ax8.legend()
ax8.set_title('CO2 spin-up')

plt.show()


# # ------ Plot fluxes ------ #
# # CH4
F_soil_CH4 = Fluxes[0]
F_soil_CO2 = Fluxes[2]

plt.figure(0)
plt.plot(SoilT_obs.index.values, F_soil_CH4)
plt.xlabel('time')
plt.ylabel('mg/m^2/h')
plt.title('CH4 flux')
plt.show()

# CO2
plt.figure(1)
plt.plot(SoilT_obs.index.values, F_soil_CO2)
plt.xlabel('time')
plt.ylabel('mg/m^2/h')
plt.title('CO2 flux')
plt.show()

F_plant_CH4 = Fluxes[1]
F_plant_CO2 = Fluxes[3]

plt.figure(2)
plt.plot(SoilT_obs.index.values, F_plant_CH4)
plt.xlabel('time')
plt.ylabel('mg/m^2/h')
plt.title('CH4 flux tree')
plt.show()

# CO2
plt.figure(3)
plt.plot(SoilT_obs.index.values, F_plant_CO2)
plt.xlabel('time')
plt.ylabel('mg/m^2/h')
plt.title('CO2 flux tree')
plt.show()
# ## -------------------------------------------------------------------------------------------------------- ##
# ##                                              Observations                                            ##
# ## -------------------------------------------------------------------------------------------------------- ##
# SOC = np.array([38.4, 16.8, 20.4, 21, 20.8, 19.6]) * 1e6

SWC_obs = pd.read_csv('/Users/y.liu13/Library/CloudStorage/OneDrive-UniversityofExeter/TropMethane/Data/Nottingham_data/tomst/'
                      'Hardwood/Hardwood_avg_SWC_6H.csv', header = 0, index_col = 0, parse_dates = [0])

plt.figure()
plt.plot(SWC_obs.index, SWC_obs.values * 0.01, label = 'observations')
plt.plot(SWC_obs.index, SWC_inferred[:, 0], label = 'inferred')
plt.legend()
plt.show()