import numpy as np
# ---- Inputs ---- #
# (1) Soil layers info: 
#     dzsoil: numpy array, shape (z,)
#     z_center: numpy array, shape (z,)
# (2) WFPS
#     Dist_to_WTD: numpy array, shape(t,z)
#     SWC_sat: numpy array, shape (z,)
#     WFPS: numpy array, shape(t,z)
# (3) SoilT: numpy array, shape(t,z)
# (4) LAI --> f_grow
#     f_grow: numpy array, shape (t,)
# (5) Litter input
#     z_root: constant
#     z_root_prop: numpy array, shape (z,)
#     Lit_input: numpy array, shape (t,z)

# -------------------------- ##
#          Constants         ##
# -------------------------- ##
D_CH4_0 = 1.952
D_CO2_0 = 1.381
R = 0.008314                                  # kJ/mol/K
P_atm = 101325                                # Pa
rho_water = 1000                              # kg/m^3
g = 9.8                                       # m/s^2
Henry_CH4_cp_298K = 1.4e-5                    # mol/m^3/Pa
M_C = 1.2e4                                   # mg/mol

def DUAL_MIC_SA(timesteps, dt, inputs, \
             SOC_F0, SOC_S0, DOC0, MIC_aero0, MIC_ana0, Enz0, CH40, CO20, \
                fixed_params, params_SA):

    # ---------------------------------------------------------------- ##
    #                               Inputs                             ##
    # ---------------------------------------------------------------- ##
    dzsoil = inputs['dzsoil']
    z_center = inputs['z_center']
    Dist_to_WTD = inputs['Dist_to_WTD']
    SWC_sat = inputs['SWC_sat']
    WFPS = inputs['WFPS']
    SoilT = inputs['SoilT']
    f_grow = inputs['f_grow']
    z_root = inputs['z_root']
    z_root_prop = inputs['z_root_prop']
    Lit_input = inputs['Lit_input']

    w_g = WFPS * SWC_sat       # volumeteric water content (m^3 water/m^3 soil)  shape(t,z)
    v_a = (1 - WFPS) * SWC_sat        # air-filled pore space (m^3 air/m^3 soil)   shape(t,z)

    # Fixed parameters
    k_oxy = fixed_params['k_oxy'] 
    W_crit_ana = fixed_params['W_crit_ana'] 
    k_water = fixed_params['k_water']
    W_crit_water = fixed_params['W_crit_water']
    Ea_decomp_SOC_F = fixed_params['Ea_decomp_SOC_F']   
    Ea_decomp_SOC_S = fixed_params['Ea_decomp_SOC_S']
    Ea_uptake_aero = fixed_params['Ea_uptake_aero']
    Ea_uptake_ana = fixed_params['Ea_uptake_ana']
    CUE_aero = fixed_params['CUE_aero']                
    CUE_ana = fixed_params['CUE_ana']
    Ea_oxid = fixed_params['Ea_oxid']

    # SA Parameters
    root_oxy = params_SA['root_oxy']
    V_max_decomp_SOC_F = params_SA['V_max_decomp_SOC_F']
    K_decomp_SOC_F = params_SA['K_decomp_SOC_F']
    V_max_decomp_SOC_S = params_SA['V_max_decomp_SOC_S'] 
    K_decomp_SOC_S = params_SA['K_decomp_SOC_S']
    V_max_uptake_aero = params_SA['V_max_uptake_aero'] 
    K_uptake_aero = params_SA['K_uptake_aero']
    V_max_uptake_ana = params_SA['V_max_uptake_ana'] 
    K_uptake_ana = params_SA['K_uptake_ana']
    r_EnzProd_aero = params_SA['r_EnzProd_aero'] 
    r_EnzProd_ana = params_SA['r_EnzProd_ana'] 
    r_EnzLoss = params_SA['r_EnzLoss']
    r_death_aero = params_SA['r_death_aero'] 
    r_death_ana = params_SA['r_death_ana']
    Eta = params_SA['Eta']
    V_max_oxid = params_SA['V_max_oxid'] 
    K_CH4 = params_SA['K_CH4']
    V_diff = params_SA['V_diff']
    lambda_ox = params_SA['lambda_ox']
    lambda_PMT = params_SA['lambda_PMT']
    z_u = params_SA['z_u']
    r_CH4 = params_SA['r_CH4']
    V_ebull = params_SA['V_ebull']
    V_PMT = params_SA['V_PMT'] 
    P_ox = params_SA['P_ox']
    resp = params_SA['resp']
    
    # -------------------------------------------------------------------------------------------- ##
    #                   Pre-calculated Modifier Terms       shape(t, z)                            ##
    # -------------------------------------------------------------------------------------------- ##
    # (1) SWC modifier
    g_WFPS = 1/(1 + np.exp(- k_water * (WFPS - W_crit_water)))       

    # (2) Oxygen proxy
    f_ana_WFPS = 1/(1 + np.exp(- k_oxy * (WFPS - W_crit_ana)))       
    Chi_oxy_bulk = 1 - f_ana_WFPS                                          

    oxy_eff = Chi_oxy_bulk + root_oxy * z_root_prop/dzsoil * f_ana_WFPS  
    oxy_eff = np.clip(oxy_eff, 0, 1)          

    # (3) Diffusivity coefficient
    diff_g_WFPS = 1e-4 + (1 - 1e-4) * np.exp(-12 * WFPS**0.4)    
    # (3.1) CH4
    D_a_T_CH4 = D_CH4_0 * ((SoilT + 273.15)/273.15)**1.81
    tilde_D_CH4 = D_a_T_CH4 * diff_g_WFPS
    Eff_D_CH4 = (np.cumsum(dzsoil).reshape(1,-1)/np.cumsum(dzsoil.reshape(1,-1)/tilde_D_CH4, axis=1))
    # (3.2) CO2
    D_a_T_CO2 = D_CO2_0 * ((SoilT + 273.15)/273.15)**1.81
    tilde_D_CO2 = D_a_T_CO2 * diff_g_WFPS
    Eff_D_CO2 = (np.cumsum(dzsoil).reshape(1,-1)/np.cumsum(dzsoil.reshape(1,-1)/tilde_D_CO2, axis=1))

    # -------------------------------------------------------------------------------------------- ##
    #                                    Other prescribed terms                                    ##
    # -------------------------------------------------------------------------------------------- ##
    Henry_CH4_cp = Henry_CH4_cp_298K * np.exp(1600 * (1/(SoilT + 273.15) - 1/298.15))     # shape(t, z)     
    Henry_CH4_cc = Henry_CH4_cp * R * 1000 * (SoilT + 273.15)                             # shape(t, z)

    # Distribution of 'Q_diff_CH4_PMT_oxid' in layers above
    Q_diff_PMT_Oxid_distri_matrix_ti = []

    for zi in range(len(z_center)): 
        if zi == 0:
            Q_diff_PMT_Oxid_distri = np.zeros(len(z_center))
        else:
            Q_diff_PMT_Oxid_distri = np.concatenate((z_root_prop[:zi]/np.sum(z_root_prop[:zi]), np.zeros(len(z_center) - zi)))

        Q_diff_PMT_Oxid_distri_matrix_ti.append(Q_diff_PMT_Oxid_distri)      

    Q_diff_PMT_Oxid_distri_matrix_ti = np.stack(Q_diff_PMT_Oxid_distri_matrix_ti, axis = 0) 

    # ---------------------------------------------------------------------------------------------- ##
    #              Tempaerature-dependant Kinetic Rates       shape(t, z)                            ##
    # ---------------------------------------------------------------------------------------------- ##
    V_decomp_SOC_F = V_max_decomp_SOC_F * np.exp(- Ea_decomp_SOC_F/(R * (SoilT + 273.15)))
    V_decomp_SOC_S = V_max_decomp_SOC_S * np.exp(- Ea_decomp_SOC_S/(R * (SoilT + 273.15)))

    V_uptake_aero = V_max_uptake_aero * np.exp(- Ea_uptake_aero/(R * (SoilT + 273.15)))
    V_uptake_ana = V_max_uptake_ana * np.exp(- Ea_uptake_ana/(R * (SoilT + 273.15)))

    V_oxid = V_max_oxid * np.exp(- Ea_oxid/(R * (SoilT + 273.15)))

    # ---------------------------------------------------------------------------------------------- ##
    #                              Ebullition threshold       shape(t, z)                            ##
    # ---------------------------------------------------------------------------------------------- ##
    water_column_above = np.where(Dist_to_WTD < 0, - Dist_to_WTD, 0)                        # shape(t,z), m
    P_thresh = (P_atm + rho_water * g * water_column_above) * r_CH4                         # shape(t,z), Pa

    CH4_thresh = P_thresh * Henry_CH4_cp                                                          # unit: mol CH4/m^3 water
    C_thresh = CH4_thresh * M_C                                                                   # unit: mg C/m^3 water

    # ---------------------------------------------------------------------------------------------- ##
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    #                                    Interate over timesteps                                     ##
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    # ---------------------------------------------------------------------------------------------- ##
    
    # State Variable Initial Values
    SOC_F = [SOC_F0]
    SOC_S = [SOC_S0]
    DOC = [DOC0]
    MIC_aero = [MIC_aero0]
    MIC_ana = [MIC_ana0]
    Enz = [Enz0]
    CH4 = [CH40]
    CO2 = [CO20]

    F_soil_CH4 = []
    F_plant_CH4 = []
    F_soil_CO2 = []
    F_plant_CO2 = []

    for ti in range(timesteps):

        # Lists to store Rates
        Decomp_SOC_F_ti = []
        Decomp_SOC_S_ti = [] 
        Uptake_aero_ti = []
        Uptake_ana_ti = []
        EnzProd_aero_ti = []
        EnzProd_ana_ti = []
        EnzLoss_ti = []
        Death_aero_ti = []
        Death_ana_ti = []

        Prod_CH4_ti = []
        Oxid_CH4_ti = []
        Q_diff_CH4_oxid_ti =[]
        Q_diff_CH4_PMT_oxid_ti = []
        Q_diff_CH4_PMT_emit_ti = []
        Q_diff_CH4_emit_ti = []
        Q_ebull_CH4_ti = []
        Q_PMT_CH4_oxid_ti = []
        Q_PMT_CH4_emit_ti = []

        Prod_CO2_aero_ti = []
        Prod_CO2_ana_ti = []
        Q_diff_CO2_ti = []
        Q_PMT_CO2_ti = []

        # ----------------------------------- ##
        #       Bulk Oxidation in Layers      ##
        # ----------------------------------- ##
        Bulk_oxid_in_layers_ti = (Eta * MIC_aero[-1]) * oxy_eff[ti] * dzsoil             # numpy array, shape(z,)
        
        # Distribution of oxidised CH4 diffusion flux in layers above
        Q_diff_Oxid_distri_matrix_ti = []

        for zi in range(len(z_center)): 
            if zi == 0:
                Oxid_distri = np.zeros(len(z_center))
            else:
                Oxid_distri = np.concatenate((Bulk_oxid_in_layers_ti[:zi]/np.sum(Bulk_oxid_in_layers_ti[:zi]), np.zeros(len(z_center) - zi)))

            Q_diff_Oxid_distri_matrix_ti.append(Oxid_distri)      

        Q_diff_Oxid_distri_matrix_ti = np.stack(Q_diff_Oxid_distri_matrix_ti, axis = 0)     

        # ----------------------------------- ##
        #            PMT in Layers            ##
        # ----------------------------------- ##
        S_sPMT_ti = np.concatenate(([1], np.exp(- lambda_PMT * np.cumsum(z_root_prop * f_grow[ti]))[:-1]))    # shape(z,)

        # ---------------------------- ##
        #     Iterate over layers      ##
        # ---------------------------- ##
        for zi in range(len(z_center)):

            CH4_w = CH4[-1]/(w_g[ti] + v_a[ti]/Henry_CH4_cc[ti])       #shape(z,)    mgC/m^3 water

            # -------------------------- ##
            #            Rates           ##
            # -------------------------- ##
            
            # ---------------------------------- CH4 and CO2 productions --------------------------------------- #
            # Decomposition
            Decomp_SOC_F_zi = V_decomp_SOC_F[ti, zi] * Enz[-1][zi] * SOC_F[-1][zi]/(K_decomp_SOC_F + SOC_F[-1][zi]) * g_WFPS[ti, zi]
            
            Decomp_SOC_S_zi = V_decomp_SOC_S[ti, zi] * Enz[-1][zi] * SOC_S[-1][zi]/(K_decomp_SOC_S + SOC_S[-1][zi]) * g_WFPS[ti, zi]

            # Uptake
            Uptake_aero_zi = V_uptake_aero[ti, zi] * MIC_aero[-1][zi] * DOC[-1][zi]/(K_uptake_aero + DOC[-1][zi]) * oxy_eff[ti, zi] * g_WFPS[ti, zi]
            
            Uptake_ana_zi = V_uptake_ana[ti, zi] * MIC_ana[-1][zi] * DOC[-1][zi]/(K_uptake_ana + DOC[-1][zi]) * (1 - oxy_eff[ti, zi]) * g_WFPS[ti, zi]

            # Production of CH4 and CO2
            Prod_CO2_aero_zi = Uptake_aero_zi * (1 - CUE_aero)
            Prod_CO2_ana_zi = Uptake_ana_zi * (1 - CUE_ana) * 1/2
            
            Prod_CH4_zi = Uptake_ana_zi * (1 - CUE_ana) * 1/2

            # Enzyme production
            EnzProd_aero_zi = r_EnzProd_aero * MIC_aero[-1][zi]

            EnzProd_ana_zi = r_EnzProd_ana * MIC_ana[-1][zi]

            # Enzyme turnover
            EnzLoss_zi = r_EnzLoss * Enz[-1][zi]

            # Microbial Death
            Death_aero_zi = r_death_aero * MIC_aero[-1][zi]
            Death_ana_zi = r_death_ana * MIC_ana[-1][zi]

            # ----------------------- CH4  Utilisation and Transport Processes -------------------------- #
            # Methanotrophy
            MIC_meth_zi = Eta * MIC_aero[-1][zi]

            Oxid_CH4_zi = V_oxid[ti, zi] * MIC_meth_zi * CH4_w[zi]/(K_CH4 + CH4_w[zi]) * oxy_eff[ti, zi]

            # Diffusion
            Q_diff_CH4_zi = V_diff * Eff_D_CH4[ti, zi]/((z_center[zi] + z_u)/(z_center[0] + z_u))**2 * CH4[-1][zi]

            if zi == 0:
                S_sOX_zi = 1.0
            else:
                S_sOX_zi = np.exp(- lambda_ox * np.sum(Bulk_oxid_in_layers_ti[:zi]))

            Q_diff_CH4_oxid_zi = Q_diff_CH4_zi * (1 - S_sOX_zi)

            Q_diff_CH4_PMT_zi = Q_diff_CH4_zi * S_sOX_zi * (1 - S_sPMT_ti[zi])
            Q_diff_CH4_PMT_oxid_zi = Q_diff_CH4_PMT_zi * P_ox
            Q_diff_CH4_PMT_emit_zi = Q_diff_CH4_PMT_zi * (1 - P_ox)

            Q_diff_CH4_emit_zi = Q_diff_CH4_zi * S_sOX_zi * S_sPMT_ti[zi]

            # Ebullition
            if WFPS[ti, zi] >= 0.9 and CH4_w[zi] >= C_thresh[ti, zi]:                                                                                                                                     
                
                Q_ebull_CH4_zi = V_ebull * (CH4_w[zi] - C_thresh[ti, zi]) * w_g[ti, zi]               # unit: mg C/m^3 soil/h 
            
            else:
                
                Q_ebull_CH4_zi = 0.0

            # PMT (CH4)
            Q_PMT_CH4_zi = V_PMT * CH4[-1][zi] * z_root_prop[zi] * f_grow[ti]

            Q_PMT_CH4_oxid_zi = Q_PMT_CH4_zi * P_ox
            Q_PMT_CH4_emit_zi = Q_PMT_CH4_zi * (1 - P_ox)

            # ----------------------- CO2  Utilisation and Transport Processes -------------------------- #
            # Diffusion
            Q_diff_CO2_zi = V_diff * Eff_D_CO2[ti, zi]/((z_center[zi] + z_u)/(z_center[0] + z_u))**2 * CO2[-1][zi]

            # PMT (CO2)
            Q_PMT_CO2_zi = V_PMT * CO2[-1][zi] * z_root_prop[zi] * f_grow[ti]

            # ---------------------------- ##
            #     Store Rates in layers    ##
            # ---------------------------- ##
            Decomp_SOC_F_ti.append(Decomp_SOC_F_zi)
            Decomp_SOC_S_ti.append(Decomp_SOC_S_zi)
            Uptake_aero_ti.append(Uptake_aero_zi)
            Uptake_ana_ti.append(Uptake_ana_zi)
            EnzProd_aero_ti.append(EnzProd_aero_zi)
            EnzProd_ana_ti.append(EnzProd_ana_zi)
            EnzLoss_ti.append(EnzLoss_zi)
            Death_aero_ti.append(Death_aero_zi)
            Death_ana_ti.append(Death_ana_zi)
            
            Prod_CH4_ti.append(Prod_CH4_zi)
            Oxid_CH4_ti.append(Oxid_CH4_zi)
            Q_diff_CH4_oxid_ti.append(Q_diff_CH4_oxid_zi)
            Q_diff_CH4_PMT_oxid_ti.append(Q_diff_CH4_PMT_oxid_zi)
            Q_diff_CH4_PMT_emit_ti.append(Q_diff_CH4_PMT_emit_zi)
            Q_diff_CH4_emit_ti.append(Q_diff_CH4_emit_zi)
            Q_ebull_CH4_ti.append(Q_ebull_CH4_zi)
            Q_PMT_CH4_oxid_ti.append(Q_PMT_CH4_oxid_zi)    
            Q_PMT_CH4_emit_ti.append(Q_PMT_CH4_emit_zi)     

            Prod_CO2_aero_ti.append(Prod_CO2_aero_zi)
            Prod_CO2_ana_ti.append(Prod_CO2_ana_zi)
            Q_diff_CO2_ti.append(Q_diff_CO2_zi)             
            Q_PMT_CO2_ti.append(Q_PMT_CO2_zi)               

        # ------------------------------------------------------------------------------------- #
        #                         Finished interate over all the layers                         #
        #           Convert all the filled 'Rates' lists to numpy for math operation            #
        # ------------------------------------------------------------------------------------- #
        Decomp_SOC_F_ti = np.asarray(Decomp_SOC_F_ti)           #  shape(z,)    
        Decomp_SOC_S_ti = np.asarray(Decomp_SOC_S_ti)
        Uptake_aero_ti = np.asarray(Uptake_aero_ti)
        Uptake_ana_ti = np.asarray(Uptake_ana_ti)
        EnzProd_aero_ti = np.asarray(EnzProd_aero_ti)
        EnzProd_ana_ti = np.asarray(EnzProd_ana_ti)
        EnzLoss_ti = np.asarray(EnzLoss_ti)
        Death_aero_ti = np.asarray(Death_aero_ti)
        Death_ana_ti = np.asarray(Death_ana_ti)
            
        Prod_CH4_ti = np.asarray(Prod_CH4_ti)
        Oxid_CH4_ti = np.asarray(Oxid_CH4_ti)
        Q_diff_CH4_oxid_ti = np.asarray(Q_diff_CH4_oxid_ti)
        Q_diff_CH4_PMT_oxid_ti = np.asarray(Q_diff_CH4_PMT_oxid_ti)
        Q_diff_CH4_PMT_emit_ti = np.asarray(Q_diff_CH4_PMT_emit_ti)
        Q_diff_CH4_emit_ti = np.asarray(Q_diff_CH4_emit_ti)
        Q_ebull_CH4_ti = np.asarray(Q_ebull_CH4_ti)
        Q_PMT_CH4_oxid_ti = np.asarray(Q_PMT_CH4_oxid_ti)
        Q_PMT_CH4_emit_ti = np.asarray(Q_PMT_CH4_emit_ti)

        Prod_CO2_aero_ti = np.asarray(Prod_CO2_aero_ti)
        Prod_CO2_ana_ti = np.asarray(Prod_CO2_ana_ti)
        Q_diff_CO2_ti = np.asarray(Q_diff_CO2_ti)
        Q_PMT_CO2_ti = np.asarray(Q_PMT_CO2_ti)

        # --------------------------------------------------------------- #
        #      Fluxes at timestep ti   single value, unit: mgC/m^2/h      #
        # --------------------------------------------------------------- #
        # CH4       
        F_diff_CH4_emit_ti = np.sum(Q_diff_CH4_emit_ti * dzsoil)
        F_ebull_CH4_ti = np.sum(Q_ebull_CH4_ti * dzsoil)
        
        if WFPS[ti, 0] >= 0.9:     # if 0th layer WFPS >= 0.9
            F_soil_CH4_ti = F_diff_CH4_emit_ti + F_ebull_CH4_ti
        else:
            F_soil_CH4_ti = F_diff_CH4_emit_ti

        F_plant_CH4_ti =  np.sum(Q_PMT_CH4_emit_ti * dzsoil)   

        # CO2
        F_soil_CO2_ti = np.sum(Q_diff_CO2_ti * dzsoil)
        F_plant_CO2_ti = np.sum(Q_PMT_CO2_ti * dzsoil)

        # ------------------------------------------------------------------------- ##
        #                                   ODE                                     ##
        # ------------------------------------------------------------------------- ##
        dSOC_F_dt = Lit_input[ti] - Decomp_SOC_F_ti                 # numpy array, shape(z,)
        dSOC_S_dt = (1 - resp) * Decomp_SOC_F_ti - Decomp_SOC_S_ti + Death_aero_ti + Death_ana_ti          # List, with z elements

        dDOC_dt = resp * Decomp_SOC_F_ti + Decomp_SOC_S_ti - Uptake_aero_ti - Uptake_ana_ti + EnzLoss_ti

        dMIC_aero_dt = Uptake_aero_ti * CUE_aero - Death_aero_ti - EnzProd_aero_ti
        dMIC_ana_dt = Uptake_ana_ti * CUE_ana - Death_ana_ti - EnzProd_ana_ti
        
        dEnz_dt = EnzProd_aero_ti + EnzProd_ana_ti - EnzLoss_ti

        # CH4
        dCH4_dt = Prod_CH4_ti - Oxid_CH4_ti \
                    - (Q_diff_CH4_oxid_ti + (Q_diff_CH4_PMT_oxid_ti + Q_diff_CH4_PMT_emit_ti) + Q_diff_CH4_emit_ti) \
                        - Q_ebull_CH4_ti \
                            - (Q_PMT_CH4_oxid_ti + Q_PMT_CH4_emit_ti)

        # CO2
        Q_diff_CH4_oxid_to_CO2 = Q_diff_CH4_oxid_ti[np.newaxis, :] @ Q_diff_Oxid_distri_matrix_ti            # numpy array, shape(1,z)
        Q_diff_CH4_PMT_oxid_to_CO2 = Q_diff_CH4_PMT_oxid_ti[np.newaxis, :] @ Q_diff_PMT_Oxid_distri_matrix_ti
        
        dCO2_dt = Prod_CO2_aero_ti + Prod_CO2_ana_ti + Oxid_CH4_ti \
                    + Q_diff_CH4_oxid_to_CO2 \
                        + Q_diff_CH4_PMT_oxid_to_CO2 \
                            + Q_PMT_CH4_oxid_ti \
                                - Q_diff_CO2_ti \
                                    - Q_PMT_CO2_ti                 # numpy array, shape(1,z)
        
        dCO2_dt = dCO2_dt.flatten()            # numpy array, shape(z,)

        # ------------------------------------------------------------------------ ##
        #                         State Variable Update                            ##
        # ------------------------------------------------------------------------ ##
        SOC_F_ti1 = SOC_F[-1] + dt * dSOC_F_dt      # numpy array, shape(z,)
        SOC_S_ti1 = SOC_S[-1] + dt * dSOC_S_dt      
        DOC_ti1 = DOC[-1] + dt * dDOC_dt
        MIC_aero_ti1 = MIC_aero[-1] + dt * dMIC_aero_dt
        MIC_ana_ti1 = MIC_ana[-1] + dt * dMIC_ana_dt
        Enz_ti1 = Enz[-1] + dt * dEnz_dt

        CH4_ti1 = CH4[-1] + dt * dCH4_dt

        if WFPS[ti, 0] < 0.9:     # 0th layer is not saturated layer
            Ebull_receive_index = len(z_center) - (np.sum(WFPS[ti] >= 0.9) + 1)
            CH4_ti1[Ebull_receive_index] = CH4_ti1[Ebull_receive_index] + F_ebull_CH4_ti/dzsoil[Ebull_receive_index]

        CO2_ti1 = CO2[-1] + dt * dCO2_dt            # numpy array, shape(z,)
        
        # Constrain the minimum and maximum values of S.V.s
        SOC_F_ti1 = np.clip(SOC_F_ti1, 1e-6, 1e9)   # numpy array, shape(z,)
        SOC_S_ti1 = np.clip(SOC_S_ti1, 1e-6, 1e11)    
        DOC_ti1 = np.clip(DOC_ti1, 1e-6, 1e9)
        MIC_aero_ti1 = np.clip(MIC_aero_ti1, 1e-6, 1e9)
        MIC_ana_ti1 = np.clip(MIC_ana_ti1, 1e-6, 1e9)
        Enz_ti1 = np.clip(Enz_ti1, 1e-6, 1e9)
        CH4_ti1 = np.clip(CH4_ti1, 1e-6, 1e9)
        CO2_ti1 = np.clip(CO2_ti1, 1e-6, 1e9)

        # -------------------------------------------------------- #
        #         Append State Variable values at t_i+1            #
        # -------------------------------------------------------- #
        SOC_F.append(SOC_F_ti1) 
        SOC_S.append(SOC_S_ti1)
        DOC.append(DOC_ti1)
        MIC_aero.append(MIC_aero_ti1)
        MIC_ana.append(MIC_ana_ti1)
        Enz.append(Enz_ti1)
        CH4.append(CH4_ti1)
        CO2.append(CO2_ti1)

        # -------------------------------------------------------- #
        #                  Append Fluxes from ti                   #
        # -------------------------------------------------------- #
        F_soil_CH4.append(F_soil_CH4_ti)
        F_plant_CH4.append(F_plant_CH4_ti)
        F_soil_CO2.append(F_soil_CO2_ti)
        F_plant_CO2.append(F_plant_CO2_ti)        

    # ---------------------------------------------------------------------------------------------- ##
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    #                                S.V. values over all timesteps                                  ##
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
    # ---------------------------------------------------------------------------------------------- ##
    SOC_F = np.stack(SOC_F, axis = 0)      # shape (t+1, z)
    SOC_S = np.stack(SOC_S, axis = 0) 
    DOC = np.stack(DOC, axis = 0) 
    MIC_aero = np.stack(MIC_aero, axis = 0) 
    MIC_ana = np.stack(MIC_ana, axis = 0) 
    Enz = np.stack(Enz, axis = 0) 
    CH4 = np.stack(CH4, axis = 0) 
    CO2 = np.stack(CO2, axis = 0) 

    F_soil_CH4 = np.asarray(F_soil_CH4)         # shape(t,)
    F_plant_CH4 = np.asarray(F_plant_CH4)
    F_soil_CO2 = np.asarray(F_soil_CO2)
    F_plant_CO2 = np.asarray(F_plant_CO2)

    return (SOC_F, SOC_S, DOC, MIC_aero, MIC_ana, Enz, CH4, CO2), (F_soil_CH4, F_plant_CH4, F_soil_CO2, F_plant_CO2) 




        
