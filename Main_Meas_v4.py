import pandas as pd
import numpy as np
import time
from gurobipy import *
import matplotlib.pyplot as plt
from Forecast_Execute_v2 import *
from gurobipy import *
import gurobipy as gp
from gurobipy import GRB
from gurobipy import Model
from gurobipy import quicksum

"""
Directory: C:/Users/anazeri/Downloads/ML/V2G/ML_Opt_RTDS_v3
file name: ML_Opt_RTDS_v4
Note: 
A multi-step LSTM model is designed. A chunck of load data will be the 
output of LSTM per ONE prediction. Multi-input-multi-output Model. 

Update:
**Main_Meas_v3.py**
06/16/2022    Replaced "scaler.fit_transform" with my "Normalizer" function.
06/17/2022    df["Load"].min() been used instead of 0 as xmin in Normalizer.
06/19/2022    PEP 8 coding standard satisfied. 
              Extracted "GenNEXTtimestamps" and "XinputUpdatedRTDS" out of "forecast_load".
              gurobi OPT removed for now. 
**Main_Meas_v4.py**
06/20/2022    gurobi OPT is placed again.
07/05/2022    future_len = 12, OPT fails we must relax the constraints
07/06/2022    Bc 12 was small-> interpolation -> was a bad idea! didnt work. switched to future_len =60 for opt-alone part in the NAPS 2022 paper 
"""

##############################
#-- Ppv upload --
P_pv = pd.read_csv("C:/Users/anazeri/Downloads/ML/V2G/load estimation/Data preparation/cleaned_year_2020_solar.csv")
P_pv['Time Stamp'] = pd.to_datetime(P_pv['Time Stamp'], infer_datetime_format=True)
# StarT = time.ctime()
# print ( "Start : %s" % time.ctime())
New_timestamp, _ = GenNEXTtimestamps(df_ini, num_timestamps, First_attampt = True)
X_input = x_train0[-unroll_length:, :].reshape(1, unroll_length, num_features )

XinputUpdatedwithRTDS = []
P_pv_batches = []
timestamps_list = []
future_len_tilnow= 0
PB_act_list = []
soc_list = []
P_Gen_list = []
Ppv_list = []
execute = True
i = 0
i_opt = 0

while execute == True:

    ##############################
    ## -- Forecast P_L --
    PL_forecasted,  y_forecasted_concat_X = forecast_load(New_timestamp, X_input )

    ##############################
    ## Optimizer part:
    # Inputs: P_L & P_V
    # Outputs: P_G & P_B
    ##############################
    # -- Ppv data prepration for Optimizer --
    """
        I used .index.values[0] because .index will output a single-element LIST, so .values[0] is needed 
        to extract the Integer out of Int64Index
    """
    Start_time = New_timestamp.iloc[0]["Time Stamp Generated"]
    P_pv_timestamp_start  = P_pv[ P_pv['Time Stamp']  == Start_time  ]
    P_pv_batch = P_pv.iloc[P_pv_timestamp_start.index.values[0]+1 : P_pv_timestamp_start.index.values[0]+1+1*future_len][['Time Stamp', 'Power(MW)']] 
    # P_pv_batch = P_pv.iloc[P_pv_timestamp_start.index.values[0]+1 : -4500][['Time Stamp', 'Power(MW)']] 

    PL = PL_forecasted
    Ppv = np.asarray(P_pv_batch["Power(MW)"]).reshape(-1,1)

    # tt = np.linspace(0, future_len-1,future_len)
    # tt_intrp =  np.linspace(0, future_len-1,future_len*1)
    # PL_intrp = np.interp( tt_intrp ,tt ,PL_forecasted.reshape(future_len) )
    # Ppv_intrp = np.interp( tt_intrp ,tt ,Ppv.reshape(future_len) )
    # plt.plot(Ppv)

    if True:
        
        # Updated PL batch used for PB optimization
        # PL = PL_intrp.reshape(-1,1)
        # Ppv = Ppv_intrp.reshape(-1,1)



        m = gp.Model('grid')

        eta = 0.85
        # soc0 = 0.6
        Bb = 20
        delta_t = 5/60

        Nvar = future_len*1
        Time = range(0,Nvar)    # Index of time [1 - 839]
        TT = range(1,Nvar)

        PG = m.addVars(Time, lb=0.5, ub=1.2 , name="gen")       
        Pb = m.addVars(Time, lb=0, ub=0.6, name="battery")
        u = m.addVars(Time, lb=0, ub=1, vtype=GRB.BINARY, name="binary")                
        soc = m.addVars(Time, lb=0.1, ub=0.9, name="soc")

        m.setObjective(quicksum(Pb[t] for t in Time), GRB.MINIMIZE)

        m.addConstrs((PG[t] - PL[t] + Ppv[t] + Pb[t]*(2*u[t]-1) == 0) for t in Time)                                                                                            
        m.addConstrs((soc[t] == soc[t-1] + eta*Pb[t-1]*(1-u[t-1])*delta_t/Bb - Pb[t-1]*u[t-1]*delta_t/(Bb*eta)) for t in TT)
        # m.addConstrs((soc[t] <= 0.05 + soc[t-1] + eta*Pb[t-1]*(1-u[t-1])/Bb - Pb[t-1]*u[t-1]/(Bb*eta)) for t in TT)
        # m.addConstrs(( soc[t-1] + eta*Pb[t-1]*(1-u[t-1])/Bb - Pb[t-1]*u[t-1]/(Bb*eta) -0.05 <= soc[t]  ) for t in TT)

        if i_opt == 0:
            soc0 = 0.5
            i_opt = i_opt +1
        else:
            soc0 = soc_last       
        
        m.addConstr( soc[0] == soc0)      
        m.addConstr( 0.3<= soc[Nvar-1])
        m.addConstr( soc[Nvar-1] <= 0.7)
        m.addConstrs( Pb[t] <= 1.1*Pb[t-1]  for t in TT)
        m.addConstrs( 0.9*Pb[t-1] <= Pb[t]  for t in TT)
        m.addConstrs( PG[t] <= 1.1*PG[t-1]  for t in TT)
        m.addConstrs( 0.9*PG[t-1] <= PG[t]  for t in TT)

        m.Params.NonConvex = 2 # because it has quadratic constraints
        m.optimize()

        # for plot
        df = pd.DataFrame(m.getAttr(GRB.Attr.X, m.getVars()))

        df1 = df.loc[0:len(Time)-1]
        df2 = df.loc[len(Time):2*len(Time)-1]
        df3 = df.loc[2*len(Time):3*len(Time)-1]
        df4 = df.loc[3*len(Time):4*len(Time)-1]
        P_Gen = df1.to_numpy()
        PB_abs = df2.to_numpy()
        u_charg_disch = df3.to_numpy()
        soc = df4.to_numpy()
        PB_act = PB_abs*(2*u_charg_disch-1)

        soc_list.append(soc)
        PB_act_list.append(PB_act)
        P_Gen_list.append(P_Gen)
        Ppv_list.append(Ppv)

        np.savetxt("PG_ConSignal_timespan60_t"+str(i+1)+".csv", P_Gen)
        np.savetxt("P_B_t"+str(i+1)+".csv", PB_act)


        soc_last = soc[-1]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        plt.figure(figsize=(6, 4))

        ax1.plot(Time,P_Gen)
        ax2.plot(Time,PB_act)
        ax3.plot(Time,u_charg_disch)
        ax4.plot(Time,soc)
    

    ##############################
    ## -- Read P_L from RTDS/Simulink --
    RTDS_output = np.loadtxt("C:/Users/anazeri/Downloads/ML/V2G/ML_Opt_RTDS_v3/Measdata_5hrs_future_len60/RTDS_PL_unscaled"+str(i+1)+".csv")
    
    
    RTDS_output = RTDS_output.reshape(len(RTDS_output),1)

    func_plot(past_data= df_train_unscaled["Load"].values, future_len_sofar= future_len_tilnow, 
             predicted_future= PL_forecasted, true_future= RTDS_output)
    
    future_len_tilnow =  future_len_tilnow + future_len
    RTDS_output_scaled= Normalizer(RTDS_output, xmin = XMIN, xmax = XMAX, 
                                   min = 0, max = 1).reshape(1, len(RTDS_output))

    X_input = XinputUpdatedRTDS(num_timestamps, y_forecasted_concat_X, 
                                RTDS_output_scaled )

    Next_timestamp, _ = GenNEXTtimestamps(New_timestamp, num_timestamps, 
                                          First_attampt = False) 
        
    New_timestamp = Next_timestamp




    i = i +1
    if i == 2:

        PB_arr = np.array(PB_act_list).reshape(-1,1)
        plt.plot(PB_arr, label = "PB")
        plt.legend()
        plt.show()
        soc_arr = np.array(soc_list).reshape(-1,1)
        plt.plot(soc_arr, label = "soc")
        plt.legend()
        plt.show()

        P_Gen_list = np.array(P_Gen_list).reshape(-1,1)
        plt.plot(P_Gen_list, label = "P_Gen")
        plt.legend()
        plt.show()

        Ppv = np.array(Ppv_list).reshape(-1,1)
        plt.plot(Ppv, label = "Ppv")
        plt.legend()
        plt.show()


        execute = False



# PB_extend_list = []
# for i in range(len(PB_act)):
#     PB_extend = PB_act[i]*np.ones(10)
#     PB_extend_list = np.append(PB_extend_list,PB_extend)

# np.savetxt( "PB_batch1_scaledTimeSteps.csv" ,PB_extend_list)   
 

def P_d2c(P_out_disc, staircase_width):
    """
        Objective: Continueous-staircase signal generator from discrete signal 
        Parameters
        ----------
        P_out : 1D np.array, size = (future_len, 1)
                A discrete output power (like: PB, PG, PL, Ppv, ...) 
                to be convert into Continueous signal for Simulink
        stepsize: number of a single datapoint to be repeated. 

        Returns
        ----------
        Conti_signal : 1D np.array, size = ( future_len*staircase_width ,1) A continueous signal 
                       converted C-signal for Simulink.
    """
    Pout_conti_list = []
    for i in range(len(P_out_disc)):
        Pout_conti_i = P_out_disc[i]*np.ones(staircase_width)
        Pout_conti_list = np.append(Pout_conti_list, Pout_conti_i)
        Conti_signal = Pout_conti_list
    return Conti_signal

PL_signal = P_d2c(PL, staircase_width = 10)
plt.plot(PL_signal, label = "PL_signal")
plt.legend()
plt.show()

