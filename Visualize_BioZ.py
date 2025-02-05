import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
# import math
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import pickle
# from sklearn.model_selection import cross_val_predict
# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error
from os.path import exists, isdir, isfile
# from scipy import signal
# from scipy import stats
from Utilities.AdvancedLoggingParser import *
from Utilities.Dataset import parse_file_for_data_sets
# from Utilities.FileParser import parse_file
# from scipy.optimize import curve_fit
from scipy.io import savemat
plt.close('all')

def save_data(data):
    with open('Data\SkinElectrodeData\longitudinal_data\corrected_saved_data\s1009_move_gel.pkl', 'wb') as f:
        pickle.dump(data, f) 
    return

# 
GFCoefficientsTIA = np.load("Data/CalibrationData/B001/TIACoeff.npy")
FileLocation = []
# for viewing ankle device data
# FileLocation.append('Data/kinematicsData/exo_study_amro/biomech_s1_exo.txt')

# for viewing knee device data
data_loc = 'biomech_s3_base_arms.txt' # my data
# data_loc = 'biomech_s5_exo.txt'

# left is run 005, right is run003 for base
# data_loc = 's2_kn_base_right/run003'


save_mat_bool = False
# FileLocation.append('Data/testingData/run032')
# FileLocation.append('Data/US Bioz Data/' + data_loc + '.txt')

FileLocation.append('/Users/chjnichols/Dropbox (GaTech)/Bioimpedance_Software/Data/US Bioz Data/new_biomech_protocol_soleus_iso_vert2_mar13')
# FileLocation.append('Data/kinematicsData/exo_study_amro/'+ data_loc)
# FileLocation.append('Data/kinematicsData/weight vest study/'+ data_loc)

# FileLocation.append('Data/kinematicsData/weight vest study/run088_q7_vest_knee')

# if old device formatting
if isfile(FileLocation[0]):
    electrodeZ = []
    tissueR = []
    tissueX = []
    for jj in FileLocation: #FileLocation
        if exists(jj+'.pkl'):
            Datasets =  pickle.load(open(jj[:-3] +'.pkl', 'rb')) 
        else:
            Datasets = parse_file_for_data_sets(jj)
            #save_data(Datasets)
        # index = [2] 
        index = np.arange(len(Datasets))
        for DatasetIndex in index:
            fig1, ax1 = plt.subplots(3)
            fig2, ax2 = plt.subplots(3)
            fig3, ax3 = plt.subplots()
            NumFrequencies = int(Datasets[DatasetIndex].JsonDict["BioZ"]["NumFrequencies"])

            Frequencies = np.linspace(5000, 100000, NumFrequencies)
            Zprotection = (1/(1j*2*np.pi*Frequencies*0.01e-6))+(1/(1j*2*np.pi*Frequencies*0.47e-6)) + 2000

            # for 5k100k set to 2
            if NumFrequencies ==2:
                GFCoefficientsTIA = GFCoefficientsTIA[:,[0,-1]]

            Vv = np.zeros([Datasets[DatasetIndex].RawDataset.BioZ.shape[1],NumFrequencies], dtype=complex)
            Vi = np.zeros([Datasets[DatasetIndex].RawDataset.BioZ.shape[1],NumFrequencies],dtype=complex)
            for i in range(Datasets[DatasetIndex].RawDataset.BioZ.shape[1]):
                Vv[i] = Datasets[DatasetIndex].RawDataset.CurrentVoltage[:,i,2]# - VvBias
                Vi[i] = Datasets[DatasetIndex].RawDataset.CurrentVoltage[:,i,1] #- ViBias
            Vtinv = 1/Vi
            Vz = Vv * Vtinv

            AtiaGF = (np.zeros([Vz.shape[0], 2, NumFrequencies]))

            ZtotEstMag = np.zeros(Vz.shape)

            for i in range(NumFrequencies):
                AtiaGF[:,0,i] = np.abs(Vtinv[:,i])
                AtiaGF[:,1,i] = 1
                ZtotEstMag[:, i] = np.dot(AtiaGF[:, :, i], GFCoefficientsTIA[:, i]) - np.abs(Zprotection[i])


            electrodeZ.append(np.mean(ZtotEstMag.T))
            tissueR.append(np.mean(Datasets[DatasetIndex].BioZ.Data[:,:,1].T))
            
            fig1.suptitle('5 kHz', fontsize=12)
            ax1b = ax1[0].twinx()

            # ax1[0].plot(Datasets[DatasetIndex].BioZ.Data[0,:,0]/1000, ZtotEstMag[:,0]) # use 60000 for minutes, 1000 for seconds. 
            ax1[0].plot(Datasets[DatasetIndex].BioZ.Data[0,:,0]/1000,Datasets[DatasetIndex].BioZ.Data[0,:,1])

            ax1b.plot(Datasets[DatasetIndex].IMU.Data[:,0]/1000,Datasets[DatasetIndex].IMU.Data[:,10], color = 'orange')

            ax1[0].set_title('Electrode Z')
            ax1[1].plot(Datasets[DatasetIndex].BioZ.Data[0,:,0]/1000, Datasets[DatasetIndex].BioZ.Data[0,:,1]) #Datasets[0].BioZ.Data[0,:,0]/60000
            ax1[1].set_title('Tissue R')
            # ax1[0].set_ylim([0, 5000])
            ax1[2].plot(Datasets[DatasetIndex].BioZ.Data[0,:,0]/1000, Datasets[DatasetIndex].BioZ.Data[0,:,2])
            ax1[2].set_title('Tissue X')
            plt.tight_layout()

            fig2.suptitle('100 kHz', fontsize=12)
            ax2b = ax2[0].twinx()
            # ax2[0].plot(Datasets[DatasetIndex].BioZ.Data[-1,:,0]/60000,(ZtotEstMag[:,-1]))         #ax[0,0].plot(self.Data[:, 0] / 3600000, self.Data[:, 1])

            ax2[0].plot(Datasets[DatasetIndex].BioZ.Data[-1,:,0]/1000,Datasets[DatasetIndex].BioZ.Data[-1,:,1])
            ax2b.plot(Datasets[DatasetIndex].IMU.Data[:,0]/1000,Datasets[DatasetIndex].IMU.Data[:,10], color = 'orange')
            ax2[0].set_title('Electrode Z')
            ax2[1].plot(Datasets[DatasetIndex].BioZ.Data[-1,:,0]/1000,Datasets[DatasetIndex].BioZ.Data[-1,:,1])
            ax2[1].set_title('Tissue R')
            ax2[2].plot(Datasets[DatasetIndex].BioZ.Data[-1,:,0]/1000, Datasets[DatasetIndex].BioZ.Data[-1,:,2])
            ax2[2].set_title('Tissue X')
            
            ax3.plot(Datasets[DatasetIndex].BioZ.Data[-1,:,0]/1000, Datasets[DatasetIndex].BioZ.Data[0,:,1]/Datasets[DatasetIndex].BioZ.Data[-1,:,1])
            ax3b = ax3.twinx()
            ax3b.plot(Datasets[DatasetIndex].IMU.Data[:,0]/1000,Datasets[DatasetIndex].IMU.Data[:,10], color = 'orange')

            ax3.set_title('Ratio')
            plt.tight_layout()
            Datasets[-1].IMU.plot_gyro()

            plt.show()
            
            if save_mat_bool:
                savestr =  '/Users/chjnichols/Dropbox (GaTech)/US BioZ Data/BioZ Data/' + data_loc + '_' + str(DatasetIndex)
                # arrange data into dictonary for going to matlab
                mdic = {"BioZ": Datasets[DatasetIndex].BioZ.Data, "Frequencies": Datasets[DatasetIndex].BioZ.Frequencies,"IMU": Datasets[DatasetIndex].IMU.Data,
                        "ElectrodeZ":ZtotEstMag}

                savemat(savestr, mdic)

    print(f"electrode impedance: {np.mean(electrodeZ)} +/- {np.std(electrodeZ)}")
    print(f"tissue impedance: {np.mean(tissueR)} +/- {np.std(tissueR)}")
    # Datasets[-1].IMU.plot_accelerometer()
    Datasets[-1].IMU.plot_gyro()
    plt.show()
    5


# if new device formatting    
elif isdir(FileLocation[0]):
    # Subject = AdvancedLoggingDataset('Data/kinematicsData/exo_study_amro/run050')
    Subject = AdvancedLoggingDataset(FileLocation[0])
    # Subject.BioZ1.ElectrodeImpedance # vector length of bioz, 1d
    # Subject.BioZ1.Data # same format as before: freq, datapoint, tt r or x
    # Subject.IMU.Data

    # Subject.BioZ1.Data[0,:,0] /= 1000
    # samplerate = 1 / (Subject.BioZ1.Data[0,1,0] - Subject.BioZ1.Data[0,0,0])
    # # Frequency = 33.57377049180328 Hz
    # f=abs(np.fft.fft(Subject.BioZ1.Data[0,1400:,1]))
    
    # N = len(Subject.BioZ1.Data[0,1400:,1])
    
    # yf = rfft(Subject.BioZ1.Data[0,1400:,1])
    # xf = rfftfreq(N, samplerate)
    # xf2 = xf / (1/samplerate) * samplerate
    # fig, ax = plt.subplots()
    # ax.plot(xf, np.abs(yf))
    # plt.show()
    fig1, ax1 = plt.subplots(3, sharex=True)
    fig2, ax2 = plt.subplots(3, sharex=True)
    fig3, ax3 = plt.subplots()

    fig1.suptitle('5 kHz', fontsize=12)
    ax1b = ax1[0].twinx()

    # ax1[0].plot(Datasets[DatasetIndex].BioZ.Data[0,:,0]/1000, ZtotEstMag[:,0]) # use 60000 for minutes, 1000 for seconds. 
    ax1[0].plot(Subject.BioZ1.Data[0,:,0]/1000,Subject.BioZ1.Data[0,:,1])

    ax1b.plot(Subject.IMU.Data[:,0]/1000,Subject.IMU.Data[:,9], color = 'orange')

    ax1[0].set_title('Electrode Z')
    ax1[1].plot(Subject.BioZ1.Data[0,:,0]/1000, Subject.BioZ1.Data[0,:,1]) #Datasets[0].BioZ.Data[0,:,0]/60000
    ax1[1].set_title('Tissue R')
    # ax1[0].set_ylim([0, 5000])
    ax1[2].plot(Subject.BioZ1.Data[0,:,0]/1000, -Subject.BioZ1.Data[0,:,2])
    ax1[2].set_title('Tissue X')
    plt.tight_layout()

    fig2.suptitle('100 kHz', fontsize=12)
    ax2b = ax2[0].twinx()
    # ax2[0].plot(Datasets[DatasetIndex].BioZ.Data[-1,:,0]/60000,(ZtotEstMag[:,-1]))         #ax[0,0].plot(self.Data[:, 0] / 3600000, self.Data[:, 1])

    ax2[0].plot(Subject.BioZ1.Data[-1,:,0]/1000,Subject.BioZ1.Data[-1,:,1])
    ax2b.plot(Subject.IMU.Data[:,0]/1000,Subject.IMU.Data[:,9], color = 'orange')
    ax2[0].set_title('Electrode Z')
    ax2[1].plot(Subject.BioZ1.Data[-1,:,0]/1000,Subject.BioZ1.Data[-1,:,1])
    ax2[1].set_title('Tissue R')
    ax2[2].plot(Subject.BioZ1.Data[-1,:,0]/1000, -Subject.BioZ1.Data[-1,:,2])
    ax2[2].set_title('Tissue X')

    ax3.plot(Subject.BioZ1.Data[-1,:,0]/1000, Subject.BioZ1.Data[0,:,1]/Subject.BioZ1.Data[-1,:,1])
    ax3b = ax3.twinx()
    ax3b.plot(Subject.IMU.Data[:,0]/1000,Subject.IMU.Data[:,9], color = 'orange')

    ax3.set_title('Ratio')
    plt.tight_layout()
    plt.show()


    Subject.BioZ1.plot()
    # Subject.BioZ1.plot_ratio()
    Subject.IMU.plot_gyro()
    Subject.IMU.plot_accelerometer()

    5
    # BioZ1 = copy.deepcopy(SubjectL.BioZ1)
    # BioZ0 = copy.deepcopy(SubjectL.BioZ0)
    # IMU = copy.deepcopy(SubjectL.IMU)
    # BioZ1.plot()
    
# fig, ax = plt.subplots()
# ax.plot(Subject.BioZ1.Data[0,:,0],Subject.BioZ1.Data[0,:,1])
# ax.set_ylim([0, 50])
# ax2 = ax.twinx()
# ax2.plot(Subject.IMU.Data[:,0],Subject.IMU.Data[:,10], color = [1,0,0])
# plt.show()
    5
