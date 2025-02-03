import numpy as np

"""
Generates a file from patient data containing rheostat indices

data: NP array of patient data, Nx1
"""
def genSynthPatFiles(path_to_file, data):
    data.reshape(-1,1)
    r_dict = np.genfromtxt("Utilities/full_cal_resitanceDictionary.csv", delimiter=',')
    f = open(path_to_file, "w")
    dict_index = 0
    errs = np.zeros((data.shape[0],3))
    for i in range(data.shape[0]):
        dict_index = int(1 + np.around((data[i]-20)/0.05))
        errs[i,0] = data[i]
        errs[i,1] = r_dict[dict_index,0]
        errs[i,2] = errs[i,0]-errs[i,1]
        f.write("%04d,%04d,%04d,%04d\n" % tuple(r_dict[dict_index,1:5]))
    f.close()
    return errs

