import numpy as np
import torch
from sklearn.datasets import make_moons, make_swiss_roll
from sklearn.preprocessing import MinMaxScaler

def generate_swiss_roll_3d(num_samples):
    swiss_roll=make_swiss_roll(n_samples=num_samples, noise=0.0, random_state=None)
    swiss_roll=swiss_roll[0]
#     swiss_roll[:,0]=swiss_roll[:,0]/np.abs(swiss_roll[:,0]).max()
#     swiss_roll[:,1]=swiss_roll[:,1]/np.abs(swiss_roll[:,1]).max()
#     swiss_roll[:,2]=swiss_roll[:,2]/np.abs(swiss_roll[:,2]).max()
    scaler = MinMaxScaler()
    scaler.fit(swiss_roll)
    swiss_roll = scaler.transform(swiss_roll)
    samples=torch.from_numpy(swiss_roll).type(torch.FloatTensor)
    return samples


def generate_moons(num_samples):
    swiss_roll=make_moons(n_samples=num_samples, noise=0.0, random_state=None)[0]
    swiss_roll=swiss_roll/np.abs(swiss_roll).max()
    samples=torch.from_numpy(swiss_roll).type(torch.FloatTensor)
    return samples

