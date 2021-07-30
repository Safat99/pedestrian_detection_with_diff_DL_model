#this code is for to store numpy arrays to hard disk... so that we don't need to read the image again for the dataset

import numpy as np
import h5py

data = np.random.randint(100,size=(4,4))
#this will make a random np array which value will be int and 
# between 0 to 100

file = h5py.File('demohdf5_file.hdf5','w')

#for creating the dataset ... this will save as a dictionary
dset1 = file.create_dataset('dataset1',data=data)
#there could be more list I can store in the same file with different name/dictionary
file.close()

#hdf5 file read
readobj = h5py.File('demohdf5.hdf5','r')
#pri
print(list(readobj.keys()))
print()
dset1 = readobj['dataset1']
dset1 = dset1[:] # now it has become to the original state
