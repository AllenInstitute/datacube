import nrrd
import numpy as np
import argparse
import os
from os import path
import json

class NrrdToMemMapped ():
    def __init__(self, nrrd_in_path, mapped_out_path):
        self.nrrd_path = nrrd_in_path
        self.out_path = path.join(mapped_out_path)
        self.file_name = path.basename(self.out_path).split('.')[0]
        self.temp_file = './temp.dat'


    def map (self):
        print("reading...")
        nrrd_file, options = nrrd.read(self.nrrd_path)

        self.arr_shape = nrrd_file.shape
        self.arr_shape = [int(i) for i in self.arr_shape]

        # Open the file, retrieve the array at [0], and save it to a new tempFile
        temp = open(self.temp_file, 'w')
        temp.write(nrrd_file)
        temp.close()

        print("writing...")
        mmap = np.memmap(self.temp_file, dtype = "uint32", shape = tuple(self.arr_shape), mode = 'c', order='f')
        
        np.save(self.file_name, mmap)
        
        jas = open (path.join(path.dirname(self.out_path), self.file_name + '_meta.json'), 'w')
        jas.write(json.dumps(options))
        jas.close()

    def verify(self):
        print "verifying write.."
        m = np.load(self.file_name + '.npy')
        print "File is valid: "  + str(m[131, 167, 259] == 1125)

    def cleanup(self):
        print "cleaning up..."
        os.remove(self.temp_file)

        print "done."


if __name__ == "__main__":
    # Get passed args
    parser = argparse.ArgumentParser(description = "Take in_file (nrrd) and creates a memmapped version at out_location. Usage: nrrd2mm input_file.nrrd output_file.dat")
    parser.add_argument("input", help = "nrrd file to convert" )
    parser.add_argument("output", help = "filepath for the memmapped output")

    args = parser.parse_args()
    
    print args

    nrrd_in_path = args.input
    mapped_out_path = args.output

    # Create mapper 
    mapper = NrrdToMemMapped(nrrd_in_path, mapped_out_path)
    mapper.map()
    mapper.verify()
    mapper.cleanup()