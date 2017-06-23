from datacube import Datacube
from twisted.internet import reactor

datacube = Datacube(npy_file=npy_file)

def run(npy_file):
    reactor.run()
