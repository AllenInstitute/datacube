from types import *
import sqlalchemy
from sqlalchemy import create_engine
from datacube import Datacube

# Connects to the database to retieve metadata, etc...
class Database:
    def __init__(self, connection):
        assert type(connection) is StringType, "connection is not a string: %r" % connection
        self.engine = create_engine(connection, echo=True)


    def get_meta(self, datacube):
        assert(isinstance(datacube, Datacube))
        return 'ok'

