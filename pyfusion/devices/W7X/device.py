from pyfusion.devices.base import Device
from pyfusion.orm.utils import orm_register

class W7X(Device):
    pass

@orm_register()
def orm_load_W7Xdevice(man):
    from sqlalchemy import Table, Column, Integer, ForeignKey
    from sqlalchemy.orm import mapper
    man.W7Xdevice_table = Table('W7Xdevice', man.metadata, 
                            Column('basedevice_id', Integer, ForeignKey('devices.id'), primary_key=True))
    mapper(W7X, man.W7Xdevice_table, inherits=Device, polymorphic_identity='W7X')
