"""Base classes for data."""

from __future__ import print_function
import operator
import uuid
from datetime import datetime
import copy
import sys
import six

from pyfusion.conf.utils import import_from_str, get_config_as_dict
from pyfusion.data.filters import filter_reg
from pyfusion.data.plots import plot_reg
from pyfusion.data.utils import unique_id
import pyfusion
import pyfusion.utils
from pyfusion.debug_ import debug_
from pyfusion.orm.utils import orm_register


try:
    from sqlalchemy import Table, Column, String, Integer, Float, ForeignKey, \
         DateTime, PickleType
    from sqlalchemy.orm import reconstructor, mapper, relation, dynamic_loader
    from sqlalchemy.orm.collections import column_mapped_collection
except ImportError:
    # TODO: use logger!
    def reconstructor(*args):
        print('need sqlalchemy for some operations - if no messages, you are OK')

    print("could not import sqlalchemy")

    
def get_history_args_string(*args, **kwargs):
    args_string = ', '.join(map(str,args))
    if args_string is not '':
        args_string += ', '
    kwargs_string = ', '.join("%s='%s'" %(str(i[0]), str(i[1]))
                              for i in kwargs.items())
    return [args_string, kwargs_string]

def history_reg_method(method):
    """Wrapper for filter and plot methods which updates the data history."""
    def updated_method(input_data, *args, **kwargs):
        do_copy = kwargs.pop('copy', True)
        if do_copy:
            original_hist = input_data.history
            # bdb testing copy - was - reverted back takes 9 sec longer (32 cf 41)
            input_data = copy.copy(input_data)
            #pyfusion.utils.warn('using deepcopy - bdb')
            #input_data = copy.deepcopy(input_data)
            copy_history_string = " > (copy)" # %(datetime.now()) no need for time on copy
            input_data.history = original_hist + copy_history_string
        
        [args_string, kwargs_string] = get_history_args_string(*args, **kwargs)

        history_string = "\n%s > %s(%s%s)" %(datetime.now(), method.__name__,
                                               args_string, kwargs_string)
        input_data.history += history_string
        output = method(input_data, *args, **kwargs)

        # TODO output.meta.update() looks wrong - if a filter modifies a meta value, does this
        # overwrite the modified version with the original?

        if output is not None:
            output.meta.update(input_data.meta)

        return output
    return updated_method

class MetaMethods(type):  # type here is used in the sense type(name, bases, dict)
    """Metaclass which provides filter and plot methods for data classes."""
    def __new__(cls, name, bases, attrs):
        attrs_in = attrs.copy()
        # if len(filter_reg) > 1: print('filter reg', filter_reg)
        for reg in [filter_reg, plot_reg]:  # this is one or the other
            reg_methods = reg.get(name, [])
            # if reg_methods is not []: print('name', reg.get(name, []))
            attrs.update((rm.__name__, history_reg_method(rm))
                         for rm in reg_methods)
# the remainer of this function was not indented enough before, so filters were ignored
    #        return super(MetaMethods, cls).__new__(cls, name, bases, attrs)
            newc = super(MetaMethods, cls).__new__(cls, name, bases, attrs)
            # Note: only executed at startup, so import pyfusion, then set DEBUG, then run
            # pyfusion.DEBUG='register' stops on all registers and metaMethods 
            if pyfusion.VERBOSE > 1: print(name)
            debug_(pyfusion.DEBUG, 3, key=['register', 'MetaMethods'])
            # a simple minded test - see proper implementation a little further on
            if pyfusion.VERBOSE > 1 and name == 'TimeseriesData' and hasattr(newc, 'plot_signals'):
                print('=============>>>>>>> TimeseriesData.plot_signals doc'),
                # __doc__ is missing at this point
                if newc.plot_signals.__doc__ is None:
                    print('is lost')
                else:
                    print(' is ' + newc.plot_signals.__doc[0:50])

            # restore the __doc__ to updated functions - adapted from: q
            # http://stackoverflow.com/questions/8100166/inheriting-methods-docstrings-in-python
            import types
            for func in reg_methods:
                if isinstance(func, types.FunctionType) and func.__doc__ is not None:
                    if pyfusion.VERBOSE>0: print(func, 'needs doc...', end='')
                    for newf in (newc.__dict__):
                        # py3 hide the func_name from version 3 for now
                        if isinstance(newc.__dict__[newf], types.FunctionType) and ((sys.version > '3.0.0') or (newf == func.func_name)):
                            newc.__dict__[newf].__doc__ = func.__doc__
                            if pyfusion.VERBOSE>0: print(' setting..')
        return(newc)

class Coords(object):
    """Stores coordinates with an interface for coordinate transforms."""
    def __init__(self, default_coords_name, default_coords_tuple,  **kwargs):
        self.default_name = default_coords_name
        self.default_value_1 = default_coords_tuple[0]
        self.default_value_2 = default_coords_tuple[1]
        self.default_value_3 = default_coords_tuple[2]
        kwargs.update(((default_coords_name, default_coords_tuple),))
        self.__dict__.update(kwargs)

    def add_coords(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def load_from_config(self, **kwargs):
        for kw in kwargs.iteritems():
            if kw[0] == 'coord_transform':
                transform_list = pyfusion.config.pf_options('CoordTransform', kw[1])
                for transform_name in transform_list:
                    transform_class_str = pyfusion.config.pf_get('CoordTransform', kw[1], transform_name)
                    transform_class = import_from_str(transform_class_str)
                    self.load_transform(transform_class)
            elif kw[0].startswith('coords_'):
                coord_values = tuple(map(float,kw[1].split(',')))
                self.add_coords(**{kw[0][7:]: coord_values})
    
    def load_transform(self, transform_class):
        def _new_transform_method(**kwargs):
            return transform_class().transform(self.__dict__.get(transform_class.input_coords),**kwargs)
        self.__dict__.update({transform_class.output_coords:_new_transform_method})

    def save(self):
        if pyfusion.orm_manager.IS_ACTIVE:
            # this may be inefficient: get it working, then get it fast
            session = pyfusion.orm_manager.Session()
            session.add(self)
            session.commit()
            session.close()
        

@orm_register()
def setup_coords(man):
    man.coords_table = Table('coords', man.metadata,
                            Column('id', Integer, primary_key=True),
                            Column('default_name', String(30), nullable=False),
                            Column('default_value_1', Float),
                            Column('default_value_2', Float),
                            Column('default_value_3', Float))
    man.metadata.create_all()
    mapper(Coords, man.coords_table)



def get_coords_for_channel(channel_name=None, **kwargs):
    config_dict = kwargs.copy()
    if channel_name:
        config_dict.update(get_config_as_dict('Diagnostic', channel_name))
    coord_name = 'dummy'
    coord_values = (0.0,0.0,0.0)
    transforms = []
    for k in config_dict.keys():
        if k.startswith('coords_'):
            coord_name = k[7:]
            coord_values = tuple(map(float,config_dict[k].split(',')))
    coords_instance = Coords(coord_name, coord_values)
    if 'coord_transform' in config_dict:
        transform_list = pyfusion.config.pf_options('CoordTransform', config_dict['coord_transform'])
        for transform_name in transform_list:
            # this seems to return all the globals too
            transform_class_str = pyfusion.config.pf_get('CoordTransform', config_dict['coord_transform'], transform_name)
            # so tyr to exclude the globals
            if pyfusion.config.has_option('global',transform_name):
                continue
            transform_class = import_from_str(transform_class_str)
            #if not hasattr(transform_class, 'output_coord'):
            #    raise Exception('??')
            coords_instance.load_transform(transform_class)
    debug_(pyfusion.DEBUG,1, key=['coord', 'device_name'])        
    return coords_instance

class Channel(object):
    # bdb starts to put the new names in, gently
    def __init__(self, name, coords, source="", parent_device="not specified"):
        self.name = name
        self.coords = coords
        self.parent_device = parent_device  # refers to LHD or H1 etc.
        self.source = source
        debug_(pyfusion.DEBUG,5, key=['class Channel','device_name'])        

    def save(self):
        """ applicable only to ORM db"""
        if pyfusion.orm_manager.IS_ACTIVE:
            # this may be inefficient: get it working, then get it fast
            self.coords.save()
            session = pyfusion.orm_manager.Session()
            session.add(self)
            session.commit()
            session.close()

@orm_register()
def orm_load_channel(man):
    man.channel_table = Table('channel', man.metadata,
                            Column('id', Integer, primary_key=True),
                            Column('name', String(200), nullable=False),
                            Column('coords_id', Integer, ForeignKey('coords.id'), nullable=False))
    #man.metadata.create_all()
    mapper(Channel, man.channel_table, properties={'coords': relation(Coords)})


    
@orm_register()
def orm_load_channel_map(man):
    man.channel_association_table = Table('channel_association', man.metadata,
                                      Column('channellist_id', Integer, ForeignKey('channellist.id'), primary_key=True),
                                      Column('channel_id', Integer, ForeignKey('channel.id'), primary_key=True),
                                      )

class ChannelList(list):
    def __init__(self, *args):
        self.extend(args)

    def save(self):
        if pyfusion.orm_manager.IS_ACTIVE:
            self._channels.extend(self)
            session = pyfusion.orm_manager.Session()
            session.add(self)
            session.commit()
            session.close()

            # to what extent does this duplicate time range?

    def get_channel_index(self, channel_name, bounds = None):
        for i,j in enumerate(self):
            if j.name == channel_name:
                return i
        else:
            return None

            
    @reconstructor
    def repopulate(self):
        for i in self._channels:
            if not i in self: self.append(i)
    
@orm_register()
def orm_load_channellist(man):
    man.channellist_table = Table('channellist', man.metadata,
                                  Column('id', Integer, primary_key=True))
                              
    #man.metadata.create_all()
    mapper(ChannelList, man.channellist_table,
           properties={'_channels': relation(Channel, secondary=man.channel_association_table)})
    


class PfMetaData(dict):
    pass


"""###  metaclass issues with python2/3
# we use @six.add_metaclass(MetaMethods)
# the alternative is (python 3)
class BaseData(metaclass=MetaMethods):
# or for python 2 
class BaseData(object):
    __metaclass__ = MetaMethods
"""
 
@six.add_metaclass(MetaMethods)
class BaseData(object):

    """Base class for handling processed data.

    In general, specialised subclasses of BaseData will be used
    to handle processed data rather than BaseData itself.

    Usage: ..........
    """
    __metaclass__ = MetaMethods

    def __init__(self, *args, **kwargs):
        self.meta = PfMetaData()
        self.history = "%s > New %s, %s" %(datetime.now(), self.__class__.__name__,pyfusion.VERSION)
        if not hasattr(self, 'channels'):
            self.channels = ChannelList()
        
    def save(self):
        if pyfusion.orm_manager.IS_ACTIVE:
            # this may be inefficient: get it working, then get it fast
            self.channels.save()
            session = pyfusion.orm_manager.Session()
            session.add(self)
            session.commit()
            session.close()

@orm_register()
def orm_load_basedata(man):
    man.basedata_table = Table('basedata', man.metadata,
                               Column('basedata_id', Integer, primary_key=True),
                               Column('type', String(30), nullable=False),
                               Column('meta', PickleType(comparator=operator.eq)),
                               Column('history', String(4096))
                               )
    #man.metadata.create_all()
    mapper(BaseData, man.basedata_table, polymorphic_on=man.basedata_table.c.type, polymorphic_identity='basedata')


@six.add_metaclass(MetaMethods)
class BaseDataSet(object):
    __metaclass__ = MetaMethods

    def __init__(self, label=''):
        self.meta = PfMetaData()
        self.created = datetime.now()
        self.history = "%s > New %s" %(self.created, self.__class__.__name__)
        if label == '':
            label = unique_id()
        self.label = label
        if not pyfusion.orm_manager.IS_ACTIVE:
            self.data = list()  # python3 - was set(), but can't be hashed (see also .add()
        
    def save(self):
        ## TODO: if orm_manager IS_ACTIVE=False, send message to logger...
        if pyfusion.orm_manager.IS_ACTIVE:
            session = pyfusion.orm_manager.Session()
            session.add(self)
            session.commit()
            session.close()

    def remove(self, item):
        self.data.remove(item)

    def update(self, item):
        self.data.update(item)
    
    def copy(self):
        return self.data.copy()
        
    def add(self, item):
        self.data.append(item)   # python3 was set, and so add used to be add()

    def __iter__(self):
        return self.data.__iter__()

    def __len__(self):
        return self.data.__len__()

    def pop(self):
        return self.data.pop()

@orm_register()
def orm_load_basedataset(man):
    man.basedataset_table = Table('basedataset', man.metadata,
                                  Column('id', Integer, primary_key=True),
                                  Column('created', DateTime),
                                  Column('label', String(100), nullable=False, unique=True),
                                  Column('type', String(30), nullable=False),
                                  Column('meta', PickleType(comparator=operator.eq)))

    # many to many mapping of data to datasets
    man.data_basedataset_table = Table('data_basedataset', man.metadata,
                                       Column('basedataset_id', Integer, ForeignKey('basedataset.id')),
                                       Column('data_id', Integer, ForeignKey('basedata.basedata_id'))
                                       )
    #man.metadata.create_all()

    mapper(BaseDataSet, man.basedataset_table,
           polymorphic_on=man.basedataset_table.c.type, polymorphic_identity='base_dataset')



class DynamicDataSet(BaseDataSet):
    pass

@orm_register()
def orm_load_dynamic_dataset(man):
    man.dynamicdataset_table = Table('dynamic_dataset', man.metadata,
                                     Column('basedataset_id', Integer, ForeignKey('basedataset.id'), primary_key=True))
    #man.metadata.create_all()
    mapper(DynamicDataSet, man.dynamicdataset_table, inherits=BaseDataSet, polymorphic_identity='dynamic_dataset',
           properties={'data': dynamic_loader(BaseData, secondary=man.data_basedataset_table, backref='dynamicdatasets', cascade='all')})
    


class DataSet(BaseDataSet):
    pass
        
        
@orm_register()
def orm_load_dataset(man):
    man.dataset_table = Table('dataset', man.metadata,
                            Column('basedataset_id', Integer, ForeignKey('basedataset.id'), primary_key=True))
    #man.metadata.create_all()
    mapper(DataSet, man.dataset_table, inherits=BaseDataSet, polymorphic_identity='dataset',
           properties={'data': relation(BaseData, secondary=man.data_basedataset_table, backref='datasets', cascade='all', collection_class=set)})


class OrderedDataSetItem(object):
    def __init__(self, item, index):
        self.item = item
        self.index = index

@six.add_metaclass(MetaMethods)
class BaseOrderedDataSet(object):
    __metaclass__ = MetaMethods

    def __init__(self, label=''):
        self.created = datetime.now()
        self.label = label
        self.history = "%s > New %s" %(self.created, self.__class__.__name__)
        if label == '':
            label = unique_id()
        if not pyfusion.orm_manager.IS_ACTIVE:
            self.data_items = []
        
    def save(self):
        if pyfusion.orm_manager.IS_ACTIVE:
            session = pyfusion.orm_manager.Session()
            session.add(self)
            session.commit()
            session.close()

    def append(self, item):
        if pyfusion.orm_manager.IS_ACTIVE:
            self.data_items[len(self)] = OrderedDataSetItem(item, len(self))
        else:
            self.data_items.append(item)
    def __len__(self):
        #if pyfusion.orm_manager.IS_ACTIVE:
        #    return self.data_items.count()
        #else:
        return self.data_items.__len__()

    def __getitem__(self, key):
        if pyfusion.orm_manager.IS_ACTIVE:
            try:
                return self.data_items[key].item
            except KeyError:
                # for loops require IndexError to detect
                # end of sequence.
                raise IndexError
        else:
            return self.data_items.__getitem__(key)


@orm_register()
def orm_load_baseordereddataset(man):
    man.baseordereddataset_table = Table('baseordereddataset', man.metadata,
                                         Column('id', Integer, primary_key=True),
                                         Column('created', DateTime),
                                         Column('label', String(50), nullable=False, unique=True),
                                         Column('type', String(30), nullable=False))

    man.ordereditems_table = Table('ordereddata_items', man.metadata,
                         Column('dataset_id', Integer, ForeignKey('baseordereddataset.id'),
                                primary_key=True),
                         Column('item_id', Integer, ForeignKey('basedata.basedata_id'),
                                primary_key=True),
                         Column('index', Integer, nullable=False)
                         )
    
    man.metadata.create_all()

    mapper(BaseOrderedDataSet, man.baseordereddataset_table,
           polymorphic_on=man.baseordereddataset_table.c.type, polymorphic_identity='base_ordered_dataset',
           properties={'data_items': relation(OrderedDataSetItem,
                                                  backref='ordered_datasets_items',
                                                  cascade='all, delete-orphan',
                                                  collection_class=column_mapped_collection(man.ordereditems_table.c.index))
                       }
           )
    mapper(OrderedDataSetItem, man.ordereditems_table, properties={
        'item': relation(BaseData, lazy='joined', backref='dataitem')
        })


class OrderedDataSet(BaseOrderedDataSet):
    pass
"""
if pyfusion.USE_ORM:
    ordered_dataset_table = Table('ordered_dataset', pyfusion.metadata,
                                  Column('base_ordered_dataset_id', Integer,
                                         ForeignKey('baseordereddataset.id'), primary_key=True))
                                  #Column('ordered_by', String(50)))

    pyfusion.metadata.create_all()
    mapper(OrderedDataSet, ordered_dataset_table, inherits=BaseOrderedDataSet, polymorphic_identity='ordered_datasets')
"""

class BaseCoordTransform(object):
    """Base class does nothing useful at the moment"""
    input_coords = 'base_input'
    output_coords = 'base_output'

    def transform(self, coords):
        return coords

class FloatDelta(BaseData):
#  Note: channel_1, 2 are channels, not channel names
    def __init__(self, channel_1, channel_2, delta, **kwargs):
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.delta = delta
        super(FloatDelta, self).__init__(**kwargs)

@orm_register()
def orm_load_floatdelta(man):
    man.floatdelta_table = Table('floatdelta', man.metadata,
                            Column('basedata_id', Integer, ForeignKey('basedata.basedata_id'), primary_key=True),
                            Column('channel_1_id', Integer, ForeignKey('channel.id')),
                            Column('channel_2_id', Integer, ForeignKey('channel.id')),
                            Column('delta', Float))    
    #man.metadata.create_all()
    mapper(FloatDelta, man.floatdelta_table, inherits=BaseData, polymorphic_identity='floatdelta',
           properties={'channel_1': relation(Channel, primaryjoin=man.floatdelta_table.c.channel_1_id==man.channel_table.c.id),
                       'channel_2': relation(Channel, primaryjoin=man.floatdelta_table.c.channel_2_id==man.channel_table.c.id)})


