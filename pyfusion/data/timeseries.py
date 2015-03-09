"""Timeseries data classes."""

from datetime import datetime
import numpy as np

from pyfusion.data.base import BaseData, BaseOrderedDataSet, FloatDelta
from pyfusion.data.utils import cps, peak_freq, remap_periodic, list2bin, bin2list
from pyfusion.data.base import PfMetaData
import pyfusion
from pyfusion.orm.utils import orm_register
from pyfusion.debug_ import debug_

try:
    from sqlalchemy import Table, Column, Integer, ForeignKey, Float
    from sqlalchemy.orm import mapper, relation
except ImportError:
    # TODO: Use logger!
    print "cannot load sqlalchemy"
        
class Timebase(np.ndarray):
    """Timebase vector with parameterised internal representation.

    see doc/subclassing.py in numpy code for details on subclassing ndarray
    """
    def __new__(cls, input_array):
        # should this follow the example in doc/subclassing.py?... (it doesn't)
        obj = np.asarray(input_array).view(cls).copy()
        if obj[1]==obj[0]: 
            print('timeseries error elts 0 and 1 are both {o}'
                  .format(o = 1.0* obj[0]))  # format bug
        obj.sample_freq = 1.0/(obj[1]-obj[0])
        obj.meta = PfMetaData()
        return obj

    def is_contiguous(self):
        return max(((self[1:]-self[:-1])-1.0/self.sample_freq)**2) < (0.1/self.sample_freq)**2

    def normalise_freq(self, input_freq):
        """Normalise input frequencies to [0,1] where 1 is pi*sample_freq"""
        try:
            return input_freq/(0.5*self.sample_freq)
        except:
            return [i/(0.5*self.sample_freq) for i in input_freq] # bdb

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.sample_freq = getattr(obj, 'sample_freq', None)
        self.meta = getattr(obj, 'meta', None)
        # We do not need to return anything
        


def generate_timebase(t0=0.0, n_samples=1.e4, sample_freq=1.e6):
    sample_time = 1./sample_freq
    return Timebase(np.arange(t0, t0+n_samples*sample_time, sample_time))
"""
class Timebase:
    def __init__(self, t0=None, n_samples=None, sample_freq=None):
        self.t0 = t0
        self.n_samples = n_samples
        self.sample_freq = sample_freq
        self.timebase = self.generate_timebase()
    def generate_timebase(self):
"""

class Signal(np.ndarray):
    """Timeseries signal class with (not-yet-implemented) configurable
    digitisation.

    see doc/subclassing.py in numpy code for details on subclassing ndarray
    """
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls).copy()
        obj.meta = PfMetaData()
        return obj

    def __array_finalize__(self,obj):
        self.meta = getattr(obj, 'meta', None)


    def n_channels(self):
        if self.ndim == 1:
            return 1
        else:
            return self.shape[0]

    def n_samples(self):
        if self.ndim == 1:
            return self.shape[0]
        else:
            return self.shape[1]

    def get_channel(self, channel_number, bounds=None):
        """allows us to use get_channel(0) no matter what ndim is"""
        if self.ndim == 1 and channel_number == 0:
            if bounds is None: return self
            else: return self[bounds[0]:bounds[1]]
        elif self.ndim > 1:
            if bounds is None:
                return self[channel_number,:]
            else:return self[channel_number,bounds[0]:bounds[1]]
        else:
            raise ValueError

        
    #def add_channel(self, additional_array):
    ##### -- problems with resizing subclass of ndarray, need to be careful 
    #    additional_signal = Signal(additional_array)
    #    new_n_channels = self.n_channels() + additional_signal.n_channels()
    #    self.resize((new_n_channels, self.n_samples()))
    #    self[-additional_signal.n_channels():,:] = additional_signal
    

class TimeseriesData(BaseData):
    def __init__(self, timebase = None, signal=None, channels=None, bypass_length_check=False, **kwargs):
        self.timebase = timebase
        self.channels = channels
        self.scales = np.array([[1]])  # retain amplitude scaling info for later
        if bypass_length_check == True:
            self.signal = signal
        else:
            if signal.n_samples() == len(timebase):
                self.signal = signal
            else:
                raise ValueError, "signal has different number of samples to timebase"
        super(TimeseriesData, self).__init__(**kwargs)

    ## Boyd tried this for fun - seems to work - should check the right way.
    # idea is to access a channel by name from a multiple diagnostic.
    def keys(self):
        if len(np.shape(self.channels))==0:
            return([self.channels.config_name])
        else:
            return([c.config_name for c in self.channels])

    def __getitem__(self, name):
        if name not in self.keys():
            raise LookupError('{k} not in list of channels {n}'
                              .format(k=name, n=self.keys()))
        else:
            if len(self.keys())==1:
                return(self.signal)
            else:
                # I bet there is a tricky python construct to do this!
                for i, nm in enumerate(self.keys()):
                    if nm==name:
                        return(self.signal[i])

    def __eq__(self, other):
        return all([np.array_equal(self.timebase, other.timebase),
                    np.array_equal(self.signal, other.signal)])

@orm_register()
def orm_load_timeseries_data(man):
    man.tsd_table = Table('timeseriesdata', man.metadata,
                          Column('basedata_id', Integer, ForeignKey('basedata.basedata_id'), primary_key=True))
    #man.metadata.create_all()
    mapper(TimeseriesData, man.tsd_table, inherits=BaseData, polymorphic_identity='tsd')




class SVDData(BaseData):
    def __init__(self, chrono_labels, topo_channels, svd_input):
        """
        svd_input is a tuple as outputted by numpy.linalg.svd(data, 0)
        """
        self.channels = topo_channels
        self.chrono_labels = chrono_labels
        self.topos = np.transpose(svd_input[0])
        self.svs = svd_input[1]
        self.chronos = svd_input[2]
        self.E = sum(self.svs*self.svs)
        self.p = self.svs**2/self.E
        self.H = float((-1./np.log(len(self.svs)))*sum(self.p*np.log(self.p)))
        super(SVDData, self).__init__()

        
    def self_cps(self):
        try:
            return self._self_cps
        except AttributeError:
            self._self_cps = np.array([np.mean(cps(i,i)) for i in self.chronos])
            return self._self_cps


@orm_register()
def orm_load_svd_data(man):
    man.svd_table = Table('svddata', man.metadata,
                      Column('basedata_id', Integer, ForeignKey('basedata.basedata_id'), primary_key=True))
    #man.metadata.create_all()
    mapper(SVDData, man.svd_table, inherits=BaseData, polymorphic_identity='svddata')


class FlucStruc(BaseData):
    def __init__(self, svd_data, sv_list, timebase, min_dphase = -np.pi, phase_pairs = None):
        # NOTE I'd prefer not to duplicate info here which is in svd_data - should be able to refer to that, once sqlalchemy is hooked in
        #self.topo_channels = svd_data.topo_channels
        self.channels = svd_data.channels
        #self.svs = sv_list
        if len(sv_list) == 1:
            self.a12 = 0
        else:
            # this ratio was inverted accidentally at first
            self.a12 = svd_data.svs[sv_list[1]]/svd_data.svs[sv_list[0]]
        self._binary_svs = list2bin(sv_list)
        self.timebase = timebase
        # peak frequency for fluctuation structure, fpkf is freq peak factor
        self.freq, self.freq_elmt, self.fpkf = \
            peak_freq(svd_data.chronos[sv_list[0]], self.timebase)
        self.phase_pairs = phase_pairs
        self.t0 = timebase[0]
        # channel peaking factor - squaring increases contrast 
        top = np.abs(svd_data.topos[sv_list[0]])
        self.cpkf = np.max(top)/np.sqrt(np.average(top**2))
        # singular value filtered signals
        self.signal = np.dot(np.transpose(svd_data.topos[sv_list,:]),
                           np.dot(np.diag(svd_data.svs.take(sv_list)), svd_data.chronos[sv_list,:]))
        # phase differences between nearest neighbour channels
        self.dphase = self._get_dphase(min_dphase=min_dphase)
        self.p = np.sum(svd_data.svs.take(sv_list)**2)/svd_data.E
        self.H = svd_data.H
        self.E = svd_data.E
        debug_(pyfusion.DEBUG, 4, key='FlucStruc')
        super(FlucStruc, self).__init__()

    def save(self):
        self.dphase.save()
        super(FlucStruc, self).save()

    def svs(self):
        return bin2list(self._binary_svs)

    def _get_dphase(self, min_dphase = -np.pi):
        """
        remap to [min_dphase, min_dphase+2pi]
        """
        phases = np.array([self._get_single_channel_phase(i) for i in range(self.signal.shape[0])])
        debug_(pyfusion.DEBUG, 2, key='_get_dphase')
        if self.phase_pairs != None:
            tmp = []
            for a,b in self.phase_pairs:
                ind_a = self.channels.get_channel_index(a)
                ind_b = self.channels.get_channel_index(b)
                tmp.append(phases[ind_b]-phases[ind_a])
            d_phases=remap_periodic(np.array(tmp), min_val=min_dphase)
                
        else:
            d_phases = remap_periodic(phases[1:]-phases[:-1], min_val = min_dphase)
        #d_phase_dataset = OrderedDataSet(ordered_by="channel_1.name")
        d_phase_dataset = BaseOrderedDataSet('d_phase_%s' %datetime.now())
        ## append then sort should be faster than ordereddataset.add() [ fewer sorts()]
        if self.phase_pairs != None:
            for i,phase_pair in enumerate(self.phase_pairs):
                ind_a = self.channels.get_channel_index(phase_pair[0])
                ind_b = self.channels.get_channel_index(phase_pair[1])
                d_phase_dataset.append(FloatDelta(self.channels[ind_a],self.channels[ind_b],d_phases[i]))
        else:
            for i, d_ph in enumerate(d_phases):
                d_phase_dataset.append(FloatDelta(self.channels[i], self.channels[i+1], d_ph))

        #d_phase_dataset.sort()
        return d_phase_dataset


    def _get_single_channel_phase(self, ch_id):
        data_fft = np.fft.fft(self.signal[ch_id])  # bdb 30% ffts
        d_val = data_fft[self.freq_elmt]
        # big change, flipped phase sign 
        # was real, imag up until Feb 24 2013 - bdb
        phase_val = np.arctan2(d_val.imag,d_val.real)
        return phase_val

@orm_register()
def orm_load_flucstrucs(man):
    man.flucstruc_table = Table('flucstrucs', man.metadata,
                            Column('basedata_id', Integer, ForeignKey('basedata.basedata_id'), primary_key=True),
                            Column('_binary_svs', Integer),
                            Column('freq', Float),
                            Column('a12', Float),
                            Column('t0', Float),    
                            Column('p', Float),    
                            Column('H', Float),    
                            Column('dphase_id', Integer, ForeignKey('baseordereddataset.id'), nullable=False))    
    man.metadata.create_all()
    mapper(FlucStruc, man.flucstruc_table, inherits=BaseData,
           polymorphic_identity='flucstruc', properties={'dphase': relation(BaseOrderedDataSet)})
