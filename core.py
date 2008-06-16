"""
The core components of PyFusion. 
The classes and functions here are imported by pyfusion/__init__.py

requirements:
numpy
sqlalchemy version >= 0.4.4 (version 0.4.4 is required for the declarative extension)
"""

from numpy import array,mean,ravel,transpose,arange,var,log, take, shape, ones, searchsorted
from numpy.dual import svd
from utils import local_import, get_conditional_select, check_same_timebase
import settings 

from sqlalchemy import Column, Integer, ForeignKey, exceptions, PickleType, Float, Boolean
from sqlalchemy.orm import relation, synonym
#from sqlalchemy.ext.declarative import declared_synonym
import pyfusion


def get_shot(shot_number):    
    try:
        existing_shot = pyfusion.session.query(Shot).filter(Shot.device_id == pyfusion._device.id).filter(Shot.shot == shot_number).one()
        return existing_shot
    except:
        print "Creating shot %s:%d" %(pyfusion._device.name, shot_number)    
        s = Shot(shot_number)
        return s
    

class Shot(pyfusion.Base):
    """
    The class to represent any shot-specific data.    
    """
    __tablename__ = "shots"
    id = Column('id', Integer, primary_key=True)
    shot = Column('shot', Integer)
    device_id = Column('device_id', Integer, ForeignKey('devices.id'))
    device = relation(pyfusion.Device, primaryjoin=device_id==pyfusion.Device.id, backref="shots")    
    data = {}
    def __init__(self, sn):
        """
        sn: shot number (integer)
        """
        self.shot = sn
        self.device_id = pyfusion._device.id
        self.metadata.create_all()
        pyfusion.session.save_or_update(self)
        pyfusion.session.commit()


    def load_diag(self, diagnostic, ignore_channels=[]):
        print "Only MultiChannel Timeseries data works for now"
        diag = pyfusion.session.query(pyfusion.Diagnostic).filter(pyfusion.Diagnostic.name==diagnostic)[0]
        channel_list = []
        print diag
        print diag.ordered_channel_list
        for ch in diag.ordered_channel_list:
            if ch not in ignore_channels:
                channel_list.append(ch)
        for chi, chn in enumerate(channel_list):
            ch = pyfusion.session.query(pyfusion.Channel).filter(pyfusion.Channel.name==chn)[0]
            if ch.processdata_override:
                _ProcessData = pyfusion._device_module.ProcessData(data_acq_type = ch.data_acq_type, processdata_override = ch.processdata_override)
            else:
                _ProcessData = __import__('pyfusion.data_acq.%s.%s' %(ch.data_acq_type,ch.data_acq_type), globals(), locals(), ['ProcessData']).ProcessData()
            if chi==0:
                channel_MCT = _ProcessData.load_channel(ch, self.shot)
            else:
                _tmp = _ProcessData.load_channel(ch, self.shot)
                channel_MCT.add_multichannel(_tmp)
        self.data[diagnostic] = channel_MCT
        

    def define_time_segments(self, diag, n_samples = settings.N_SAMPLES_TIME_SEGMENT):
        """
        Create a list of time segments defined with width n_samples for primary_diag. 
        gives self.time_segments = [elements, times]. elements[:-1] and elements[1:] respectively provide lists for t0 and t1 elements for segements. Likewise, times[:-1] and times[1:] give t0 and t1 time lists
        @param n_samples: width of time segment (using sample rate of primary_diag)
        """
        len_timebase = len(self.data[diag].timebase)
        element_list = range(0,len_timebase,n_samples)
        times_list = take(self.data[diag].timebase, element_list).tolist()
        #if (len_timebase%n_samples != 0) and zeropad:
        #    element_list.append(len_timebase)
        #    times_list.append(self.data[diag].timebase[-1])
	self.time_segments = [[element_list[i], times_list[i]] for i in range(len(element_list))]

    def time_segment(self, segment_number, diag = ''):
        """
        return time segment for diag. 
        @param segment_number: segment is defined by self.time_segments[0][segment_number] : self.time_segments[0][segment_number+1] (or time_segments[0] -> time_segments[1])
        """
        if diag == '':
            diag = self.primary_diag

        if diag == self.primary_diag:
            # skip timebase test
            [e0,e1] = [self.time_segments[0][segment_number], self.time_segments[0][segment_number+1]]
            return self.data[diag].timesegment(e0,e1-e0, use_samples=[True, True])
        elif check_same_timebase(self.data[diag],self.data[self.primary_diag]):
            [e0,e1] = [self.time_segments[0][segment_number], self.time_segments[0][segment_number+1]]
            return self.data[diag].timesegment(e0,e1-e0, use_samples=[True, True])
        else:
            [t0,t1] = [self.time_segments[1][segment_number], self.time_segments[1][segment_number+1]]
            return self.data[diag].timesegment(t0,t1-t0, use_samples=[False, False])


class MultiChannelTimeseries(object):
    """
    A class to hold multichannel data. 
    """
    def __init__(self,timebase, parent_element = 0):
        """
        initiate with the timebase
        parent_element is to help keep track of where splitting occurs...
        """
        timebase = array(timebase)
        # check timebase is monotonically increasing
        if min(timebase[1:]-timebase[:-1]) <= 0:
            raise ValueError, "timebase is not monotonically increasing"
        self.timebase = timebase
        self.len_timebase = len(timebase)
        self.nyquist = 0.5/mean(self.timebase[1:]-self.timebase[:-1])
        self.signals = {}
        self.parent_element = parent_element
        self.t0 = min(timebase)
        self.norm_info = {} # raw (not normalised) data has empty dict.
        self.ordered_channel_list = [] # keep the order in which channels are added - use this ordering for SVD, etc

    def t_to_element(self, time_list):
        return searchsorted(self.timebase,time_list)

    def add_channel(self,signal,channel_name):
        signal = array(signal)
        if len(signal) == self.len_timebase:
            self.signals[str(channel_name)] = signal
            self.ordered_channel_list.append(str(channel_name))
        else:
            print "Signal '%s' not same length as timebase. Not adding to multichannel data" %channel_name

    def add_multichannel(self, multichanneldata):
        """
        join another MultiChannelTimeseries object to this one
        """
        if check_same_timebase(self, multichanneldata):
            for channel_name in multichanneldata.ordered_channel_list:
                self.add_channel(multichanneldata.signals[channel_name], channel_name)
        else:
            print "Timebase not the same. Not joining multichannel data"
    
    def export(self, filename, compression = 'bzip2', filetype = 'csv'):
        if compression != 'bzip2':
            raise NotImplementedError
        if filetype != 'csv':
            raise NotImplementedError
        import bz2
        if filename[-4:] != '.bz2':
            filename = filename + '.bz2'
        outfile = bz2.BZ2File(filename,'w')
        header_line = 't'
        for chname in self.ordered_channel_list:
            header_line = header_line + ', %s' %chname
        outfile.write(header_line+'\n')
        for i in range(len(self.timebase)):
            line = str(self.timebase[i])
            for chname in self.ordered_channel_list:
                line = line + ', '+str(self.signals[chname][i])
            outfile.write(line+'\n')
        outfile.close()

    def timesegment(self, t0, dt, use_samples=[False, False]):
        """
        return a reduced-time copy of the current MultiChannelTimeseries object
        @param t0: start time of segment (if use_samples[0] = True, then t0 is sample number rather than time)
        @param dt: width (t1-t0) of segment (if use_samples[1] = True, then dt is number of samples rather than length of time)
        @param use_samples: interpret t0, dt as samples instead of time.
        """
        # element for t0
        if use_samples[0]:
            e0 = t0
        else:
            e0 = self.timebase.searchsorted(t0)
        
        if use_samples[1]:
            e1 = t0+dt
        else:
            e1 = self.timebase.searchsorted(self.timebase[e0] + dt)

        new_mc_data = MultiChannelTimeseries(self.timebase[e0:e1], parent_element=e0)
        
        for ch in self.ordered_channel_list:
            new_mc_data.add_channel(self.signals[ch][e0:e1],ch)
        return new_mc_data

    def plot(self):
        import pylab as pl
        for ch_i, ch in enumerate(self.ordered_channel_list):
            pl.subplot(len(self.ordered_channel_list),1,ch_i+1)
            pl.plot(self.timebase,self.signals[ch])
            pl.ylabel(ch)
        pl.show()

    def spectrogram(self, max_freq = -1, noverlap=0, NFFT=1024):
        import pylab as pl
        for ch_i, ch in enumerate(self.ordered_channel_list):
            pl.subplot(len(self.ordered_channel_list),1,ch_i+1)
            Pxx, freqs, bins, im = pl.specgram(self.signals[ch], NFFT=NFFT, Fs=2.*self.nyquist,noverlap=noverlap)
            pl.ylabel(ch)
            if max_freq> 0:
                pl.ylim(0,max_freq)
        pl.show()
            
class TimeSegment(pyfusion.Base):    
    __tablename__ = 'timesegments'
    id = Column('id', Integer, primary_key=True)
    shot_id = Column('shot_id', Integer, ForeignKey('shots.id'))
    shot = relation(Shot, primaryjoin=shot_id==Shot.id)    
    primary_diagnostic_id = Column('primary_diagnostic_id', Integer, ForeignKey('diagnostics.id'))
    parent_min_sample = Column('parent_min_sample', Integer)
    n_samples = Column('n_samples', Integer)
    data = {}
    def _load_data(self, diag = None):
        # if there is no data in the shot (ie - reading from previous run) then try loading the primary diagnostic
        if len(self.shot.data.keys()) == 0:
            if diag:
                self.shot.load_diag(diag)
            else:
                pd = pyfusion.session.query(pyfusion.Diagnostic).filter_by(id = self.primary_diagnostic_id).one()
                self.shot.load_diag(pd.name)
        for diag_i in self.shot.data.keys():
            self.data[diag_i] = self.shot.data[diag_i].timesegment(self.parent_min_sample, self.n_samples, use_samples=[True, True])

class MultiChannelSVD(pyfusion.Base):
    __tablename__ = 'svds'
    id = Column('id', Integer, primary_key=True)    
    timesegment_id = Column('timesegment_id', Integer, ForeignKey('timesegments.id'))
    timesegment = relation(TimeSegment, primaryjoin=timesegment_id==TimeSegment.id, backref='svd')
    diagnostic_id = Column('diagnostic_id', Integer, ForeignKey('diagnostics.id'))
    diagnostic = relation(pyfusion.Diagnostic, primaryjoin=diagnostic_id==pyfusion.Diagnostic.id)
    svs = relation("SingularValue", backref='svd')
    entropy = Column('entropy', Float)
    energy = Column('energy', Float)
    timebase = Column('timebase', PickleType)
    channel_norms = Column('channel_norms', PickleType)
    used_channels = Column('used_channels', PickleType)
    normalised = Column('normalised', Boolean)
    def _get_chrono(self, chrono_number):
        #print '---'
        #print self.timesegment.data.keys()
        #data = array([self.timesegment.data[self.diagnostic.name].signals[c] for c in self.timesegment.data[self.diagnostic.name].ordered_channel_list])
        #data = []
        if not self.diagnostic.name in self.timesegment.data.keys():
            self.timesegment._load_data(diag=self.diagnostic.name)
        data = array([self.timesegment.data[self.diagnostic.name].signals[c] for c in self.timesegment.data[self.diagnostic.name].ordered_channel_list])
        #for c in self.timesegment.data[self.diagnostic.name].ordered_channel_list:
        #    if not c in self.timesegment.data.keys():
                
        #self.timebase = self.timesegment.data[self.diagnostic.name].timebase.tolist()
        #self.used_channels = self.timesegment.data[self.diagnostic.name].ordered_channel_list
        if self.normalised == True:
            #norm_list = []
            for ci,c in enumerate(data):
                #normval = c.var()
                #norm_list.append(normval)
                data[ci] /= self.channel_norms[ci]
            #self.channel_norms = norm_list
        #else:
        #    self.normalised = False
        #    self.channel_norms = []
        [tmp,svs,chronos] = svd(data,0)
        return chronos[chrono_number]

    def _do_svd(self, store_chronos=False, normalise = False):
        data = array([self.timesegment.data[self.diagnostic.name].signals[c] for c in self.timesegment.data[self.diagnostic.name].ordered_channel_list])
        self.timebase = self.timesegment.data[self.diagnostic.name].timebase.tolist()
        self.used_channels = self.timesegment.data[self.diagnostic.name].ordered_channel_list
        if normalise == True:
            self.normalised = True
            norm_list = []
            for ci,c in enumerate(data):
                normval = c.var()
                norm_list.append(normval)
                data[ci] /= normval
            self.channel_norms = norm_list
        else:
            self.normalised = False
            self.channel_norms = []
        [tmp,svs,chronos] = svd(data,0)
        topos = transpose(tmp)
        print 'done svd for %s' %(str(self.id))
        for svi,sv in enumerate(svs):
            if store_chronos:
                tmp1 = SingularValue(svd_id = self.id, number=svi, value=sv, chrono=chronos[svi].tolist(), topo=topos[svi].tolist())
            else:
                tmp1 = SingularValue(svd_id = self.id, number=svi, value=sv, chrono=None, topo=topos[svi].tolist())
            pyfusion.session.save(tmp1)
            self.svs.append(tmp1)
        ### (I read somewhere that x*x is faster than x**2)
        sv_sq = svs*svs
        
        ### total energy of singular values
        self.energy = sum(sv_sq)
        
        ### normalised energy of singular values
        p = sv_sq/self.energy

        ### entropy of singular values
        self.entropy = (-1./log(len(svs)))*sum(p*log(p))

    def plot(self):
        from pyfusion.visual import interactive_svd_plot
        interactive_svd_plot(self)

class SingularValue(pyfusion.Base):
    __tablename__ = 'svs'
    id = Column('id', Integer, primary_key=True)        
    svd_id = Column('svd_id', Integer, ForeignKey('svds.id'))
    number = Column('number', Integer)
    store_chrono = Column('store_chrono', Boolean)
    value = Column('value', Float)
    _chrono = Column('_chrono', PickleType)
    # if we don't store the chrono in sql, keep it here for as long as the object instance lasts..
    _tmp_chrono = []
    topo = Column('topo', PickleType)
    def _reload_chrono(self):
        parent_svd = pyfusion.session.query(MultiChannelSVD).filter_by(id=self.svd_id).one()
        self._tmp_chrono = parent_svd._get_chrono(self.number)
        return self._tmp_chrono
    
    def _get_chrono(self):
        if self.store_chrono:
            return self._chrono
        else:
            try:
                if len(self._tmp_chrono) > 0:
                    return self._tmp_chrono
                else:
                    return self._reload_chrono()
            except:
                return self._reload_chrono()

    def _set_chrono(self, chr):
        if self.store_chrono:
            self._chrono = chr
            self._tmp_chrono = []
        else:
            self._chrono = []
            self._tmp_chrono = chr
        
    chrono = synonym('_chrono', descriptor=property(_get_chrono, _set_chrono))


def get_time_segments(shot, primary_diag, n_samples = settings.N_SAMPLES_TIME_SEGMENT):
    shot.define_time_segments(primary_diag, n_samples = n_samples)
    output_list = []
    diag_inst = pyfusion.session.query(pyfusion.Diagnostic).filter_by(name = primary_diag).one()
    for seg_i, seg_min in enumerate(shot.time_segments):
        try:
            seg = pyfusion.session.query(TimeSegment).filter_by(shot = shot, primary_diagnostic_id=diag_inst.id, parent_min_sample=seg_min[0], n_samples=n_samples).one()
        except:# exceptions.InvalidRequestError:
            print "Creating segment %d" %seg_i
            seg  = TimeSegment(shot=shot, primary_diagnostic_id = diag_inst.id, n_samples = n_samples, parent_min_sample = seg_min[0])
        pyfusion.session.save_or_update(seg)
        output_list.append(seg)
    pyfusion.session.flush()
    return output_list


def new_timesegment(shot_instance, primary_diagnostic_name, t0, t1):
    diag_inst = pyfusion.session.query(pyfusion.Diagnostic).filter(pyfusion.Diagnostic.name == primary_diagnostic_name).one()
    t_els = shot_instance.data[primary_diagnostic_name].t_to_element([t0,t1])
    ts = pyfusion.TimeSegment(shot_id=shot_instance.id, primary_diagnostic_id=diag_inst.id, parent_min_sample = t_els[0],n_samples = t_els[1]-t_els[0])
    pyfusion.session.save(ts)
    pyfusion.session.flush()
    ts._load_data()
    return ts

def new_svd(timesegment_instance, diagnostic_id = -1):
    if diagnostic_id < 0:
        diagnostic_id = timesegment_instance.primary_diagnostic_id
    new_svd = pyfusion.MultiChannelSVD(timesegment_id=timesegment_instance.id, diagnostic_id = diagnostic_id)
    pyfusion.session.save(new_svd)
    pyfusion.session.flush()
    new_svd._do_svd()
    return new_svd
