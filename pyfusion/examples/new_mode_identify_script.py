"""
Replaced by mode_identify_script.py

w=where(array(sd)<10)[0]
for ii in decimate(w,limit=2000): pl.plot(dd["phases"][ii],'k',linewidth=0.02)
mode.plot()

dd['N']=-1+0*dd["shot"].copy()

"""
import pylab as pl
import sys

from numpy import intersect1d, pi, shape, sqrt, average
from numpy import loadtxt
import pickle
import numpy as np

# This simple strategy works when the number is near zero +- 2Npi,
# which is true for calculating the deviation from the cluster centre.
# does not attempt to make jumps small (might be a good idea!)
def twopi(x, offset=pi):
    return ((offset+np.array(x)) % (2*pi) -offset)


shot_list = []

def askif(message, quiet=0):
    """ give messge, ask if continue (if quiet=0) or go on 
         anyway (quiet-1)
    """
    if quiet == 0: suffix =  " Continue?(y/N) "
    else: suffix = "continuing, set quiet=0 to stop..."
    if quiet==0: 
        ans = raw_input("Warning: {0}: {1}".format(message, suffix))
        if len(ans) ==0 or ans[0].lower() != 'y':
            raise LookupError(message)

class Mode():
    def __init__(self,name, N, NN, cc, csd, threshold=None, shot_list=[],MP2010_trick=False):
        self.name = name
        self.N = N
        self.NN = NN
        self.cc = np.array(cc)
        if MP2010_trick:  # don't use this - now separate mode lists!
            self.cc = -twopi(self.cc + np.pi, offset=4)  # this works for MP2010 if -B is the standard
        self.csd = np.array(csd)
        if threshold is None: threshold = 1
        self.threshold = threshold
        self.shot_list = shot_list
        leng = len(self.cc)
        if np.sum(self.cc)>0: Nest = np.sum(twopi(np.array(self.cc)-2)+2)
        else: Nest = np.sum(twopi(np.array(self.cc)+2)-2)

        Nest = float(leng+1)/leng/(2*np.pi) * Nest
        self.comment = self.name+",<N>~{0:.1f}".format(Nest)
        self.num_set = 0  # these are counters used when classifying
        self.num_reset = 0

    def store(self, dd, threshold=None,Nval=None,NNval=None,shot_list=None,quiet=0, mask=None):
        """ store coarse and fine mode (N, NN) numbers according to a threshold std and an optional shot_list.  If None the internal shot_list is used.
        which would have defaulted to [] at __init__
        """
        if shot_list is None: shot_list = self.shot_list
        if threshold is None: threshold=self.threshold
        else: self.threshold=threshold  # save the last manually set.

        if Nval is None: Nval = self.N
        if NNval is None: NNval = self.NN
        if NNval in np.unique(dd['NN']): 
            askif('NNval {0} already used'.format(NNval),quiet=quiet)

        if not(hasattr(dd['phases'],'std')):
            askif('convert phases to nd.array?',quiet=quiet)
            dd['phases'] = np.array(dd['phases'].tolist())

        if mask != None:
            sd = self.std_masked(phases,mask=mask)
        else:
            sd = self.std(phases)

        w = np.where(sd<threshold)[0]

        # normally apply to all shots, but can restrict to a particular
        # range of shots - e.g. dead MP1 for shot 54186 etc.
        if shot_list != []:  # used to be None  - be careful, there are
                             # two shot_list's one in mode, one input here
            where_in_shot = []
            for sht in shot_list:
                ws = where(dd['shot'][w] == sht)[0]
                where_in_shot.extend(w[ws])
            # this unique should not be required, but if the above logic
            # is changed, it might
            w = np.unique(where_in_shot)    

        if len(w) == 0: 
            print('threshold {th:.2g} is too low for phases: '
                  'minimum std for {m} is {sd:.1f}'
                  .format(th=threshold, m=self.name, sd=1*np.min(sd))) # format bug fix
            return()

        w_already = np.where(dd['NN'][w]>=0)[0]
        if len(w_already)>0:
            (cnts,bins) = np.histogram(dd['NN'][w], np.arange(-0.5,1.5+max(dd['NN'][w]),1))
            amx = np.argsort(cnts)
            print("NN already set in {0}/{1} locations {2:.1f}% of all data"
                  .format(len(w_already), len(w),
                          100*len(w_already)/float(len(dd['shot']))))
            print("NN={0} is most frequent ({1} inst.)"
                  .format(amx[-1],cnts[amx[-1]]))
            fract = len(w_already)/float(len(w))
            if fract>0.2: askif("{0:.1f}% of intended insts already set?".
                                format(fract*100),quiet=quiet)

        mode.num_set += len(w)
        mode.num_reset += len(w_already)

        dd['NN'][w]=NNval
        dd['N'][w]=Nval
        print("N={N}: set {s:.1f}%, total N set is now {t:.1f}%".
              format(s=100*float(len(w))/len(dd['shot']),N=Nval,
                     t=100*float(len(where(dd['N']>min(dd['N']))[0]))/len(dd['shot'])
                     ))
           
    def storeM(self, dd, threshold=None,Mval=None,MMval=None,shot_list=None,quiet=0):
        """ store coarse and fine mode (M, MM) numbers according to a threshold std and an optional shot_list.  If None the internal shot_list is used.
        which would have defaulted to [] at __init__
        """
        if shot_list is None: shot_list = self.shot_list
        if threshold is None: threshold=self.threshold
        else: self.threshold=threshold  # save the last manually set.

        if Mval is None: Mval = self.M
        if MMval is None: MMval = self.MM
        if MMval in np.unique(dd['MM']): 
            askif('MMval {0} already used'.format(MMval),quiet=quiet)

        if not(hasattr(dd['phases'],'std')):
            askif('convert phases to nd.array?',quiet=quiet)
            dd['phases'] = np.array(dd['phases'].tolist())

        sd = self.std(phases)
        w = np.where(sd<threshold)[0]

        # normally apply to all shots, but can restrict to a particular
        # range of shots - e.g. dead MP1 for shot 54186 etc.
        if shot_list != []:  # used to be None  - be careful, there are
                             # two shot_list's one in mode, one input here
            where_in_shot = []
            for sht in shot_list:
                ws = where(dd['shot'][w] == sht)[0]
                where_in_shot.extend(w[ws])
            # this unique should not be required, but if the above logic
            # is changed, it might
            w = np.unique(where_in_shot)    

        if len(w) == 0: 
            print('threshold {th:.2g} is too low for phases: '
                  'minimum std for {m} is {sd:.1f}'
                  .format(th=threshold, m=self.name, sd=np.min(sd)))
            return()

        w_already = np.where(dd['MM'][w]>=0)[0]
        if len(w_already)>0:
            (cnts,bins) = np.histogram(dd['MM'][w], np.arange(-0.5,1.5+max(dd['MM'][w]),1))
            amx = np.argsort(cnts)
            print("MM already set in {0}/{1} locations {2:.1f}% of all data"
                  .format(len(w_already), len(w),
                          100*len(w_already)/float(len(dd['shot']))))
            print("MM={0} is most frequent ({1} inst.)"
                  .format(amx[-1],cnts[amx[-1]]))
            fract = len(w_already)/float(len(w))
            if fract>0.2: askif("{0:.1f}% of intended insts already set?".
                                format(fract*100),quiet=quiet)

        mode.num_set += len(w)
        mode.num_reset += len(w_already)

        dd['MM'][w]=MMval
        dd['M'][w]=Mval
        print("set {s:.1f}%, total M set is now {t:.1f}%".
              format(s=100*float(len(w))/len(dd['shot']),
                     t=100*float(len(where(dd['M']>=0)[0]))/len(dd['shot'])
                     ))
           
    def plot(self, axes=None, label=None, suptitle=None, **kwargs):
        if suptitle is None:
            pl.suptitle("{0}, cc={1} sd={2} ".
                        format(self.name,self.cc,self.csd))               

        xd = np.arange(len(self.cc))
        #pl.plot(xd, self.cc, label=self.name, **kwargs)
        if axes != None: ax = axes
        else: ax=pl.gca()
        if label is None: label =self.name
        ax.plot(xd, self.cc,label=label, **kwargs)
        current_color = ax.get_lines()[-1].get_color()
        ax.errorbar(xd, self.cc, self.csd, ecolor=current_color, color=current_color)
        ax.set_xlim(xd[0]-0.1,xd[-1]+.1)
    def one_rms(self, phases):
        """ Return the standard deviation normalised to the cluster sds
            a point right on the edge of each sd would return 1
        """
        return(np.sqrt(np.average((twopi(self.cc-phases)/self.csd)**2)))
    def std(self, phase_array):
        """ Return the standard deviation normalised to the cluster sds
            a point right on the edge of each sd would return 1
            Need to include a mask to allow dead probes to be ignored
        """
        if not(hasattr(phase_array, 'std')):
            print('make phase_array into an np arry to speed up 100x')
            phase_array = np.array(phase_array.tolist())

        cc = np.tile(self.cc, (shape(phase_array)[0],1))
        csd = np.tile(self.csd, (shape(phase_array)[0],1))
        sq = (twopi(phase_array-cc)/csd)**2
        return(sqrt(average(sq,1)))

    def std_masked(self, phase_array, mask=None):
        """ Return the standard deviation normalised to the cluster sds
            a point right on the edge of each sd would return 1
            Innclude a mask to allow dead probes to be ignored
        """
        if mask is None: mask=np.arange(len(self.cc))

        if not(hasattr(phase_array, 'std')):
            print('make phase_array into an np array to speed up 100x')
            phase_array = np.array(phase_array.tolist())

        cc = np.tile(self.cc[mask], (shape(phase_array)[0],1))
        csd = np.tile(self.csd[mask], (shape(phase_array)[0],1))
        sq = (twopi(phase_array[:,mask]-cc)/csd)**2
        return(sqrt(average(sq,1)))

if __name__ == '__main__':

    # manually enter the mean and sd for the modes, called by color
    # n=1
    OM=[]  # old modes (pre MP2010)
    OM.append(Mode('N=1', N=1, NN=100, cc = [1.020, 1.488, 1.348, -0.080, 0.989], csd= [0.4, 0.2, 0.3, 0.20, 0.2 ]))
    #blue.csd= [0.077, 0.016, 0.048, 0.020, 0.008 ]


    OM.append(Mode('N=1,46747', N=1, NN=101, cc = [0.6, 1.7, 0.8, 0.2, 0.989], csd= [0.4, 0.2, 0.3, 0.20, 0.2 ]))
    # obscured by other noise....


    OM.append(Mode('N=1,or 0?', N=1, NN=102, cc = [1.0, -0.65, 1.84, 1.8, 1.95], csd= [0.2, 0.2, 0.3, 0.20, 0.2 ]))
    # very clear on 38100


    OM.append(Mode('N=1 - broad', N=1, NN=103, cc =[0.3, 0.25, 1.3, 2.0, 1.8], csd=[ 0.48, 0.5, 0.5, 0.5, 0.42]))

    OM.append(Mode('N=1 - residual', N=1, NN=104, cc =[1.2, 0.86, 1.8, -0.1, 1.0], csd=[ 0.2, 0.2, 0.3, 0.3, 0.3]))


    # n=0
    OM.append(Mode('N=0', N=0, NN=50, cc =[-1.146, -1.094, 0.608, 0.880, 0.164], csd=[ 0.48, 0.33, 0.34, 0.33, 0.42]))
    # mainly low 27s
    #lblue.csd=[ 0.048, 0.033, 0.034, 0.033, 0.042]


    OM.append(Mode('N=0 - fract', N=0, NN=51, cc =[0.3, -.75, 0.608, 0.880, 0.45], csd=[ 0.48, 0.33, 0.34, 0.33, 0.42]))

    OM.append(Mode('N=0 - fract1', N=0, NN=52, cc =[-2.1, -1.1, 0.5, 1.06, 0.17], csd=[ 0.3, 0.33, 0.34, 0.33, 0.25]))



    # n=2
    OM.append(Mode('N=2', N=2, NN=200, cc =[2.902, 2.217, 2.823, 1.021, 2.157], csd= [0.2, 0.2, 0.2, 0.3, 0.2 ]))
    #red.csd= [0.023, 0.006, 0.028, 0.025, 0.007 ]
    # saved as 200  with sumsd2 <10


    OM.append(Mode('N=2, LF', N=2, NN=201, cc =[2., 2.4, 2.0, 1.5, 2.2], csd= [0.2, 0.2, 0.2, 0.3, 0.2 ]))
    #red.csd= [0.023, 0.006, 0.028, 0.025, 0.007 ]
    # saved 201 with sumsd2 <10 


    OM.append(Mode('N=2, 60k', N=2, NN=202, cc =[2.3, 2.4, 1.7, 2.1, 1.7], csd= [0.3, 0.2, 0.2, 0.2, 0.2 ]))
    #red.csd= [0.023, 0.006, 0.028, 0.025, 0.007 ]
    # saved NN=202 sumsd2<10  - -model is 47634 (3928 overlap with  redlow)

    OM.append(Mode('N=2, 60k MP1', N=2, NN=204, cc =[0, 2.4, 1.7, 2.1, 1.7], csd= [5, 0.2, 0.2, 0.2, 0.2 ],shot_list=[54184,54185,54194,54195,54196,54197,54198]))
    # ref60, but allow for MP1 to be dead (large csd)

    OM.append(Mode('W1', N=2, NN=203, cc =[2.5, 0.3, -1.4, 0.7, 2.2], csd =[0.5, 0.5, 0.3, 0.3, .25]))
    # too rare to worry

    def f1(cc):
        return( -twopi(np.array(cc) + np.pi, offset=4))

    MP2010=[]  # old modes (pre MP2010)
    MP2010.append(Mode('N=2', N=2, NN=200, cc = f1([1.020, 1.488, 1.348, -0.080, 0.989]), csd= [0.4, 0.2, 0.3, 0.20, 0.2 ]))
    #blue.csd= [0.077, 0.016, 0.048, 0.020, 0.008 ]


    MP2010.append(Mode('N=2,46747', N=2, NN=201, cc = f1([0.6, 1.7, 0.8, 0.2, 0.989]), csd= [0.4, 0.2, 0.3, 0.20, 0.2 ]))
    # obscured by other noise....


    MP2010.append(Mode('N=2,or 0?', N=2, NN=202, cc = f1([1.0, -0.65, 1.84, 1.8, 1.95]), csd= [0.2, 0.2, 0.3, 0.20, 0.2 ]))
    # very clear on 38100


    MP2010.append(Mode('N=2 - broad', N=2, NN=203, cc =f1([0.3, 0.25, 1.3, 2.0, 1.8]), csd=[ 0.48, 0.5, 0.5, 0.5, 0.42]))

    MP2010.append(Mode('N=2 - residual', N=2, NN=204, cc =f1([1.2, 0.86, 1.8, -0.1, 1.0]), csd=[ 0.2, 0.2, 0.3, 0.3, 0.3]))


    # n=0
    MP2010.append(Mode('N=3', N=3, NN=300, cc =f1([-1.146, -1.094, 0.608, 0.880, 0.164]), csd=[ 0.48, 0.33, 0.34, 0.33, 0.42]))
    # mainly low 27s
    #lblue.csd=[ 0.048, 0.033, 0.034, 0.033, 0.042]


    MP2010.append(Mode('N=3 - fract', N=3, NN=301, cc =f1([0.3, -.75, 0.608, 0.880, 0.45]), csd=[ 0.48, 0.33, 0.34, 0.33, 0.42]))

    MP2010.append(Mode('N=-3  ', N=-3, NN=-301, cc =([-2.74, +2.1, -2.7, -2.3, -2.5]), csd=[ 0.48, 0.33, 0.34, 0.33, 0.42]))

    MP2010.append(Mode('N=3 - fract1', N=3, NN=302, cc =f1([-2.1, -1.1, 0.5, 1.06, 0.17]), csd=[ 0.3, 0.33, 0.34, 0.33, 0.25]))

    # was n=2
    MP2010.append(Mode('N=1', N=1, NN=100, cc =f1([2.902, 2.217, 2.823, 1.021, 2.157]), csd= [0.2, 0.2, 0.2, 0.3, 0.2 ]))
    #red.csd= [0.023, 0.006, 0.028, 0.025, 0.007 ]
    # saved as 200  with sumsd2 <10


    MP2010.append(Mode('N=1, LF', N=1, NN=101, cc =f1([2., 2.4, 2.0, 1.5, 2.2]), csd= [0.2, 0.2, 0.2, 0.3, 0.2 ]))
    #red.csd= [0.023, 0.006, 0.028, 0.025, 0.007 ]
    # saved 201 with sumsd2 <10 


    MP2010.append(Mode('N=1, 60k', N=1, NN=102, cc =f1([2.3, 2.4, 1.7, 2.1, 1.7]), csd= [0.3, 0.2, 0.2, 0.2, 0.2 ]))
    #red.csd= [0.023, 0.006, 0.028, 0.025, 0.007 ]
    # saved NN=202 sumsd2<10  - -model is 47634 (3928 overlap with  redlow)

    MP2010.append(Mode('N=1, 60k MP1', N=1, NN=104, cc =f1([0, 2.4, 1.7, 2.1, 1.7]), csd= [5, 0.2, 0.2, 0.2, 0.2 ],shot_list=[54184,54185,54194,54195,54196,54197,54198]))
    # ref60, but allow for MP1 to be dead (large csd)

    MP2010.append(Mode('W1?', N=1, NN=103, cc =f1([2.5, 0.3, -1.4, 0.7, 2.2]), csd =[0.5, 0.5, 0.3, 0.3, .25]))
    # too rare to worry

    # need to track down with MP1 should be with another shot
    MP2010.append(Mode('N=-1',N=-1, NN=-100, cc=[-0.416, -0.814, -1.386, -1.190, -1.147], csd = [4, 0.060, 0.041, 0.086, 0.072]))
    # from the N~0 cases where M was found but not M, in 65139
    MP2010.append(Mode('N~0',N=0, NN=0, cc=[-0.829, -0.068, -0.120, 0.140, -0.032],csd=  [3, 0.025, 0.044, 0.037, 0.029]))
    # and this from the other ones (N ~ -1)
    MP2010.append(Mode('N=-1',N=-1, NN=-101, cc=[-0.454, -0.775, -1.348, -1.172, -1.221],  csd=[ 3, 0.071, 0.048, 0.133, 0.113]))

    ideal_modes=[]
    ideal = np.load('pyfusion/ideal_toroidal_modes.npz')['subset']
    ideal = ideal[:,np.arange(5)]  # need to adjust
    for i in range(10):
        N=i-5
        ideal_modes.append(Mode('N={N}'.format(N=N),N=N, NN=i*(100), cc=ideal[i], csd=0.5*np.ones(len(ideal[0]))))

    srh_thesis = pickle.load(file('srh_thesis.pickle','r'))

    ind = None
    mode_list = MP2010
    mode_list = ideal_modes
    mode_list = srh_thesis
    mode=None
    threshold=None
    mask=None
    doM = False
    doN = False
    sel = np.arange(11,16)

    import pyfusion.utils
    exec(pyfusion.utils.process_cmd_line_args())

    if mode is None: mode = mode_list[0]
    if not(doM) and not(doN): raise ValueError('Need to choose doN=True and/or doM=True')

    if ind is None: ind = np.arange(len(dd['shot']))
    # the form phases = dd['phases'][ind,11:16] consumes less memory
    if (sel is not None) and  (np.average(np.diff(sel))==1):   # smarter version
        phases = dd['phases'][ind,sel[0]:sel[-1]+1]
    else:
        phases = dd["phases"][ind]
        if sel is not None:
            phases = phases.T[sel].T
    #phases = np.array(phases.tolist())

    sd = mode.std(phases)

    for mname in 'N,NN,M,MM'.split(','):
        if mname not in dd:
            use_dtype=int16
            minint = np.iinfo(use_dtype).min
            dd[mname] = minint*np.ones(len(dd['shot']),dtype=use_dtype)

    tot_set, tot_reset = (0,0)


    for mode in mode_list:
        if doN: mode.store(dd, threshold, mask=mask)
        if doM: mode.storeM(dd, threshold)

        tot_set   += mode.num_set
        tot_reset += mode.num_reset
    print('Total set = {t}, reset = {r}'.format(t=tot_set, r=tot_reset))

    """
    29th Nov method- obsolete

    sd = []
    sdlb = []
    sdred = []

    sh = []
    amp = []
    ind = []
    #arr=loadtxt('MP_27233_cf_syama.txt',skiprows=4)
    # arr=loadtxt('MP512all.txt',skiprows=4)
    # read in the delta phases from the aggregated pyfusion database
    # and build up a list of flucstrucs and the sum of squares relative to the
    # three modes.
    #fsfile='MP512all.txt'
    #fsfile='PF2_120229_MP_27233_27233_1_256.dat'

    ## let plot_text_pyfusion choose the file and skip
    #fsfile='PF2_120229_MP_50633_50633_1_256.dat'
    #skip=4
    hold=1
    dt=140e-6
    ind = None
    mode=redlow

    import pyfusion.utils
    exec(pyfusion.utils.process_cmd_line_args())
    #run -i ./examples/plot_text_pyfusion.py filename='MP_27233_cf_syama.txt' skip=4
    #run -i ./examples/plot_text_pyfusion.py filename=fsfile skip=skip
    #sys.argv = ['filename='+fsfile, 'skip='+str(skip)]

    #execfile("./examples/plot_text_pyfusion.py") 

    if ind is None: ind = np.arange(len(dd['shot']))
    phases = dd["phases"][ind]

    for i in ind:
        if (i % 100000) == 0: print(i),
        sd.append(sum((twopi(dd["phases"][i]-mode.cc)/mode.csd)**2))
        sh.append(dd["shot"][i])
    """
    """
    exec(pyfusion.utils.process_cmd_line_args())

    # find the indices of modes within a  short distance from the classification
    neq0 = (array(sdlb) < 1).nonzero()[0]
    neq1 = (array(sd) < 1).nonzero()[0]
    neq2 = (array(sdred) < 1).nonzero()[0]

    # do a crude plout by color=mode only
    msize=40
    print(hold)
    pl.scatter(dt+ds['t_mid'][neq0],fsc*ds['freq'][neq0],hold=hold,label='N=0',c='cyan',s=msize)
    pl.scatter(dt+ds['t_mid'][neq1],fsc*ds['freq'][neq1],label='N=1',s=msize)
    pl.scatter(dt+ds['t_mid'][neq2],fsc*ds['freq'][neq2],label='N=2',c='red',s=msize)

    pl.legend()
    pl.suptitle(filename)
    #for x in array([ds['t_mid'][neq0],1e3*ds['freq'][neq0],neq0]).T.tolist(): text(x[0],x[1],int(x[2])) 
    # inds=[5106,5302,5489,1228,1233,1236,478,657,1260] ; average(arr[inds,8:],0); arr[inds,8:]

    """
