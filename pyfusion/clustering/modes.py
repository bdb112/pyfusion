import numpy as np
import pylab as pl

""" 
17 Nov
This version allows omitting channels by adding two differences.  
This is accomplished by using a matrix for mask instead of a vector, allowing much more than just omission.

Big change - the phases are passed directly - the calling routine does the masking both in mask and shot. Goes with mode_identify_script.py

Example - adding two deltas (last two), and ignoring one(the first
dot(array([200,1,20,20]), array([[0,1, 0, 0],[0,0, 1, 1]]).T)
==> array([ 1, 40])
This case illustrates matching with the entire cc
If we want to  match with part of the cc, then we use csel=(vector)

"""


def twopi(x, offset=np.pi):
    return ((offset+np.array(x)) % (2*np.pi) -offset)

def askif(message, quiet=0):
    """ give messge, ask if continue (if quiet=0) or go on 
         anyway (quiet-1)
    """
    if quiet == 0: suffix =  " Continue?(y/N) "
    else: suffix = "continuing, set quiet=0 to stop..."
    if quiet==0: 
        ans = raw_input("Warning: {0}: {1}".format(message, suffix))
        if len(ans) ==0 or ans[0].lower() == 'n':
            raise LookupError(message)
        elif ans[0] in '0123456789':
            return(int(ans))

class Mode():
    def __init__(self,name, N, NN, cc, csd, id=None, threshold=None, shot_list=[],MP2010_trick=False):
        self.name = name
        self.N = N
        self.NN = NN
        self.id = id   # this should be unique - should rewrite to force it
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

    def store(self, phases, dd, inds, csel=None, mask=None, threshold=None,Nval=None,NNval=None,shot_list=None,quiet=0):
        """ store coarse and fine mode (N, NN) numbers according to a threshold
        std a selection of channels (mask) and an optional shot_list.
       If None the internal shot_list is used which would have been set or
       defaulted to [] at __init__

       mask selects the probes - if None, select all.
        """
        if shot_list is None: shot_list = self.shot_list
        if threshold is None: threshold = self.threshold
        else: self.threshold=threshold  # save the last manually set.

        if Nval is None: Nval = self.N
        if NNval is None: NNval = self.NN

        uniq_modes = np.unique(dd['mode_id'][np.where(dd['mode_id']>=0)[0]])
        if self.id is None:
            if len(np.unique(dd['mode_id'])) == 0:
                self.id = 100
            else:
                self.id = np.max(np.unique(dd['mode_id']))+1

        while self.id in uniq_modes: 
            new_num = askif('mode_id {0} already used in this data'
                            'set - enter new number or y for next, n to abort '
                            .format(self.id),quiet=quiet)
            if newnum is None:  # none is the return from 'yes'
                self.id = np.max(uniq_modes)+1
            else: 
                self.id = new_num

        if not(hasattr(phases,'std')):
            askif('convert phases to nd.array?',quiet=quiet)
            phases = np.array(phases)

        if mask is None: mask = np.identity(np.shape(phases)[1])
        if csel is None: csel = np.arange(len(self.cc))

        sd = self.std(phases, csel=csel, mask=mask)

        w = np.where(sd<threshold)[0]

        # normally apply to all shots, but can restrict to a particular
        # range of shots - e.g. dead MP1 for shot 54186 etc.
        if shot_list != []:  # used to be None  - be careful, there are
                             # two shot_list's one in mode, one input here
            where_in_shot = []
            for sht in shot_list:
                ws = np.where(dd['shot'][inds[w]] == sht)[0]
                where_in_shot.extend(w[ws])  # relative to inds
            # this unique should not be required, but if the above logic
            # is changed, it might
            w = np.unique(where_in_shot)    

        if len(w) == 0: 
            print('threshold {th:.2g} is too low for phases: '
                  'minimum std for {m} is {sd:.1f}'
                  .format(th=threshold, m=self.name, sd=1*np.min(sd))) # format bug fix
            return()

        w_already = np.where(dd['NN'][inds[w]]>=0)[0]
        if len(w_already)>0:
            (cnts,bins) = np.histogram(dd['NN'][inds[w]], np.arange(-0.5,1.5+max(dd['NN'][inds[w]]),1))
            amx = np.argsort(cnts)
            print("NN already set in {0}/{1} locations {2:.1f}% of all data"
                  .format(len(w_already), len(w),
                          100*len(w_already)/float(len(dd['shot']))))
            print("NN={0} is most frequent ({1} inst.)"
                  .format(amx[-1],cnts[amx[-1]]))
            fract = len(w_already)/float(len(w))
            if fract>0.2: askif("{0:.1f}% of intended NN insts already set?".
                                format(fract*100),quiet=quiet)

        self.num_set += len(w)
        self.num_reset += len(w_already)

        dd['NN'][inds[w]]=NNval
        dd['N'][inds[w]]=Nval
        dd['mode_id'][inds[w]]=self.id
        print("N={N}: set {s:.2f}%, total N set is now {t:.2f}%".
              format(s=100*float(len(w))/len(dd['shot']),N=Nval,
                     t=100*float(len(np.where(dd['N']>min(dd['N']))[0]))/len(dd['shot'])
                     ))
           
    def storeM(self, dd, threshold=None, mask=None, Mval=None,MMval=None,shot_list=None,quiet=0):
        raise ValueError('Not updated to phases as a var yet')
        
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
                ws = np.where(dd['shot'][w] == sht)[0]
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
            if fract>0.2: askif("{0:.1f}% of intended MM insts already set?".
                                format(fract*100),quiet=quiet)

        self.num_set += len(w)
        self.num_reset += len(w_already)

        dd['MM'][w]=MMval
        dd['M'][w]=Mval
        print("set {s:.1f}%, total M set is now {t:.1f}%".
              format(s=100*float(len(w))/len(dd['shot']),
                     t=100*float(len(np.where(dd['M']>=0)[0]))/len(dd['shot'])
                     ))
           
    def plot(self, axes=None, label=None, csel=None, color=None, suptitle=None, **kwargs):
        """ plot a mode showing its SD as error bars
        """
        if csel is None: csel = np.arange(len(self.cc))
        if suptitle is None:
            pl.suptitle("{0}, cc={1} sd={2} ".
                        format(self.name,self.cc[csel],self.csd[csel]))               

        xd = np.arange(len(self.cc[csel]))
        #pl.plot(xd, self.cc, label=self.name, **kwargs)
        if axes != None: ax = axes
        else: ax=pl.gca()
        if label is None: label =self.name
        ax.plot(xd, self.cc[csel],label=label, color=color, **kwargs)
        current_color = ax.get_lines()[-1].get_color()
        ax.errorbar(xd, self.cc[csel], self.csd[csel], ecolor=current_color, color=current_color,**kwargs)
        ax.set_xlim(xd[0]-0.1,xd[-1]+.1)

    def hist(self, phases, first_std, NDim=None, n_bins=20, n_iters=10, histtype='bar',linewidth=None, equal_bins=False):
        """  
        This histogram can bin non-uniformly so that a uniform random 
        distribution will have a uniform number of counts. (equal_bins=False)

        histtype='stepfilled doesn't work
        Use linewidth=0 instead (now the default for nonequal bins)
        
        """
        if (not equal_bins) and (linewidth is None):
            linewidth=0
        NDim = np.shape(phases)[1]
        dist = self.std(phases)
        if equal_bins:
            hst = pl.hist(dist, bins=n_bins,log=1,histtype=histtype, 
                    elinewidth=None)
            return(hst)
        bins = [0,float(first_std)]
        while len(bins) < n_bins:
            bins.append(2*bins[-1] - bins[-2])
            for iter in range(n_iters):
                corrected_width = (
                    (bins[-2]-bins[-3]) * 
                    # this is ratio of the definite integrals of s**NDim-1
                    # for the last two bins, converted from counts/unit s to to counts
                    ((bins[-2]**NDim - bins[-3]**NDim)/(bins[-2] - bins[-3]))/
                    ((bins[-1]**NDim - bins[-2]**NDim)/(bins[-1] - bins[-2])))
                # print(corrected_width)
                bins[-1]=bins[-2] + corrected_width
        if np.min(dist)>np.max(bins):
            raise ValueError('no counts in the first {n} bins up to {lastbin},'
                             ' increase first bin size (first_std)'
                             .format(n=len(bins), lastbin=bins[-1]))
        hst = pl.hist(dist, bins=bins,log=1,histtype=histtype, 
                      linewidth=linewidth)
        maxsd = np.sqrt(np.max(self.csd**2))
        max_valid_s = 1.5/maxsd  # I would have thought Pi/maxsd
        pl.semilogy([max_valid_s, max_valid_s],pl.ylim(),'r--',linewidth=2)
        pl.show
        return(hst)

    def one_rms(self, phases):
        """ Return the standard deviation normalised to the cluster sds
            a point right on the edge of each sd would return 1
        """
        return(np.sqrt(np.average((twopi(self.cc-phases)/self.csd)**2)))

    def old_std(self, phase_array, mask=None):
        """ Return the standard deviation normalised to the cluster sds
            a point right on the edge of each sd would return 1
            Need to include a mask to allow dead probes to be ignored

            the following should return [1.,0.5]
            ml[0].std(array([ml[0].cc+ml[0].csd,ml[0].cc+0.5*ml[0].csd]))
            masl selects the channels, but only for the modes - the phases
            data is already selected.
        """
        if not(hasattr(phase_array, 'std')):
            print('make phase_array into an np.array to speed up 100x')
            phase_array = np.array(phase_array.tolist())

        phstype = type(phase_array[0][0])     # maybe less of a memory hog if we use the same type as phases
        cc = np.tile(self.cc.astype(phstype), (np.shape(phase_array)[0],1))
        csd = np.tile(self.csd.astype(phstype), (np.shape(phase_array)[0],1))
        sq = (twopi(phase_array-cc)/csd)**2
        return(np.sqrt(np.average(sq,1)))

    def std(self, phase_array, csel=None, mask=None):
        """ Return the standard deviation normalised to the cluster sds
            a point right on the edge of each sd would return 1
            mask replaced by the mask matrix to allow dead probes to be ignored
            or compbined with others

            the following should return [1.,0.5]
            ml[0].std(array([ml[0].cc+ml[0].csd,ml[0].cc+0.5*ml[0].csd]))
            mask selects the channels, but only for the modes - the phases
            data is already selected.
        """
        if mask is None: mask=np.identity(len(self.cc))
        if csel is None: csel = np.arange(len(self.cc))

        if not(hasattr(phase_array, 'std')):
            print('make phase_array into an np array to speed up 100x')
            phase_array = np.array(phase_array.tolist())

        cc_t = np.tile((np.dot(self.cc[csel],mask)),
                     (np.shape(phase_array)[0],1))
        # for csd, sqrt sum squares.
        csd_t = np.tile(np.sqrt(np.dot(self.csd[csel]**2,mask)),
                      (np.shape(phase_array)[0],1))
        sq = (twopi(phase_array-cc_t)/csd_t)**2
        return(np.sqrt(np.average(sq,1)))

