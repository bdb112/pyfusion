from numpy import shape, array
import csv

def debug_(debug,level=1,key=None,msg=''):
    """ Nice debug function, breaks to debugger if debug>level, or if
    key is asubstring of the string value in debug.  This way, one
    debug input can control several breakpoints, without too much
    recoding.  Once a possible dedbug point is reasonably OK, its
    level can be raised, so that it is still accessible, but only at
    high debug levels.  Most calls can be like debug_(debug,3) - where
    the 3 reduces the priority of this relative to other calls that
    share the same debug parameter.

    What does this mean?  Say there are three possible calls to debug
    in a piece of code where the debug argument is passed in at one
    point only.  Then _debug(debug,3) will break out if debug is very
    high, whereas debug_(debug) will more easily break out.

    key is a string which allows string values of debug to selectively 
    trigger breakout:

    e.g key='lookup' and debug='lookup,open' then execution breaks
    """
    import warnings
    if type(debug) == type('abc'):
        if key.find(debug) < 0: return            
    
    elif debug<level: return

    try:
        """
        # pydbgr is nice, but only works once - then debug is ignored
        print('try to import pydbgr')
        from pydbgr.api import debug
        'debugging '+ msg + ' in correct frame,  c to continue '        
        print('execute debug()')
        debug()

    except ImportError:
        # pydb is nice, but overwrites the history file.
        # could import readline, and save history in a backup UNLESS it 
        # is < 50 or so
        # 
        import pydb
        pydb.set_trace(['up'])
        'debugging '+ msg + ' in correct frame,  c to continue '        
    except ImportError:
    """
        from sys import _getframe
        frame = _getframe()
        print(frame.f_lineno,frame.f_code)
        frame = _getframe().f_back
        print(frame.f_lineno,frame.f_code)
        from ipdb import set_trace
        print('debugging '+ msg)
        set_trace(_getframe().f_back)
        ' use up to get to frame, or c to continue '
    except ImportError:
        warnings.warn('error importing ipdb: try pdb instead')
        from pdb import set_trace
        set_trace()
        'debugging '+ msg + ' use up to get to frame, or c to continue '
    except ImportError:
        print('failed importing debugger pdb')


def read_csv_data(file_or_list,debug=0,columns=None, header=1, dialect='excel', maxlines=None, openMode='rU'):
    """ Reads csv files with given dialect, returns dictionary of arrays of strings
    If there is a header row, get number of columns from that.
    about 130k 4 value lines/second (i5/760)
    Works for LTSPice "export" files, and Mauro's test data
    """
    import csv
    if len(shape(file_or_list)) == 0:
        if '.bz2' in file_or_list:
            import bz2
            fd = bz2.BZ2File(file_or_list, mode=openMode)
        else:
            fd = open(file_or_list,openMode)
    else:
        fd = file_or_list
    reader=csv.reader(fd,dialect=dialect)
    names = []
    data = []
    for (r,row) in enumerate(reader):
        if maxlines != None and r>maxlines: break
        if debug>1: print(r,row)

        if (r < (header-1)):
            pass # skip before header
            
        elif (r == header-1) or (header is None):
            # this code deals both with header lines and data lines - messy!
            for (c,col) in enumerate(row):
                if columns is None: 
                    data.append([])
                    if header is None:
                        names.append(str("col%2.2d" %(c)))
                        data[c].append(col)
                    else: 
                        names.append(col)
                elif c in columns: 
                    if header is None:
                        names.append(str("col%2.2d" %(c)))
                        data[c].append(col)
                    else: 
                        names.append(col)
            if (r+1 == header) and (len(names) == 0):          
                raise ValueError('header row specified, but no columns found')
                
        else:
            if len(row) > len(names):
                raise ValueError('Too many columns ({0}>{1}) in row {2} [{3}]'
                                 .format(len(row), len(names), r, row))
            for (c,col) in enumerate(row):
                data[c].append(col)

    if len(file_or_list)==1: fd.close()
    data_dict = {}
    for (n,name) in enumerate(names):
        data_dict.update({name: array(data[n])})

    debug_(debug)    
    return(data_dict)    
