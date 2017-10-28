import time, datetime, calendar

def utc_ns(s, fmt=None):
    """ return utc corresponding to various string formats 
    20160302.14:55   20160302 14:55:30  
    e.g.  utc_ns(['20160303.143659','20160303.143661'])
    """
    #   time.strptime('20160302.14:55','%Y%m%d.%H:%M'
    if isinstance(s, (list, tuple)):
        return([utc_ns(s[0]), utc_ns(s[1])])
    if fmt is not None:
        pass  # use it!
    elif len(s) > 18:
        fmt = '%Y-%m-%d %H:%M:%S:%f'
    elif len(s) == 17:
        fmt = '%Y%m%d %H:%M:%S'
    elif len(s) == 15:
        fmt = '%Y%m%d.%H%M%S'
    elif len(s) == 14:
        fmt = '%Y%m%d.%H:%M'
    elif len(s) == 13:
        fmt = '%Y%m%d.%H%M'
    elif len(s) == 11:
        fmt = '%Y%m%d.%H'
    elif len(s) == 8:
        fmt = '%Y%m%d'
    else:
        raise ValueError('time string {s} has an unexpected length {l}'
                         .format(s=s, l=len(s)))
    tm = time.strptime(s, fmt)
    #secs = time.strftime('%s', tm)  # seems like %s is not standard?
    secs = calendar.timegm(tm)
    return(int(1e9) * int(secs))

def utc_GMT(ns, fmt='%Y%m%d %H:%M:%S.%f'):
    """ return the time in a GMT string, based on the format.  
    In order to match to the nanosecond, use format
    """
    if isinstance(ns, (list, tuple)):
        return([utc_GMT(n) for n in ns])
    tstrct = time.gmtime(ns/1e9)
    if (ns % 1000000000L) == 0:
        fmt = fmt.replace('.%f','')
    tstrct = datetime.datetime.utcfromtimestamp(ns/1e9)
    tstrg = datetime.datetime.strftime(tstrct, fmt)
    if (ns % 1000) != 0:
        tstrg += str('{nspart:03d}'.format(nspart=ns % 1000))
    return(tstrg)
