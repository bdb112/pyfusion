import time, calendar

def utc_ns(s):
    """ return utc corresponding to the string s format 20160302.14:55  
    e.g.  utc_ns(['20160303.143659','20160303.143661'])
    """
    #   time.strptime('20160302.14:55','%Y%m%d.%H:%M'
    if isinstance(s, (list, tuple)):
        return([utc_ns(s[0]), utc_ns(s[1])])

    if len(s) == 15:
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
