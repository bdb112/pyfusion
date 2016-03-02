import time

def utc_ns(s):
    """ return utc corresponding to the string s format 20160302.14:55  """
    #   time.strptime('20160302.14:55','%Y%m%d.%H:%M'
    if isinstance(s, (list, tuple)):
        return([utc_ns(s[0]), utc_ns(s[1])])

    if len(s) == 14:
        fmt = '%Y%m%d.%H:%M'
    if len(s) == 13:
        fmt = '%Y%m%d.%H%M'
    elif len(s) == 11:
        fmt = '%Y%m%d.%H'
    elif len(s) == 8:
        fmt = '%Y%m%d'
    else:
        raise ValueError('time string {s} has an unexpected length {l}'
                         .format(s=s, l=len(s)))
    tm = time.strptime(s, fmt)
    secs = time.strftime('%s', tm)
    return(int(1e9) * int(secs))
