""" From get_basic_diagnostics.py  - maybe just use that?
However get_basic_diagnostics gets diag just for the given times. (which si the main thing)
"""
            varname = info['name']
            subfolder = info['format'].split('@')[0]
            filepath = os.path.sep.join([localigetfilepath,subfolder,info['format']])
            if ':' in varname: (oper,varname) = varname.split(':')
            else: oper = None

            if info['format'].find('.csv') > 0:
                try:
                    test=lhd_summary.keys()
                except:    
                    csvfilename = acq_LHD+'/'+info['format']
                    if pyfusion.DBG() > 1: print('looking for lhd summary in' + csvfilename)
                    if not os.path.exists(csvfilename):
                        csvfilename += ".bz2"
                    print('reloading {0}'.format(csvfilename))
                    lhd_summary = read_csv_data(csvfilename, header=3)
                    # should make this more formal - last shots
                    # from an 'extra' file, and finally, from shot info
                if shot>117000: # fudge to get latest data
                    lhd_summary = np.load(acq_LHD+'/LHD_summary.npz')['LHD'].tolist()
                    print('loading newer shots from a separate file - fix-me')
                    #  val = lhd_summary[varname][shot-70000]    # not needed
                #  else:
                val = lhd_summary[varname][shot]    
                valarr = np.double(val)+(times*0)
            else:    
                debug_(max(pyfusion.DEBUG, debug), level=4, key='find_data')
                try:

                    dg = igetfile(filepath, shot=shot, debug=debug-1)
                except IOError:
                    try:
                        dg = igetfile(filepath+'.bz2', shot=shot, debug=debug-1)
                    except IOError:
                        try:
                            dg = igetfile(filepath + '.gz', shot=shot, debug=debug-1)
                        except exception, details:
                            if debug>0: print('diag at {fp} not found'
                                              .format(fp=filepath))
                            print(details)
                            dg=None
                            #break  # give up and try next diagnostic
                if dg is None:  # messy - break doesn't do what I want?
                    valarr=None
                else:
                    nd=dg.vardict['DimNo']
                    if nd != 1:
                        raise ValueError(
                            'Expecting a 1 D array in {0}, got {1}!'
                            .format(dg.filename, nd))

                    # pre re. w = np.where(np.array(dg.vardict['ValName'])==varname)[0]
                    matches = [re.match(varname,nam) 
                               != None for nam in dg.vardict['ValName']]
                    w = np.where(np.array(matches) != False)[0]
                    # get the column(s) of the array corresponding to the name
                    if (oper in 'sum,average,rms,max,min'.split(',')):
                        if oper=='sum': op = np.sum
                        elif oper=='average': op = np.average
                        elif oper=='min': op = np.min
                        elif oper=='std': op = np.std
                        else: raise ValueError('operator {o} in {n} not known to get_basic_diagnostics'
                                               .format(o=oper, n=info['name']))
                        valarr = op(dg.data[:,nd+w],1)
                    else:
                        if len(w) != 1:
                            raise LookupError(
                                'Need just one instance of variable {0} in {1}'
                                .format(varname, dg.filename))
                        if len(np.shape(dg.data))!=2:
                           raise LookupError(
                                'insufficient data for {0} in {1}'
                                .format(varname, dg.filename))
                             
                        valarr = dg.data[:,nd+w[0]]

                    tim =  dg.data[:,0] - delay

                    if oper == 'ddt':  # derivative operator
                        valarr = np.diff(valarr)/(np.average(np.diff(tim)))
                        tim = (tim[0:-1] + tim[1:])/2.0

                    if oper == 'ddt2':  # abd(ddw)*derivative operator
                        dw = np.diff(valarr)/(np.average(np.diff(tim)))
                        ddw = np.diff(dw)/(np.average(np.diff(tim)))
                        tim = tim[2:]
                        valarr = 4e-6 * dw[1:] * np.abs(ddw)

                    if (len(tim) < 10) or (np.std(tim)<0.1):
                        raise ValueError('Insufficient points or degenerate'
                                         'timebase data in {0}, {1}'
                                         .format(varname, dg.filename))

                    valarr = (stineman_interp(times, tim, valarr))
                    w = np.where(times > max(tim))
                    valarr[w] = np.nan

            if valarr != None: vals.update({diag: valarr})
