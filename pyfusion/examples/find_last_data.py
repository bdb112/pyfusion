import pyfusion

_var_defaults="""
diag_name = 'MP6'
dev_name = 'LHD'
rng = [50639,60000]
"""
exec(_var_defaults)

# ideally should be a direct call, passing the local dictionary
import pyfusion.utils
exec(pyfusion.utils.process_cmd_line_args())


dev=pyfusion.getDevice(dev_name)

def getdat(shot):
    return(dev.acq.getdata(shot, diag_name, quiet=1, exceptions=Exception))

print(pyfusion.utils.find_last_shot(getdat, srange=rng))
