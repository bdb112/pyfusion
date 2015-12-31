import pyfusion

_var_defaults="""
diag_name = 'MP'
dev_name = 'LHD'
rng = None
"""
exec(_var_defaults)

# ideally should be a direct call, passing the local dictionary
import pyfusion.utils
exec(pyfusion.utils.process_cmd_line_args())


dev=pyfusion.getDevice(dev_name)

def getdat(shot):
    return(dev.acq.getdata(shot, diag_name))

print(pyfusion.utils.find_last_shot(getdat, range=rng))

