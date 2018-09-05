#!/usr/bin/python3
""" To (selectively) print matching lines in a flattened Minerva map dictionary
    or any other dictionary, matching python re. regexp wild_card

    wild_card:
    excl: typically used to exclude time dimension to avoid many 'spurious'
             differences
Example:
run pyfusion/examples/print_dict_tree.py wild_card=".*scr.*istor.val.*" shot=[20180719,37]
 modeFactor.va  actor.va will pick out volt channels


if shot is an integer that is big enough to to be a utc, assume it is
 
Result is left in matches, and saved in /tmp

"""
from pyfusion.acquisition.W7X.get_url_parms import MinervaMap, flatten_dict, flatten_dict_by_concatenating_keys, iteritems_nested
from pyfusion.utils.time_utils import utc_ns  # for convenience in terminal input
import numpy as np
import sys
import re

sys.path.append('/home/bdb112/python')
from pyfusion.utils.process_cmd_line_args import process_cmd_line_args

_var_defaults = """
wild_card = 'istor.val'
date_string =''
shot=(20180718,37)
excl='imensions' 
"""
# grab the first arg if it does not contain an "="
def __help__():  # must be before exec() line
    print(__doc__)
    print('local help routine for sqlplot!')

exec(_var_defaults)
exec(process_cmd_line_args())
wc = r'.*istor.val.*'
mm = MinervaMap(shot)
op = ['{k}: {v}'.format(k=k, v=v) for k,v in six.iteritems(flatten_dict(mm.parmdict))]
if ('*' not in wild_card) and ('+' not in wild_card):
    wild_card = '.*' +  wild_card + '.*'

print(mm.parm_URL + ', valid since ' + mm.get_parm('parms/validSince/values')[0])

rgx = re.compile(wild_card)
matches = rgx.findall('\n'.join(op))
print('{lm} matches to regexp "{wc}"'.format(lm=len(matches), wc=wild_card))
for lin in np.sort(matches):
    print(lin)
print('see matches or /tmp/parm_URL* for more detail')
with open('/tmp/parm_URL_{v}'.format(v=mm.parm_URL.split('_')[-2]), 'w') as fout:
    fout.writelines('\n'.join([m for m in np.sort(matches) if not excl in m]))
