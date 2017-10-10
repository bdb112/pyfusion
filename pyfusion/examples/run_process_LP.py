"""
Convenience script to run process_Langmuir over a range of shots.

to replace/add keywords simply, use replace_kw (if you use dict() remember to quote '..=dict(...)'
   run pyfusion/examples/run_process_LP.py  shot_list='[[20160309,7],[20160310,7]]' "replace_kw=dict(t_range=[1,1.1])"

# this example is about 2.8 (11 if ALLI) seconds. - beginning of shot 309_52
_PYFUSION_TEST_@@ select='[0,1]' replace_kw='dict(t_range=[0.88,0.9])' shot_list='[[20160309, 52]]' lpdiag='W7X_L5{s}_LP0107'
"""

import pyfusion
from pyfusion.data.process_swept_Langmuir import Langmuir_data, fixup
# rename the function to avoid name clash
from pyfusion.data.shot_range import shot_range as expand_shot_range


_var_defaults="""
dev_name='W7X'
shot_list=[[20160309, s] for s in [41,42,43]]  # [13,41,42,43]] 13:many start later
shot_list=[[20160308, 27]]
shot_list=[]
shot_list.extend([[20160309, s] for s in [44,49,50,51,52,45,46]])
shot_list=[[20160309, 52]]
shot_list.extend([[20160308, 32], [20160309, 35]])
shot_list.extend([[20160310, s] for s in [38, 39]])
replace_kw={}
#replace_kw=dict(t_range=[1.0,1.01])   # selectively override proc_kwargs
select=None                            # e.g. select=[0,1]
exception=Exception
#  t_comp was [0.85,0.88] for a while (to work with tiny files), but it fails on 0309_22
# shot really have this default to closer to 0 - e.g. [0,0.1]
proc_kwargs = dict(overlap=2,dtseg=2000,initial_TeVfI0=dict(Te=30,Vf=5,I0=None),fit_params=dict(alg='amoeba',maxits=300,lpf=21,esterr=1,track_ratio=1.2),filename='/tmp/*2k2',threshold=0.001,t_comp=[.81,.84]) # debug ,t_range=[0.5,0.51])
#select=1
lpdiag='W7X_L5{s}_LPALLI'
seglist=[3,7]
"""

exec(_var_defaults)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

for k in replace_kw:  # override and
    proc_kwargs.update({k: replace_kw[k]})

for dateshot in shot_list:
    for seg in seglist:
        try:
            LP = Langmuir_data(dateshot, lpdiag.format(s=seg),'W7X_L5UALL')
            if select is not None:  # selected channels
                LP.select = select

            print(proc_kwargs)
            LP.process_swept_Langmuir(**proc_kwargs)
        except exception as reason:
            pyfusion.logger.error('failed reading shot {shot}, segment {seg} \n{r}'
                                  .format(shot=dateshot, seg=seg, r=str(reason)))
        #LP.process_swept_Langmuir(threshchan=0,t_comp=[0.85,0.87],filename='*2k2small')
