#bands=2
# paste below the line (only works with mouse middle - not %paste)
#Also, gen_fs_bands needs outfile=0 in the defaults, until process_cmd_line_args is fixed.
figure()
fn='PF2_160221_50136_{n}b'.format(n=bands)
run -i pyfusion/examples/gen_fs_bands.py n_samples=None df=2 exception=() max_bands=bands dev_name="HeliotronJ" 'time_range="default"' seg_dt=1 overlap=2.5  diag_name='HeliotronJ_MP_array' shot_range=[50136] info=0 "outfile=fn" exception=()
run -i pyfusion/examples/merge_text_pyfusion.py  file_list=[fn] pyfusion.DEBUG=0 exception=Exception
from pyfusion.visual import window_manager, sp, tog
sp(dd,'t_mid','freq','amp','a12',size_scale=.05,dot_size=50)
(xl,yl)=(xlim(),ylim())
run pyfusion/examples/plot_specgram.py dev_name='HeliotronJ' shot_number=50136 diag_name=HeliotronJ_MP_array hold=1
xlim(xl); ylim(0,yl[1])
suptitle(fn)
