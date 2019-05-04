#_PYFUSION_TEST_@@SCRIPT
run pyfusion/examples/W7X_get_shot_times.py shot=[20170912,21] ECH=W7X_ECH_Rf_C5
clf()
run -i pyfusion/examples/plot_signals diag_name='W7X_LTDU_LP20_I' shot_number=[20170912,21] hold=1
run  -i pyfusion/examples/plot_signals diag_name='W7X_WDIA_TRI' shot_number=[20170912,21] hold=2 fun2=None 'plotkws=dict(scale=1.25,color="r",lw=2)'
run  -i pyfusion/examples/plot_signals diag_name='W7X_ROG_CONT' shot_number=[20170912,21] hold=2 fun2=None 'plotkws=dict(color="g",lw=2, scale=.1)'
xlim(0,0.4)
ylim(0,2)  # ylim(0.8,2.2)
