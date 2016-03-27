# overlay Te, Ne, ECH for 20160310:9
import pickle
run -i pyfusion/examples/plot_signals.py  dev_name='W7X' diag_name=W7X_TotECH shot_number=[20160310,9] hold=0 sharey=2
figure()
rcParams['lines.linewidth']=2
q=1.e-19
mi=1.67e-27
A=1.8e-6
tresults=pickle.load(open('LP_20160310_9_W7X_L57_LP1_I_5120_20160311.pickle'))
ax=subplot(1,1,1)
ax.plot(array(tresults).T[0],1/(0.6*q*A)*sqrt(mi/(q))/1e18*array(tresults).T[3]/sqrt(array(tresults).T[1]),label='ne_18')
ylim(0,8)
ax2=ax.twinx()
ax2.plot(array(tresults).T[0],array(tresults).T[1],'r',label='Te')
ax2.plot(data.timebase-4,data.signal/1000,'g',label='totECH (MW)')
ylim(0,80)
xlim(0.5,2.2)
ax2.legend()
ax.legend(loc='upper left')
title('High Performance, 4MW: 2016 03 10 9  L57_LP10')
show

