""" Example 3 - SQL example, similar to the python Dictionary of arryas version in example4.py
Plot all flucstrucs found, and the select the large rotating modes.

SQLAchemy has many ways to access data from python.  This is not the neatest of them, but I chose 
it because the queries match textbook 'text mode' SQL query-based approach.

== Aside on alternatives ==
One alternative SQLAlchemy access model means tables (everthing in fact) look like objects 
Using this mode, you could write more concisely 
    plt.plot(H1.t_mid, H1.freq)
Which is really nice and simple, unlike the messy iteration in this example.
    [row['t_mid'] for row in alldata]

Queries would contain phrases like and_(H1.freq.between(4,20), H1.comment.line('%argon%')
which are quite different to text mode, but easier to compose 'on the fly'
   
"""
from pyfusion.data.DA_datamining import DA
from sqlalchemy import create_engine 
import matplotlib.pyplot as plt
import os

if not os.path.isfile('H1_766.sqlite'):  #  if not there, make it
    myDA = DA('H1_766.npz')
    myDA.to_sqlalchemy('sqlite:///H1_766.sqlite', newfmts=dict(phases='dphi_{i:02d}'),
                       mytable='H1_766', chunk=5000, n_recs=1e9)

engine = create_engine('sqlite:///H1_766.sqlite')
conn = engine.connect()
result = conn.execute('select shot, t_mid, freq, a12, amp from H1_766')
alldata = result.fetchall() # all the instances, and overplot in feint colour
plt.plot([row['t_mid'] for row in alldata], [row['freq'] for row in alldata],'o',alpha=0.02)
plt.xlabel('time (s)') ; plt.ylabel('freq (kHz)')

# Ex 3a. this time, select the large rotating modes, (note - we combine two lines here, 
#    even though it makes it less readable) 
plt.figure()
sel = conn.execute('select shot, t_mid, freq, a12, amp from H1_766 where amp> 0.05 and a12>0.7').fetchall()
#  for the selected data,  plot freq against t_mid in red circles('r'), whose size reflects the amplitude 
plt.scatter([row['t_mid'] for row in sel], [row['freq'] for row in sel],[300*row['amp'] for row in sel],'r')
plt.show()
