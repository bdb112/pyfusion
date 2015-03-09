"""Quick way to extract basic summary data
""" 
import pg8000
from pg8000 import DBAPI
import numpy as np

conn = DBAPI.connect(host="egdb.lhd.nifs.ac.jp", user="plasma", password="", database="db1")
cursor = conn.cursor()

vars='nShotnumber,dDatacreationTime,MagneticField,MagneticAxis,Quadruple,GAMMA'

varlist = vars.split(',')

LHD = {}
dim = 140000


varlist.remove('nShotnumber')

LHD.update(dict(nShotnumber = -1 + np.zeros(dim, dtype = np.int32)))

varlist.remove('dDatacreationTime')
LHD.update(dict(dDatacreationTime = np.array(dim * ['                   '])))

for k in varlist:
    LHD.update({k: np.nan+np.zeros(dim, dtype=np.float32)})

sql=str('select {vars} from explog2 where nshotnumber between 117000 and 200000 order by nShotnumber'
        .format(vars=vars))

cursor.execute(sql)

varlist = vars.split(',')
for row in cursor.fetchall():
    shot = row[0]
    for (i,var) in enumerate(varlist):
        if 'Time' in var:  #  in UTC at present
            if row[i] is None: 
                LHD[var][shot] = 'None'
            else:
                LHD[var][shot] = row[i].strftime('%Y-%m-%d %H:%M:%S')

        else:
            LHD[var][shot] = row[i]
                
np.savez_compressed('lhd_summary_new', LHD=LHD)

cursor.close()
conn.close()

