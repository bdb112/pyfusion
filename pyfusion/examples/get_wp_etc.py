#_PYFUSION_TEST_@@Skip
import subprocess
import numpy as np

shots = \
        np.array([105059, 105066, 105085, 105096, 105097, 105098, 105109, 105112,
               105116, 105326, 105367, 105376, 105387, 105388, 105392, 105396,
               105397, 105401, 105402, 105403, 105406, 105552, 105739, 105828,
               105830, 105832, 105855, 106403, 106525, 106527, 106678, 107221,
               107236, 107238, 107239, 107256, 107327, 107942, 107943, 107944,
               107947, 107954, 108037, 108244, 109901])
for s in shots:
  cmd = str('wget http://egftp1.lhd.nifs.ac.jp/data/wp/{s1k}/{s}/000001/wp@{s}.dat.zip'
            .format(s=s, s1k=1000*(s/1000)))
  sub_pipe = subprocess.Popen(cmd,  shell=True, stdout=subprocess.PIPE,
							  stderr=subprocess.PIPE)
  (resp,err) = sub_pipe.communicate()
  if (err != '') or (sub_pipe.returncode != 0): 
    print(resp,err,'.') #
  print(resp[-10000:])
     
