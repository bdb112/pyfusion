import pyfftw
import os

default_path = os.getenv('HOME')+'/.pyfftw'

if not(os.path.isdir(default_path)):
    os.mkdir(default_path)

default_filenames=[default_path+'/wisdom_'+ x for x in ['d','f','l']]

def save_wisdom(filenames=default_filenames):
    """ Save the wisdom accumulated since the last load_wisdom, which occurs when pyfusion is imported
    Best to run typical codes before saving.
    To see the current state of wisdom
    !fftw-wisdom - or look at the file.  I guess these are just timings for the routines tested
    e.g extensive tests of a 32 element complex forward out of place transform.
     !fftw-wisdom  -x cof32
    (fftw-3.3.3 fftw_wisdom #x458a31c8 #x92381c4c #x4f974889 #xcd46f97e
      (fftw_codelet_t2fv_4_avx 0 #x1040 #x1040 #x0 #x12dc3d8e #xe40293c9 #x508e7f21 #x18911bc9)
      (fftw_codelet_n2fv_8_avx 0 #x1040 #x1040 #x0 #xb916fb98 #xf394dae5 #xe9a593f6 #x4ce2ac3f)
    )
    for in place, just
    (fftw-3.3.3 fftw_wisdom #x458a31c8 #x92381c4c #x4f974889 #xcd46f97e
      (fftw_codelet_n2fv_32_sse2 0 #x1040 #x1040 #x0 #x8ee86d9a #x3bf651fe #x73d5cbe4 #xfd6f3826)
    )

    """
    wisdom = pyfftw.export_wisdom()
    for (fn,w) in zip(filenames,wisdom):
        f = open(fn,'w')
        f.write(w)
        f.close()

def load_wisdom(filenames=default_filenames):
    allwisdom = []
    for fn in filenames:
        f=open(fn,'r')
        allwisdom.append(f.read())
        f.close()
    
    pyfftw.import_wisdom(tuple(allwisdom))
