Flucstrucs
==========

This reflects the latest version which optionally splits the spectrum
up into bands based on the strongest peaks.

Definitions
-----------

 Shot: shot

 time: The time of the mid point of the fluctuation found

 SVS:  The sigular values present in the FS, in the order of the SVS

 freq: the dominant frequency of the fs

 Amp:  estimated amplitude (normalisation distorts this)

 a12:  the ratio a2/a1, the th relative strength of the second largest SV in the FS

 p  : 

 H:    Entropy of the FS

 frlow, frhigh: The range of the bandpass filter used to extract this
 if any. If no filter is used, frhigh= nyquist.  If 1 band is used,

 frhigh=0.98*nyquist so that the filter mask avoids aliasing.

 cpkf: channel peaking factor  - how much the peak channel exceeds the others

 fpkf: frequency peaking factor - sim but freq

 phases - actually nearest neighbour phases differences.
