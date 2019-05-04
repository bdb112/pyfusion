# first step - 
# these imports should find syntax errors 
import pyfusion
from pyfusion.utils.utils import fix2pi_skips, modtwopi
from pyfusion.visual.sp import sp
from pyfusion.data.convenience import between, bw, btw, decimate, his, broaden
from pyfusion.utils import process_cmd_line_args
from pyfusion.data.DA_datamining import DA, report_mem
import pyfusion.clustering
import pyfusion.clustering.modes
from pyfusion.acquisition.LHD.read_igetfile import igetfile
from pyfusion.data.signal_processing import smooth, smooth_n, cross_correl, analytic_phase
import pyfusion.clustering as clust
from pyfusion.visual.window_manager import rmw, cmw
from pyfusion.utils import compact_str, modtwopi
from pyfusion.utils.dist_mp import dist_mp as dist

# these are imports from examples, and should be put elsewhere
#from pyfusion.examples.density_cluster import dists
