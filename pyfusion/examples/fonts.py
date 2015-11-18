""" see http://matplotlib.org/examples/pylab_examples/fonts_demo.html
"""
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

plt.subplot(111, axisbg='w')

font0 = FontProperties()
alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
sizes = ['xx-small', 'x-small', 'small', 'medium', 'large','x-large', 'xx-large']
families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
styles = ['normal', 'italic', 'oblique']
weights = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']
variants = ['normal', 'small-caps']

yp = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

for k, style in enumerate(styles):
    font = font0.copy()
    font.set_family('sans-serif')
    font.set_style(style)
    t = plt.text(0.4, yp[k], style, fontproperties=font,
                 **alignment)
plt.show()
