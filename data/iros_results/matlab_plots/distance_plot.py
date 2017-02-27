import numpy as np
from ss_plotting.make_plots import plot
import matplotlib.pyplot as plt


data_translation = [0.221, 0.306, 0.323, 0.337, 0.474, 0.205, 0.388, 0.499, 0.261, 0.625, 0.393, 0.508, 0.456]
test_translation =  [0, 0, 0, 0, 0, 0.001, 0, 0.032, 0.024, 0.084, 0.081, 0.077, 0.075]

x_trans = np.arange(0.65, 1.85, 0.1)

series = [(x_trans, data_translation), (x_trans, test_translation)]
series_labels = ['RGB', 'RGBD']
series_colors = ['red', 'blue']

ylabel = 'Error Percentage'
xlabel = 'Distance (cm)'
title = 'Rotation Error vs Distance with sigma = 0.5 pixels'

fig, ax = plot(series, 
     series_labels=series_labels,
     series_colors=series_colors,
     linewidth=5,
     plot_ylabel = ylabel,
     plot_xlabel = xlabel,
	 plot_title = title,
	 fontsize=13,
	 legend_fontsize=13)