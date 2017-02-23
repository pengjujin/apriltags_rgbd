#!/usr/category/env python
import numpy as np
from ss_plotting.make_plots import plot_bar_graph
import matplotlib.pyplot as plt
# Pretty version of this plot: http://matplotlib.org/examples/api/barchart_demo.html

categories = ['2 lux', '43 lux', '90 lux', '243 lux']
RGB_means = [0.24, 0.13, 0.03, 0.02]
RGBD_means = [0.03, 0.01, 0.0, 0.0]


series = [RGB_means, RGBD_means]
series_labels = ['RGB', 'RGBD']
series_colors = ['red', 'blue']


ylabel = 'Error Percentage'
title = 'Rotation Error w.r.t. to Lighting'

plot_bar_graph(series, series_colors,
               series_labels=series_labels,
               category_labels = categories,
               plot_ylabel = ylabel,
               plot_title = title,
               category_padding = 0.45,
               fontsize=13,
               legend_fontsize=13)

