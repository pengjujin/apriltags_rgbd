#!/usr/category/env python
import numpy as np
from ss_plotting.make_plots import plot_bar_graph
import matplotlib.pyplot as plt
# Pretty version of this plot: http://matplotlib.org/examples/api/barchart_demo.html

categories = ['']
proposed_means = [3.31]
proposed_errs = [2.67]
alvar_means = [5.08]
alvar_errs = [2.39]

series = [proposed_means, alvar_means]
series_labels = ['Proposed', 'Alvar']
series_colors = ['red', 'blue']


ylabel = 'Rotation Error (degrees)'
title = 'Average Rotation Errors'

plot_bar_graph(series, series_colors,
               series_labels=series_labels,
               series_errs = [proposed_errs, alvar_errs],
               category_labels = categories,
               plot_ylabel = ylabel,
               plot_title = title,
               barwidth = 0.25,
               fontsize=13,
               legend_fontsize=13)
