# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 08:58:32 2023

@author: lawashburn
"""

import plotly.graph_objects as go

import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

x = ['Product A', 'Product B', 'Product C']
y = [20, 14, 23]

fig = go.Figure(data=[go.Bar(
            x=x, y=y,
            text=y,
            textposition='auto',
        )])
fig.show()
