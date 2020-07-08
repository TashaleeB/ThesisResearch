#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Wed Jan  15 16:38:00 2020
    
    @author: tashalee
"""

# first, copy your data MS to a new file, perhaps "model.MS"
# open the model dataset that we're going to stuff the visibilities into
ms_file = "model.MS"
tb.open(ms_file, nomodify=False)
model_data = tb.getcol("DATA")
tb.close()

# put the data and weights into the MS
tb.open("file_to_calibrate.MS")
tb.putcol("MODEL_DATA", model_data)

tb.flush()
tb.close()
