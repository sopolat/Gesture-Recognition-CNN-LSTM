# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:05:05 2020

@author: suley
"""

import pandas as pd
data=pd.read_csv("jester-v1-validation.csv",sep=';',names=["index","label"])
ps=["Zooming In With Two Fingers",
    "Zooming Out With Two Fingers",
    "Swiping Left",
    "Swiping Right",
    "Sliding Two Fingers Left",
    "Sliding Two Fingers Right",
    "Swiping Down"]
pdata=data[data["label"].isin(ps)]
ns=["Thumb Up",
    "Thumb Down",
    "Stop Sign",
    "No gesture",
    "Shaking Hand"]
ndata=data[data["label"].isin(ns)]
pdata=pdata.reset_index(drop=True)
ndata=ndata.reset_index(drop=True)
pdata.to_csv("positivesT.csv",index=False)
ndata.to_csv("negativesT.csv",index=False)

#from os import listdir
#from os.path import isfile, join
#images = [f for f in listdir("20bn-jester-v1/1") if isfile(join("20bn-jester-v1/1", f))]
#sizes=[]
#for i,row in data.iterrows():
#    images = [f for f in listdir("20bn-jester-v1/"+str(row["index"])) if isfile(join("20bn-jester-v1/"+str(row["index"]), f))]
#    sizes.append(len(images))