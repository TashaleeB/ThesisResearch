#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:54:44 2018

@author: Saul Kohn
"""
#printf "\e[?2004l"

import numpy as np, matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
from matplotlib import gridspec

#%%

d = np.load('2457548.45923.npz')

NOT_REAL_ANTS="0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 90, 91, 92, 93, 94, 95, 98, 99, 100, 101, 102, 103, 106, 107, 108, 109, 110, 111"#HERA-19

ex_ants = [int(a) for a in NOT_REAL_ANTS.split(', ')] # Turns str to list
keep_ants = [a for a in range(d['gains'].shape[2]) if a not in ex_ants] # Equivalen to having an if no statement in for loop. You are looping over a list.
good_ants = [a for a in keep_ants if a not in [81,22,43,80]]

# manually flagged RFI channels based on Saul's memos
low = list(range(0,95))
orbcomm = list(range(375,390))
others = list(range(695,705))+list(range(759,761))+list(range(768,771))+list(range(830,832))+list(range(850,852))+[912,932,933,934,935]+[960,961]+list(range(990,1023))
others+=[510,511,512,850,851,852,853,854]
msk=low+orbcomm+others

msk_spec = np.zeros((1024))

good_chans = [c for c in range(1024) if c not in msk]
freqs = np.linspace(100,200,num=1024)

for c in range(1024):
    if c in good_chans:
        msk_spec[c] = 1

f,ax = plt.subplots(figsize=(15,8))
f.suptitle("gc.2457549.uvcRU: D-Term Solutions", fontsize=14)#Title centered above all subplots
ax.fill_between(freqs[201:300],0,3, facecolor='k',alpha=0.2)
ax.fill_between(freqs[581:680],0,3,facecolor='k',alpha=0.2)

weird_ants = []

for i,a in enumerate(good_ants):
    if a==80:
        continue
    if i <= 8:
        ls='-'
    else:
        ls='--'
    msk_gain = np.ma.masked_where(np.abs(d['gains'][0,:,a])*msk_spec==0.,
                                  np.abs(d['gains'][0,:,a])*msk_spec) #forcing python to not plot RFI
    index = np.where(msk_gain >0.5)[0]
    if np.sum(index) !=0:
        weird_ants.append(a)

    ax.plot(freqs,msk_gain,ls,label=str(a),lw=2)
    #ax.plot(freqs[0:len(freqs[0:np.where(freqs<=189)[0].max()])],msk_gain[0:len(freqs[0:np.where(freqs<=189)[0].max()])],ls,label=str(a),lw=2)

ax.set_ylim(0,2.75)
ax.set_xlim(100,200)
plt.legend()
plt.xlabel('Frequency [MHz]',size=12)
plt.ylabel('Amplitude [arb.]',size=12)
plt.grid()
f.suptitle("GAIN Only Simulation zen.2457755.87936.HH.uvU:Y B-Term Solutions", fontsize=14)#Title centered above all subplots
#f.savefig('2458115.24482.HH.uvOCR_BP.png')
#print(weird_ants)
plt.show()
#------------------------------------------------------
#%% 15 Plots for 15 good antenna
#------------------------------------------------------

npzlist = glob.glob('*.R.npz')
d = np.load(npzlist[0])

#NOT_REAL_ANTS="0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 90, 91, 92, 93, 94, 95, 98, 99, 100, 101, 102, 103, 106, 107, 108, 109, 110, 111"#HERA-19
#badant = [81,22,43,80] #HERA-19
NOT_REAL_ANTS = "3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22, 28, 29, 30, 31, 32, 33, 34, 35, 42, 43, 44, 45, 46, 47, 48, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 89, 90, 91, 92, 93, 94, 95, 96, 97, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135"#H1C-IDR2
badant=[0,14,50,122,136]#H1C-IDR2

ex_ants = [int(a) for a in NOT_REAL_ANTS.split(', ')] # Turns str to list
keep_ants = [a for a in range(d['gains'].shape[2]) if a not in ex_ants] # Equivalen to having an if no statement in for loop. You are looping over a list.
good_ants = [a for a in keep_ants if a not in badant]

# manually flagged RFI channels based on Saul's memos
low = list(range(0,95))
orbcomm = list(range(375,390))
others = list(range(695,705))+list(range(759,761))+list(range(768,771))+list(range(830,832))+list(range(850,852))+[912,932,933,934,935]+[960,961]+list(range(990,1023))
others+=[510,511,512,850,851,852,853,854]
msk=low+orbcomm+others

msk_spec = np.zeros((1024))

good_chans = [c for c in range(1024) if c not in msk]
freqs = np.linspace(100,200,num=1024)

for c in range(1024):
    if c in good_chans:
        msk_spec[c] = 1

nrow,ncol = 3,5
numplots = nrow*ncol
ndays = len(npzlist)
iarr = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
jarr = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]

f,axarr = plt.subplots(nrow,ncol,sharex=True,sharey=True,figsize=(15,8))
f.suptitle("gc.2457549.uvcRU: D-Term Solutions", fontsize=14)#Title centered above all subplots
gs = gridspec.GridSpec(nrow,ncol)#, wspace=0.01, hspace=0.1)

for i,a in enumerate(good_ants):
    
    #for ind in range(numplots):
    ax=plt.subplot(gs[iarr[i],jarr[i]])
    ax.set_ylim(0,2.75)
    ax.set_xlim(100,200)
    ax.annotate(str(a), xy=(180,2.55), size=9, color='deeppink')
    ax.fill_between(freqs[201:300],-0.2,3.2, facecolor='k',alpha=0.2)
    ax.fill_between(freqs[581:680],-0.2,3.2,facecolor='k',alpha=0.2)

    for daynum in range(ndays):
        data = np.load(npzlist[daynum])       
        msk_xgain = np.ma.masked_where(np.abs(data['gains'][0,:,a])*msk_spec==0.,
                                      np.abs(data['gains'][0,:,a])*msk_spec)
        msk_ygain = np.ma.masked_where(np.abs(data['gains'][1,:,a])*msk_spec==0.,
                                      np.abs(data['gains'][1,:,a])*msk_spec)
#print(list(msk_xgain))
        depenvar =msk_xgain
        if i > 9:
            ax.set_xlabel('Frequency [MHz]',size=12)
            ax.grid(color='k', linestyle='--', linewidth=1)
        if i == 0:
            ax.set_ylabel('Amplitude [arb.]',size=12)
            ax.plot(freqs,depenvar,'-',label= 'Day '+str(daynum+1),lw=2)
        if i == 5:
            ax.set_ylabel('Amplitude [arb.]',size=12)
            ax.plot(freqs,depenvar,'-',label= 'Day '+str(daynum+1),lw=2)
        #ax[0,0].set_title('Name')#gives plot at location (1,1) a title
        if i == 10:
            ax.set_ylabel('Amplitude [arb.]',size=12)
            ax.plot(freqs,depenvar,'-',label= 'Day '+str(daynum+1),lw=2)
        
        else:
            ax.plot(freqs,depenvar,'-',label= 'Day '+str(daynum+1),lw=2)
            ax.grid(color='k', linestyle='--', linewidth=1)

ax.legend(bbox_to_anchor=(1.6, 1), loc='lower right', borderaxespad=0.)

#f.savefig('2458115.24482.HH.uvOCR_BP.png')
plt.show()











#-------------------------------------------------------------------------------------------------------
#%% Quick Plot of the Cal Solutions per Antenna over all frequencies Having issues here I think
#-------------------------------------------------------------------------------------------------------

import numpy as np, matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
from matplotlib import gridspec
d = np.load("zen.2458115.24482.HH.uvR.MS.Dcal.npz")
NOT_REAL_ANTS = "3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22, 28, 29, 30, 31, 32, 33, 34, 35, 42, 43, 44, 45, 46, 47, 48, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 89, 90, 91, 92, 93, 94, 95, 96, 97, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135"#H1C-IDR2
badant=[0,14,50,122,136]#H1C-IDR2

ex_ants = [int(a) for a in NOT_REAL_ANTS.split(', ')] # Turns str to list
keep_ants = [a for a in range(d['gains'].shape[2]) if a not in ex_ants] # Equivalen to having an if no statement in for loop. You are looping over a list.
    
good_ants = [a for a in keep_ants if a not in badant]

# manually flagged RFI channels based on Saul's memos
low = list(range(0,95))
orbcomm = list(range(375,390))
others = list(range(695,705))+list(range(759,761))+list(range(768,771))+list(range(830,832)+list(range(850,852))+[912,932,933,934,935]+[960,961]+list(range(990,1023))
others+=[510,511,512,850,851,852,853,854]
msk=low+orbcomm+others
msk_spec = np.zeros((1024))

good_chans = [c for c in range(1024) if c not in msk]
freqs = np.linspace(100,200,num=1024)

weird_ants=[]

for c in range(1024):
    if c in good_chans:
        msk_spec[c] = 1

f,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,8))
f.suptitle("gc.2457549.uvcRU: D-Term Solutions", fontsize=14)#Title centered above all subplots
ax.fill_between(freqs[201:300],0,3, facecolor='k',alpha=0.2)
ax.fill_between(freqs[581:680],0,3,facecolor='k',alpha=0.2)

for i,a in enumerate(good_ants[0:10]):
    if a==80 :
        continue
    if i<=8:
        ls='-'
    else:
        ls='--'
"""#Use this section is you need to cut off parts of the data
    msk_gain = np.ma.masked_where(np.abs(d['gains'][1,0:np.where(freqs<=189)[0].max(),a])*msk_spec[0:np.where(freqs<=189)[0].max()]==0.,
    np.abs(d['gains'][1,0:np.where(freqs<=189)[0].max(),a])*msk_spec[0:np.where(freqs<=189)[0].max()])
    index = np.where(msk_gain >0.5)[0]
    if np.sum(index) !=0:
         weird_ants.append(a)
    ax.plot(freqs[0:np.where(freqs<=189)[0].max()],msk_gain,ls,label=str(a),lw=2)
"""
    msk_xgain = np.ma.masked_where(np.abs(d['gains'][0,:,a])*msk_spec==0.,np.abs(d['gains'][0,:,a])*msk_spec)
    msk_ygain = np.ma.masked_where(np.abs(d['gains'][1,:,a])*msk_spec==0.,np.abs(d['gains'][1,:,a])*msk_spec)
    #print(list(msk_gain))
    index = np.where(msk_xgain >0.5)[0]
    if np.sum(index) !=0:
        weird_ants.append(a)
    ax[0].set_title('X Bandpass Solutions')
    ax[0].plot(freqs,msk_xgain,ls,label=str(a),lw=2)
                                                                             
    ax[1].set_title('Y Bandpass Solutions')
    ax[1].plot(freqs,msk_ygain,ls,label=str(a),lw=2)
#ax.set_ylim(0,4)
ax[0].set_xlim(100,200)
ax[1].set_xlim(100,200)
ax[0].legend()
ax[1].legend()
#plt.legend()
plt.xlabel("Frequency [MHz]",size=12)
plt.ylabel("Amplitude [arb.]",size=12)
plt.grid()
f.savefig("Bterms_CalSol.png")
plt.show()

#[13, 99, 100, 101, 102, 103, 104, 117, 118, 119, 121]
