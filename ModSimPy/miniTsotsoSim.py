# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy as scp
from scipy.signal import convolve2d 
from astropy.convolution import Gaussian2DKernel
import seaborn as sns
import pandas as pd

# %%
#Transformation Matrix

def RGB2LMS( imagePath ):
    """
    takes an imagePath with RGB stimulus values
    loads the image and
    returns the transformed LMC cone weights
    transformation matrices from: 
    https://ixora.io/projects/colorblindness/color-blindness-simulation-research/
    """
    M_RGB  = np.zeros((3,3), dtype='float')
    M_HPE   = np.zeros((3,3), dtype='float')
    #transform for linear RGB to XYZ space
    M_RGB = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    #tranfrom for XYZ to LMS color space
    M_HPE = np.array([[ 0.4002, 0.7076, -0.0808],
                 [-0.2263, 1.1653,  0.0457],
                 [      0,      0,  0.9182]])
    #transformation matrix for converting colors from RGB to the LMS color space.
    rgb2lms = M_HPE @ M_RGB #matrix multiplication of RGB by HPE
    lms2rgb = np.linalg.inv(rgb2lms)
    #load image
    imgIN = cv2.imread(imagePath,cv2.IMREAD_UNCHANGED)
    imgINrgb = cv2.cvtColor(imgIN, cv2.COLOR_BGR2RGB)
    #get image shape & preallocate for transform
    x,y,z = imgINrgb.shape
    imgLMS = np.zeros((x,y,z), dtype='float')
    #reshape the image such that it can be tranformed
    imgReshaped = imgINrgb.transpose(2, 0, 1).reshape(3,-1)
    imgLMS = rgb2lms @ imgReshaped #Convert to LMS
    imgLMS = imgLMS.reshape(z, x, y).transpose(1, 2, 0).astype(np.uint8)
    return imgLMS, imgINrgb

#LGN cone weights

LGN_weights = np.array([[ 1.1, -1.0, 0.0],
                       [-1.0, 1.1, 0.0],
                       [-0.9, -0.1, 1.1],
                       [-1.1, 1.0, 0.0],
                       [1.0, -1.1, 0.0],
                       [0.5, 0.1, -1.1]] )

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    ref: https://gist.github.com/andrewgiessel/4635563
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def LGN2V1( LGNchannels, V1RFsize ):
    """
    propagate the single opponent LGN responses, 
    but convolve with a larger Gaussian to account for 
    increase in RF size in area V1.then rectify the response.
    Takes in LGNchannels (np.array) & V1RFsize (int)
    Returns V1channels
    """
    #convolve with a larger RF Gaussian
    a2DG = makeGaussian( V1RFsize )
    mV1 = np.zeros( LGNchannels.shape )
    for anLGNchannel in np.arange( LGNchannels.shape[2] ):
        layer = LGNchannels[:,:,anLGNchannel]
        aChannel_mV1 = convolve2d(layer, a2DG, boundary='symm', mode='same')
        mV1[:,:,anLGNchannel] = aChannel_mV1
    #rectify the response  
    mV1[mV1 < 0] = 0
    return mV1


def V12V2( mV1, V2RFsize):
    """
    propagate the single opponent V1 responses
    with 2 hue selective mechanisms: 
    1: convolve with a larger Gaussian to account for 
    increase in RF size in area V2.then rectify the response.
    2:convolve with a V2 RF Gaussian and make multiplicative
    combinations of L|M V1 channels with S
    Takes in mV1 (np.array) & V2RFsize (int)
    Returns mV2 with 6 sopp + 8 mult channels
    """ 
    #preallocate for mV2 result
    mV2 = np.zeros( [mV1.shape[0], mV1.shape[0], 6+8 ] )
    #mV2 Gaussian
    a2DG = makeGaussian( V2RFsize )
    #single-opponent channels
    for aV2channel in np.arange( mV1.shape[2] ):
        layer = mV1[:,:,aV2channel]
        aChannel_mV2 = convolve2d(layer, a2DG, boundary='symm', mode='same')
        mV2[:,:,aV2channel] = aChannel_mV2
    #multiplicative channels
    mV2[:,:,6] = mV2[:,:,0] * mV2[:,:,2]
    mV2[:,:,7] = mV2[:,:,1] * mV2[:,:,2]
    mV2[:,:,8] = mV2[:,:,0] * mV2[:,:,5]
    mV2[:,:,9] = mV2[:,:,1] * mV2[:,:,5]
    mV2[:,:,10] = mV2[:,:,3] * mV2[:,:,2]
    mV2[:,:,11] = mV2[:,:,4] * mV2[:,:,2]
    mV2[:,:,12] = mV2[:,:,3] * mV2[:,:,5]
    mV2[:,:,13] = mV2[:,:,4] * mV2[:,:,5]
    #rectify the response  
    mV2[mV2 < 0] = 0
    return mV2

#V4 weights
V4_weights = np.array([[ 0.25151,0,0,0,0.25307,0.084352,0.021472,0.07919,0,0,0,0.032343,0,0.27806],
                        [0.17009,0.011735,0,0,0.049806,0.31483,0,0.28675,0,0.085706,0,0,0,0.081083],
                        [0.00062658,0.10285,0,0.045216,0,0.072422,0,0.02433,0,0.37631,0,0,0.37824,0],
                        [0,0.3822,0.006119,0.36753,0,0,0,0,0.22912,0,0.015024,0,0,0],
                        [0,0.0044941,0.46447,0.0099449,0,0,0,0,0.072729,0,0.44836,0,0,0],
                        [0.049644,0.0014075,0.083582,0,0.052261,0,0.40391,0,0.0052898,0,0,0.40391,0,0 ]])
V4_weights_rows = ['red','cyan','green','yellow','blue','magenta']
V4_weight_cols = ['V2_L_On','V2_M_On','V2_S_On','V2_L_Off','V2_M_Off','V2_S_Off', \
                   'V2_mul_LOn_SOn', 'V2_mul_LOn_SOff', 'V2_mul_MOn_SOn', 'V2_mul_MOn_SOff', \
                  'V2_mul_LOff_SOn', 'V2_mul_MOff_SOn', 'V2_mul_LOff_SOff', 'V2_mul_MOff_SOff' ]

mV4_weights = pd.DataFrame(V4_weights, columns=V4_weight_cols, index=V4_weights_rows)
mV4_weights

def V22V4( mV2, V4RFsize, V4weights ):
    """
    take the 14 V2 channels as input and return 6 local hue channels:
    weight each V2 input channel with weighting values
    convolve with a larger Gaussian to account for 
    increase in RF size in area V4.then rectify the response.
    Takes in mV2 (np.array) & V4RFsize (int) & V4Weights (pandas df)
    Returns mV4 with 6 local Hue channels
    """ 
    #preallocate for mV2 result
    mV4 = np.zeros( [mV2.shape[0], mV2.shape[0], 6 ] )
    #print( 'mV4.shape:', mV4.shape )
    #mV2 Gaussian
    a2DG = makeGaussian( V4RFsize )
    #print( '2DG shape:', a2DG.shape )
    for aHue in np.arange( V4_weights.shape[0] ):
        #print( 'aHue:', aHue)
        #hue Weights
        hueWeights = V4_weights[ aHue,: ]
        #print( 'hue weights:', hueWeights)
        #R_LMS cone activities in RF
        V4rf = np.ones((14, V4RFsize, V4RFsize))
        hueVals = np.array( hueWeights )
        V4rf = (V4rf.swapaxes(0,2) * hueVals).swapaxes(2,0)
        #2DGaussian
        mask = V4rf * a2DG
        #convolve the Gaussian with the stimulus
        mV4_channel = np.zeros( mV2.shape )
        for aChannel in np.arange( V4_weights.shape[1] ):
            #print( 'aChannel:', aChannel )
            layer = mV2[:,:,aChannel]
            channelmask = mask[aChannel,:,:]
            aChannel_mV4 = convolve2d(layer, channelmask, boundary='symm', mode='same')
            mV4_channel[:,:,aChannel] = aChannel_mV4
        total = np.sum(mV4_channel, axis=2)
        mV4[:,:,aHue] = total
    #response rectification  
    mV4[mV4 < 0] = 0
    return mV4

#Determine probability distributions for LMS cones from random distributions that approximate observed values (25)
def getMaxVals( mLayer ):
    """
    find the max values for each channel of a model layer
    take a model layer (np.array)
    return the max values along the 3rd array dimention
    """
    numChannels = mLayer.shape[2]
    result = np.zeros([numChannels,2])
    for aChannel in np.arange(numChannels):
        idx = [0,0]
        idx = np.where(mLayer[:,:,aChannel] == np.amax(mLayer[:,:,aChannel]))
        #if idx[0].shape[0] > 1:
        #    idx = [0,0]
        result[aChannel,:] = idx
    return result

#update mLGN to take the New LMS population ratios
def LMS2LGN( LMSimage, LGN_weights, RF_size ):
    """
    takes in the LMS weighted image transform
    returns 6 channel array with DoG LGN representations of the stimulus
    """
    #(each channel weight from LGN_weights * cone activations from LMSimage convolve with a Gaussian)
    mLGN = np.zeros( [LMSimage.shape[0], LMSimage.shape[1],len( LGN_weights )] )
    for aNeuron in np.arange( len( LGN_weights ) ):
        #Cone Weights
        coneWeights = LGN_weights[ aNeuron ]
        #add some noise to the cone contributions
        adjustedConeW = coneWeights    * ( np.random.normal(0, 0.1, 3) + 1 )
        #R_LMS cone activities in RF
        LGNrf = np.ones((3, RF_size, RF_size))
        LMSVals = np.array( adjustedConeW  )
        LGNrf = (LGNrf.swapaxes(0,2) * LMSVals).swapaxes(2,0)
        #2DGaussian
        a2DG = makeGaussian( RF_size )
        mask = LGNrf * a2DG
        #convolve the Gaussian with the stimulus
        mLGN_channel = np.zeros( LMSimage.shape )
        for aLayer in np.arange( 3 ):
            layer = LMSimage[:,:,aLayer]
            channelmask = mask[aLayer,:,:]
            aChannel_mLGN = convolve2d(layer, channelmask, boundary='symm', mode='same')
            mLGN_channel[:,:,aLayer] = aChannel_mLGN
        total = np.sum(mLGN_channel, axis=2)
        mLGN[:,:,aNeuron] = total
    #response rectification  
    mLGN[mLGN < 0] = 0
    #mLGN[mLGN] Think about how to set saturation
    return mLGN
# %%

#RBG Color Wheel Stimulus
imgpath = "/home/bonzilla/Desktop/MSDS2020/DATA604_SimPy/ModSimPy/colorwheel.png"
LMSim, RGBim = RGB2LMS( imgpath )

fig1 = plt.figure()
a1 = plt.gca()
a1.axes.get_xaxis().set_visible(False)
a1.axes.get_yaxis().set_visible(False)
plt.title('Original RGB Image')
plt.imshow(RGBim)
plt.show()

#%%
imgpath = "/home/bonzilla/Desktop/MSDS2020/DATA604_SimPy/ModSimPy/colorwheel.png"
#Run Simulation
numTrials = 50
LGN_rfsize = 19
LGN_maxvals  = pd.DataFrame()
V4_maxvals  = pd.DataFrame()

for aTrial in np.arange( numTrials ):
    trialLabel = 'Trial'+str(aTrial)
    print( trialLabel )
    #generate LMS responses
    mLMS, RGBim = RGB2LMS( imgpath )
    #generate LMS population weight
    mLGN = LMS2LGN( mLMS, LGN_weights, LGN_rfsize )
    try:
        LGN_maxval = getMaxVals( mLGN )
    except:
        print('Exception!')
        pass
    LGN_maxvals[trialLabel] = np.vsplit( LGN_maxval, 6 )
    print( LGN_maxval )
    
    #mV1
    mV1 = LGN2V1( mLGN, LGN_rfsize*2 )
    
    #mV2
    mV2 = V12V2( mV1, LGN_rfsize*3 )
    
    #mV4
    mV4 = V22V4( mV2, LGN_rfsize*3, mV4_weights )
    V4_maxval = getMaxVals( mV4 )
    print( V4_maxval )
    V4_maxvals[trialLabel] = np.vsplit( V4_maxval, 6 )   
    
    
    # %%
colNames = V4_maxvals.columns 
L = LGN_maxvals.copy()
V = V4_maxvals.copy()   
LT = L.transpose()
V4T = V.transpose()
plotDat = pd.DataFrame()
plotDat2 = pd.DataFrame()
ChanNames = [['0x','0y'],['1x','1y'],['2x','2y'], \
                 ['3x','3y'],['4x','4y'],['5x','5y']]

for aChan in np.arange( 6 ):
    holdDat = []
    holdDat2 = []
    for aTrial in np.arange( numTrials ):
        coord = list( LT[aChan][aTrial].flatten() )
        coord2 = list( V4T[aChan][aTrial].flatten() )
        holdDat.append(coord)
        holdDat2.append(coord2)
    Dat = pd.DataFrame( holdDat, columns = ChanNames[aChan])
    Dat2 = pd.DataFrame( holdDat2, columns = ChanNames[aChan])
    #print( Dat )
    plotDat = pd.concat( [plotDat, Dat], axis = 1 )
    plotDat2 = pd.concat( [plotDat2, Dat2], axis = 1 )
    
# %%    


fig2 = plt.figure()
a1 = plt.gca()
a1.axes.get_xaxis().set_visible(False)
a1.axes.get_yaxis().set_visible(False)
plt.title('Original RGB Image')
plt.imshow(RGBim)
plt.plot(plotDat['0y'], plotDat['0x'], 'w*')
plt.plot(plotDat['1y'], plotDat['1x'], 'wo')
plt.plot(plotDat['2y'], plotDat['2x'], 'ws')
plt.plot(plotDat['3y'], plotDat['3x'], 'k*')
plt.plot(plotDat['4y'], plotDat['4x'], 'ko')
plt.plot(plotDat['5y'], plotDat['5x'], 'ks')
plt.show()

# %%
    
    
fig3 = plt.figure()
a1 = plt.gca()
a1.axes.get_xaxis().set_visible(False)
a1.axes.get_yaxis().set_visible(False)
plt.title('Original RGB Image')
plt.imshow(RGBim)
plt.plot(plotDat2['0y'], plotDat2['0x'], 'ro')
plt.plot(plotDat2['1y'], plotDat2['1x'], 'co')
plt.plot(plotDat2['2y'], plotDat2['2x'], 'go')
plt.plot(plotDat2['3y'], plotDat2['3x'], 'yo')
plt.plot(plotDat2['4y'], plotDat2['4x'], 'bo')
plt.plot(plotDat2['5y'], plotDat2['5x'], 'mo')
plt.show()  
    
# %%
    
df1 = LGN_maxvals.copy() 
df1.to_csv( "/home/bonzilla/Desktop/MSDS2020/DATA604_SimPy/ModSimPy/LGN_maxvals2.csv", index = False )   
df2 = V4_maxvals.copy()
df2.to_csv( "/home/bonzilla/Desktop/MSDS2020/DATA604_SimPy/ModSimPy/V4_maxvals2.csv", index = False )   
aV4 = mV4.copy()
#aV4.to_csv( "/home/bonzilla/Desktop/MSDS2020/DATA604_SimPy/ModSimPy/aV4.csv", index = False ) 
    
    
    