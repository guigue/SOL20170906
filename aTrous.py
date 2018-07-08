# External methods
import sys, string, os, struct, glob
import numpy as np
import datetime
from scipy.optimize import curve_fit
import xml.etree.ElementTree as xmlet
from astropy.io import fits

#############################################################################################
#
#  aTrous
#
#  A series of python routines to get the  'a trous' wavelet transform of 1 & 2D data.
#  They are an implementation of the Starck, J.-L. & Murtagh, F. description in 
#  "Astronomical Image and Data Analysis" (2006), Srpinger Verlag Berlin Heidelberg, 
#  2nd edition, Appendix Material.  
#
#  Procedures included
#  triangle2D: 2D matrix input, triangle mother wavelet.
#  B3_spline2D: 2D matrix inout, B3 spline mother wavelet.
#  triangle: 1D sequence input, triangle mother wavelet.
#  B3_spline: 1D sequence input, triangle mother wavelet.
#
##############################################################################################
#
#    Author:  @guiguesp
#             guigue@craam.mackenzie.br  , guigue@gcastro.net
#             http://www.guigue.gcastro.net
#
#
#    Change Record :  
#             First written during very warm February 2018 days in São Paulo...
#             Comments added in 2018.07.08 
#
#    If you find it useful, please make a mention in your published work.
#
###############################################################################################



def triangle2D(m,verbose=False):

#
#   triangle2D
#
#   Usage:
#         cjk,djk = aTrous.triangle2D(m)
#         where m : a 2D matrix of X x Y dimensions.  The maximum wavelet scale is
#                   determined by the min(X,Y)
#         cjk,djk = a cube with the outputs of the wavelet transform. 
#                   cjk is the blurred representation of the input matrix
#                   djk is the wavelet itself. 
#
###################################################################################
   
    nRow  = m.shape[0]
    nCol  = m.shape[1]
    m_len = np.min([ nRow, nCol ])
    NScal = int(np.log(m_len)/np.log(2))
    if verbose:
        print ' '
        print 'Number of Scales = {0:=03d}'.format(NScal)
    cjk = np.zeros((nRow,nCol,NScal),dtype=float)
    djk = np.zeros((nRow,nCol,NScal-1),dtype=float)
    cjk[:,:,0] = m

    kernel = np.array( [ [1.0/16,1.0/8,1.0/16], [1.0/8,1.0/4,1.0/8], [1.0/16,1.0/8,1.0/16]] )
    
    # Let's know the efficiency
    t_ref = datetime.datetime.now()
    kShift = 1
    for dyada in np.arange(1,NScal):
        
        for r in np.arange(0,m.shape[0]):
            for k in np.arange(0,m.shape[1]): 
            
                kLow   = k - kShift
                if kLow < 0:
                    kLow = nCol + kLow
                    
                kHigh  = k + kShift
                if kHigh > nCol-1:
                    kHigh = kHigh - nCol

                rLow   = r - kShift 
                if rLow < 0:
                    rLow = nRow + rLow
               
                rHigh  = r + kShift
                if rHigh > nRow-1:
                    rHigh = rHigh - nRow

                subm   = np.array([ [cjk[rLow,kLow,dyada-1] , cjk[rLow,k,dyada-1]  , cjk[rLow,kHigh,dyada-1] ] ,
                                    [cjk[r,kLow,dyada-1]    , cjk[r,k,dyada-1]     , cjk[r,kHigh,dyada-1]    ] ,
                                    [cjk[rHigh,kLow,dyada-1], cjk[rHigh,k,dyada-1] , cjk[rHigh,kHigh,dyada-1]] ])
                
                cjk[r,k,dyada] = (kernel*subm).sum()
                djk[r,k,dyada-1] = cjk[r,k,dyada-1] - cjk[r,k,dyada]
            
        kShift = 2 * kShift
        if verbose:
            print 'Iteration {0:3d} out of {1:3d}'.format(dyada,NScal-1)
            
    t_fin = datetime.datetime.now()

    if verbose:
        print 'Total Time elapsed ', t_fin-t_ref
                
    return cjk,djk

def B3_spline2D(m):
#
#   B3_spline2D
#
#   Usage:
#         cjk,djk = aTrous.B3_spline2D(m)
#         where m : a 2D matrix of X x Y dimensions.  The maximum wavelet scale is
#                   determined by the min(X,Y)
#         cjk,djk = a cube with the outputs of the wavelet transform. 
#                   cjk is the blurred representation of the input matrix
#                   djk is the wavelet itself. 
#
###################################################################################

    nRow  = m.shape[0]
    nCol  = m.shape[1]
    m_len = np.min([ nRow, nCol ])
    NScal = int(np.log(m_len)/np.log(2))
    print ' '
    print 'Number of Scales = {0:=03d}'.format(NScal)
    cjk = np.zeros((nRow,nCol,NScal),dtype=float)
    djk = np.zeros((nRow,nCol,NScal-1),dtype=float)
    cjk[:,:,0] = m

    kernel = np.matmul( np.array([[1./16],[1./4],[3./8],[1./4],[1./16.]]), np.array([[1./16,1./4,3./8,1./4,1./16]]) )
    
    # Let's know the efficiency
    t_ref = datetime.datetime.now()
    kShift = 1
    for dyada in np.arange(1,NScal):
        
        for r in np.arange(0,m.shape[0]):
            for k in np.arange(0,m.shape[1]): 

                kLow   = k - 2 * kShift
                if (kLow < 0):
                    kLow  = nCol + kLow # ciclic matrix
                
                kLowMid   = k - kShift
                if (kLowMid < 0):
                    kLow  = nCol + kLowMid # ciclic matrix
                
                kHigh  = k + 2 * kShift
                if (kHigh > nCol-1):
                    kHigh = kHigh - nCol  # ciclic matrix
                
                kHighMid  = k + kShift
                if (kHighMid > nCol-1):
                    kHighMid = kHighMid - nCol  # ciclic matrix

                rLow   = r - 2 * kShift 
                if rLow < 0:
                    rLow = nRow + rLow
                    
                rLowMid   = r - kShift 
                if rLowMid < 0:
                    rLowMid = nRow + rLowMid
               
                rHigh  = r + 2 * kShift
                if rHigh > nRow-1:
                    rHigh = rHigh - nRow

                rHighMid  = r + kShift
                if rHighMid > nRow-1:
                    rHighMid = rHighMid - nRow

                subm   = np.array([ [ cjk[rLow,kLow,dyada-1]    , cjk[rLow,kLowMid,dyada-1]    , cjk[rLow,k,dyada-1]    , cjk[rLow,kHighMid,dyada-1]    , cjk[rLow,kHigh,dyada-1]    ], 
                                    [ cjk[rLowMid,kLow,dyada-1] , cjk[rLowMid,kLowMid,dyada-1] , cjk[rLowMid,k,dyada-1] , cjk[rLowMid,kHighMid,dyada-1] , cjk[rLowMid,kHigh,dyada-1] ],
                                    [ cjk[r,kLow,dyada-1]       , cjk[r,kLowMid,dyada-1]       , cjk[r,k,dyada-1]       , cjk[r,kHighMid,dyada-1]       , cjk[r,kHigh,dyada-1]       ],
                                    [ cjk[rHighMid,kLow,dyada-1], cjk[rHighMid,kLowMid,dyada-1], cjk[rHighMid,k,dyada-1], cjk[rHighMid,kHighMid,dyada-1], cjk[rHighMid,kHigh,dyada-1]],
                                    [ cjk[rHigh,kLow,dyada-1]   , cjk[rHigh,kLowMid,dyada-1]   , cjk[rHigh,k,dyada-1]   , cjk[rHigh,kHighMid,dyada-1]   , cjk[rHigh,kHigh,dyada-1]   ] ])
                
                cjk[r,k,dyada] = (kernel*subm).sum()
                djk[r,k,dyada-1] = cjk[r,k,dyada-1] - cjk[r,k,dyada]
            
        kShift = 2 * kShift
        print 'Iteration {0:3d} out of {1:3d}'.format(dyada,NScal-1)
            
    t_fin = datetime.datetime.now()

    print 'Total Time elapsed ', t_fin-t_ref
                
    return cjk,djk

    
def triangle(sig,tr):
#
#   triangle
#
#   Usage:
#         cjk,djk = aTrous.triangle(v)
#         where v : a vector or sequence of numbers.
#         cjk,djk = a matrix (2D) with the outputs of the wavelet transform. 
#                   cjk is the blurred representation of the input sequence.
#                   djk is the wavelet itself. 
#
###################################################################################

    sig_len = len(sig)
    NScal = int(np.log(sig_len)/np.log(2))
    print ' '
    print 'Number of Scales = {0:=03d}'.format(NScal)

    # Initialize the vector results
    cjk = np.zeros((sig_len,NScal),dtype=float)
    djk = np.zeros((sig_len,NScal-1),dtype=float)
    cjk[:,0] = sig
    wps = np.zeros((2,NScal-1),dtype=float)

    # Let's know the efficiency
    t_ref = datetime.datetime.now()

    kShift = 1
    for dyada in np.arange(1,NScal):
        
        for k in np.arange(0,sig_len):
            
            kLow   = k - kShift
            if (kLow < 0):
                kLow  = sig_len + kLow # ciclic matrix
                
            kHigh  = k + kShift
            if (kHigh >= sig_len):
                kHigh = kHigh - sig_len  # ciclic matrix
                
            cjk[k,dyada] = 0.5 * cjk[k,dyada-1] + 0.25 * (cjk[kLow,dyada-1] + cjk[kHigh,dyada-1])
            djk[k,dyada-1] = cjk[k,dyada-1] - cjk[k,dyada]
            
        kShift = 2 * kShift 

    ts = tr 
    for dyada in np.arange(0,NScal-1):
        wps[1,dyada] = (djk[:,dyada]**2).sum()/sig_len
        tr = 2 * tr
        wps[0,dyada] = tr

    t_fin = datetime.datetime.now()

    print 'Total Time elapsed ', t_fin-t_ref
                
    return cjk,djk,wps

def B3_spline(sig,tr):
#
#   B3_spline
#
#   Usage:
#         cjk,djk = aTrous.B3_spline(v)
#         where v : a vector or sequence of numbers.
#         cjk,djk = a matrix (2D) with the outputs of the wavelet transform. 
#                   cjk is the blurred representation of the input sequence.
#                   djk is the wavelet itself. 
#
###################################################################################

    sig_len = len(sig)
    NScal = int(np.log(sig_len)/np.log(2))
    print ' '
    print 'Number of Scales = {0:=03d}'.format(NScal)

    # Initialize the vector results
    cjk = np.zeros((sig_len,NScal),dtype=float)
    djk = np.zeros((sig_len,NScal-1),dtype=float)
    cjk[:,0] = sig
    wps = np.zeros((2,NScal-1),dtype=float)

    # Let's know the efficiency
    t_ref = datetime.datetime.now()

    conv_mask = np.array([1./16., 1./4., 3./8., 1./4., 1./16.],dtype=float)
    
    kShift = 1
    for dyada in np.arange(1,NScal):
        
        for k in np.arange(0,sig_len):
            
            kLow   = k - 2 * kShift
            if (kLow < 0):
                kLow  = sig_len + kLow # ciclic matrix
                
            kLowMid   = k - kShift
            if (kLowMid < 0):
                kLow  = sig_len + kLowMid # ciclic matrix
                
            kHigh  = k + 2 * kShift
            if (kHigh >= sig_len):
                kHigh = kHigh - sig_len  # ciclic matrix
                
            kHighMid  = k + kShift
            if (kHighMid >= sig_len):
                kHighMid = kHighMid - sig_len  # ciclic matrix
                
            sub_matrix = np.array([cjk[kLow,dyada-1],cjk[kLowMid,dyada-1],cjk[k,dyada-1],cjk[kHighMid,dyada-1],cjk[kHigh,dyada-1]])
            cjk[k,dyada] = (conv_mask*sub_matrix).sum()
            djk[k,dyada-1] = cjk[k,dyada-1] - cjk[k,dyada]
            
        kShift = 2 * kShift 

    ts = tr 
    for dyada in np.arange(0,NScal-1):
        wps[1,dyada] = (djk[:,dyada]**2).sum()/sig_len
        tr = 2 * tr
        wps[0,dyada] = tr

    t_fin = datetime.datetime.now()

    print 'Total Time elapsed ', t_fin-t_ref
                
    return cjk,djk,wps
