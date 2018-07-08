import os, sys, glob
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy import interpolate
from astropy.io import fits

# My tools!!
from CraamTools.filters import RunningMean as rm
from CraamTools.wavelet import aTrous
from CraamTools.fit import Gauss
import calTb

# Convertion factor btwn Tb and flux density
k_bolztmann= 1.38065800E-23
speed_light= 2.99792458E+08
freq       = 30.0E+12
pxSize     = 8.67
Om         = np.radians(pxSize/3600)**2
tb2fx      = 2.0E+00*k_bolztmann*freq*freq*Om/speed_light/speed_light

###################################################################
#
# getFlirFlux
#
# python program to extract the flux of the IR source during the
# SOL2017-09-06T12 event observed with the FLIR camera installed
# at CRAAM. Images were converted to FITS from the original FLIR
# Public Format (FPF) and aligned using align2.py script for
# aligning.
#
# After many trials I decided to use a cross centered on pixels
# (118,195) with 3 pixels in row and column directions. The
# background is a square box with corners at (113,191) and (121,199)
#
# A better technique seems to use a wavelet transform. I implemented
# the 'a trous' triangle 2D. Transform a subset of the image centered
# over the source. Two different methods evaluate the brightness temperature
# 1) take the mean of the square at scale = 2 (can be changed) or
# 2) determine the location of the original spot (before flaring) and
# return always the wavelet coeficient corresponding to this location at level 2.
# Using wavelet reduces the fluctuations, but introduces an uncertainty
# over the absolute returned value. An special method to evaluate the relation
# between the direct obseravtion of a temperature and the wavelet
# coefficient is also included in the module.
#
# There is also another method to compute a possible instrumental
# derive: getRefLev (and getRefLevWVT) computes the non-flaring spot
# brightness temperature, referred to the first frame. It is observed
# a change of arounf 10K in Tb along the event. This valu can be directly
# subtracted from the original Tb obtained.
#
# The arbitrary units are converted to brightness temperature for every
# image.  I used conventionally 5000 K for the quiet sun at 30 THz (check!)
# Every flux is thus converted to temperature and subtracted from a
# reference (first value obtained).
#
# There is a module to write a fits file and another one to read
# goes data.
#
# Modules
#
# getVersion() : returns the version of this program.
# wvt2tb()     : compares the Tb obtained of an artificial flaring source using
#                different methods.
# getGoes()    : reads the goes fits file.
# getRefLev()  : gets the spot Tb and flux (direct method)
# getRefLevWVT(): gets the spot Tb and flux (wavelet method)
# getFlux()    : extract and calibrates the source flux.
# getFluxWVT() : extract and calibrates the source flux (wavelet method)
# getMS()      : converts datetime to ms since 0 UT.
# TimeProfile  : extract and creates a dictionary with the data.
# getCal()     : get calibration factor for an image.
# plot()       : simple plot procedure.
# writeFits()  : writes a fits file with FLIR data.
# getTime()    : convert the string time i the header of a fits image to datetime.
#
##################################################################################


def getVersion():
    return '2018-04-24T17:00BRT'

"""
####
#
# This wvt2tb explores the relationship between the wavelet coeifficients 
# and the real amplitude of a Gaussian source.

def wvt2tb(wvt_scale):
    nx = 35
    ny = 35
    xx = np.linspace(0,nx-1,nx)
    yy = np.linspace(0,ny-1,ny)
    sx = 0.4
    sy = 0.4
    a0 = 0.0
    a1 = 100.0
    
    xp = np.matmul ( (np.ones(ny)).reshape(ny,1)  , (xx-nx/2).reshape(1,nx)/sx )
    yp = np.matmul ( (yy-ny/2).reshape(ny,1)/sy , (np.ones(nx)).reshape(1,nx)  )
    
    u = np.exp(-0.5 * (xp*xp + yp*yp))
    s = a0 + a1 * u

    c,d=aTrous.triangle2D(s)

    par,zfit,cov=Gauss.fit2d(d[:,:,wvt_scale]**2)
    w_tmax=np.sqrt(d[int(np.round(par[4])),int(np.round(par[5])),wvt_scale]**2)
    
    Row_flr_0 = 13
    Row_flr_1 = 18

    Col_flr_0 = 13
    Col_flr_1 = 18

    src_area = (Row_flr_1-Row_flr_0)*(Col_flr_1-Col_flr_0)        
    w_tmean = np.sqrt((d[Row_flr_0:Row_flr_1,Col_flr_0:Col_flr_1,2]**2).sum()/src_area)

    par,zfit,cov=Gauss.fit2d(s)
    tmax = s[int(np.round(par[4])),int(np.round(par[5]))]

    kernel = np.matmul( np.array([[1./16],[1./4],[3./8],[1./4],[1./16.]]), np.array([[1./16,1./4,3./8,1./4,1./16]]) )
    twmean = (kernel*s[Row_flr_0:Row_flr_1,Col_flr_0:Col_flr_1]).sum()

    return tmax,twmean,w_tmean,w_tmax
"""

def wvt2tb(z,slice_y,slice_x,wvt_scale,x0,sx0,y0,sy0):
    
    wvTb,Ok = getFluxWVT(z[slice_y,slice_x],wvt_scale,x0,sx0,y0,sy0,method='fit')
    Tb = calTb.Tb0(z)    
    return np.abs(Tb/wvTb)

def getRefLev(z,flr_src):
    
    """
    Obtains the temperature and flux of the non-flaring spot.
    It uses a direct approach with two different shapes (cross, box) 
    and the median over the predefined area.
    """
    
    Row_ref_0 =  83
    Row_ref_1 = 100
    
    Col_ref_0 = 160  
    Col_ref_1 = 177  

    Row_flr_0 =  88
    Row_flr_1 =  93
    
    Col_flr_0 = 163
    Col_flr_1 = 168

    kernel = np.matmul( np.array([[1./16],[1./4],[3./8],[1./4],[1./16.]]), np.array([[1./16,1./4,3./8,1./4,1./16]]) )
    sum1 = z[Row_ref_0:Row_ref_1,Col_ref_0:Col_ref_1].sum()
    bkg_area = (Row_ref_1-Row_ref_0)*(Col_ref_1-Col_ref_0)
    
    if (flr_src == 'cross'):
        sum2 = z[116,195]+z[117,194:197].sum()+z[118,195]
        src_area = 5
    elif (flr_src == 'box'):
        sum2 = z[Row_flr_0:Row_flr_1,Col_flr_0:Col_flr_1].sum()
        src_area = (Row_flr_1-Row_flr_0)*(Col_flr_1-Col_flr_0)
    elif (flr_src == 'wbox'):
        sum2 = z[Row_flr_0:Row_flr_1,Col_flr_0:Col_flr_1].sum()
        flr  = (kernel*z[Row_flr_0:Row_flr_1,Col_flr_0:Col_flr_1]).sum()
        src_area = (Row_flr_1-Row_flr_0)*(Col_flr_1-Col_flr_0)
    elif (flr_src == 'median'):
        sum2 = np.median(z[Row_flr_0:Row_flr_1,Col_flr_0:Col_flr_1])
        src_area = 1.0
        
    ref = (sum1-sum2)/(bkg_area-src_area)
    if (flr_src != 'wbox'):
        Tb   = sum2 / src_area - ref
    else:
        Tb   = flr - ref

    return Tb

def getFluxWVT(z,wvt_scale,x0,sx0,y0,sy0,method='max'):

    cjk,djk=aTrous.triangle2D(z)
    
    
    if method == 'max':
        return djk[y0,x0,wvt_scale],True
    elif method == 'fit':
        nx=20
        ny=20
        try:
            p,zf,cov=Gauss.fit2d(djk[5:25,5:25,wvt_scale]**2,a=[0,1500,sx0,sy0,x0,y0])
            if ( ((p[4]-x0)**2+(p[5]-y0)**2) > 16 ) | (p[1] < 0) :
                return -np.sqrt(np.mean(djk[14:19,16:21,wvt_scale]**2)),False
            else:
                return -np.sqrt(p[1]),True
        except:
            return -np.sqrt(np.mean(djk[14:19,16:21,wvt_scale]**2)),False
    else:
        Row_flr_0 = 15
        Row_flr_1 = 18
    
        Col_flr_0 = 16
        Col_flr_1 = 19

        src_area = (Row_flr_1-Row_flr_0)*(Col_flr_1-Col_flr_0)

        Tb = np.sqrt((djk[Row_flr_0:Row_flr_1,Col_flr_0:Col_flr_1,wvt_scale]**2).sum()/src_area)
        return -Tb, True
    
def getFlux(z,flr_src):

    Row_ref_0 = 108
    Row_ref_1 = 125
    
    Col_ref_0 = 187  
    Col_ref_1 = 204  

    Row_flr_0 = 114
    Row_flr_1 = 119
    
    Col_flr_0 = 193
    Col_flr_1 = 198

    sum1 = z[Row_ref_0:Row_ref_1,Col_ref_0:Col_ref_1].sum()
    bkg_area = (Row_ref_1-Row_ref_0)*(Col_ref_1-Col_ref_0)

    kernel = np.matmul( np.array([[1./16],[1./4],[3./8],[1./4],[1./16.]]), np.array([[1./16,1./4,3./8,1./4,1./16]]) )

    if (flr_src == 'cross'):
        sum2 = z[116,195]+z[117,194:197].sum()+z[118,195]
        src_area = 5
    elif (flr_src == 'box'):
        sum2 = z[Row_flr_0:Row_flr_1,Col_flr_0:Col_flr_1].sum()
        src_area = (Row_flr_1-Row_flr_0)*(Col_flr_1-Col_flr_0)
    elif (flr_src == 'wbox'):
        sum2 = z[Row_flr_0:Row_flr_1,Col_flr_0:Col_flr_1].sum()
        flr  = (kernel*z[Row_flr_0:Row_flr_1,Col_flr_0:Col_flr_1]).sum()
        src_area = (Row_flr_1-Row_flr_0)*(Col_flr_1-Col_flr_0)
    elif (flr_src == 'median'):
        sum2 = np.median(z[Row_flr_0:Row_flr_1,Col_flr_0:Col_flr_1])
        src_area = 1.0
        
    ref = (sum1-sum2)/(bkg_area-src_area)
    if (flr_src != 'wbox'):
        Tb   = sum2 / src_area - ref
    else:
        Tb   = flr - ref

    return Tb

def getSrcXY(z,wvt_scale):

    nx=20
    ny=20
    cjk,djk=aTrous.triangle2D(z)
    p,zfit,c = Gauss.fit2d(djk[5:25,5:25,wvt_scale]**2)
    return int(np.round(p[5])),p[3],int(np.round(p[4])),p[2]
    
def getCal(fits):
    h,x      = np.histogram(np.ravel(fits[0].data),bins=200)
    sky      = (x[np.argmax(h[0:100])] + x[np.argmax(h[0:100])+1])/2
    quiet    = (x[np.argmax(h[100:])+100] + x[np.argmax(h[100:])+101])/2
    Tb_quiet = 5.0e+03
    dsun     = quiet - sky
    return Tb_quiet/dsun,sky

def getMS(f):
    ms=[]
    for dt in f['time']:
        ms.append(dt.hour*3600000L+dt.minute*60000L+dt.second*1000L+dt.microsecond/1000)
    return np.asarray(ms)

def getTime(fitsimage):

    year=2017
    month=9
    day=6

    tstring=fitsimage[0].header[13]
    hour=int(tstring[11:13])
    min=int(tstring[14:16])
    sec=int(tstring[17:19])
    msec=int( (float(tstring[17:23])-sec)* 1.0e6)
    return dt.datetime(year,month,day,hour,min,sec,msec),hour*3600.0+min*60.0+sec+msec/1.0e6

def plot(g,f,mean=True,logscale=False,rmean=False,n=50):

    fig=plt.figure()
    ax1=fig.add_subplot(111)

    if (rmean):
        flux=rm.rm1d(f['Tb'],len(f['Tb']),n)
        time=f['time']
    else:
        flux=f['Tb']
        time=f['time']

#    plt.ylim(ymax=np.max(flux))
    plt.ylim(ymax=50.0)
#    plt.ylim(ymin=np.min(flux))
    plt.ylim(-1)
    plt.ylabel('30 THz [K]')
    plt.xlabel('UT')
    
    if (logscale):
        ax1.semilogy(time,flux)
    else:
        ax1.plot(time,flux)
                     
    x0=np.argmax(np.where(g['dtime'] <= f['time'][0] ))
    x1=np.argmax(np.where(g['dtime'] <= f['time'][-1]))

    ax2=ax1.twinx()
    ax2.set_ylabel('GOES [W/m**2]')
    if (logscale):
        ax2.semilogy(g['dtime'][x0:x1],g['flow'][x0:x1],'r')
    else:
        ax2.plot(g['dtime'][x0:x1],g['flow'][x0:x1],'r')
        
    plt.show()

    return fig

def smooth(f,N):
    x=np.where(~np.isnan(f['Tb']))
    tbi=interpolate.interp1d(f['ms'][x],f['Tb'][x],kind='nearest')
    tbs=tbi(f['ms'])
    tbs=rm.rm1d(tbs,N)
    return tbs

def background():
    """
    background: it is obtained from the light crve of the non-flaring sunspot,
                and normalized to the flaring light curve.  

                Prepared for the whole data set (6600 frames)

    Author: @guiguesp - 2018-03-02

    """
    p = [-9.17674652E-14, 9.60992374E-10, -2.16569032e-06, -2.54233047E-04, 1.74019586E+00]
    p[2] = 1.1*p[2]
    x = np.linspace(0,6599,6600)
    
    return p[0]*x**4 + p[1]*x**3+p[2]*x**2+p[3]*x
    
def writeFits(f,FITSfname):
    
    hdu = fits.PrimaryHDU()
    hdu.header.append(('origin','CRAAM/Universidade Presbiteriana Mackenzie',''))
    hdu.header.append(('telescop','30 THZ','FLIR camera'))
    hdu.header.append(('observat','CRAAM',''))
    hdu.header.append(('station','Lat = -23.547461, Lon = -46.652255, Height = 0.75 km',''))
    hdu.header.append(('tz','GMT-3',''))
    
    hdu.header.append(('date-obs','2017-09-06 {0:%Y-%m-%dT%H:%M:%S}'.format(f['time'][0]),''))
    hdu.header.append(('t_start','{0:%Y-%m-%dT%H:%M:%S}'.format(f['time'][0]),''))
    hdu.header.append(('t_end','{0:%Y-%m-%dT%H:%M:%S}'.format(f['time'][-1]),''))
    hdu.header.append(('data_typ','Tb','Assumming quiet sun is 5000 K'))
    hdu.header.append(('recal',f['recalibration'],'Recalibration factor from wavelet decomposition'))
    

    hdu.header.append(('comment','Brightness temp time profiles extracted from individual fits files.'))
    hdu.header.append(('comment','COPYRIGHT. Grant of use.',''))
    hdu.header.append(('comment','These data are property of Universidade Presbiteriana Mackenzie.'))
    hdu.header.append(('comment','The Centro de Radio Astronomia e Astrofisica Mackenzie is reponsible'))
    hdu.header.append(('comment','for their distribution. Grant of use permission is given for Academic ')) 
    hdu.header.append(('comment','purposes only.'))

    hdu.header.append(('history','Created by getFlirFlux.py version '+getVersion()))
    hdu.header.append(('history','Created on '+ str(dt.datetime.today())))
    hdu.header.append(('history','Created by '+ os.getlogin() + ' on ' + os.uname()[1]))
    #### Some comments for the current version. Comment when needed
    hdu.header.append(('history','Light curve obtained with wavelet decomposition.'))
    hdu.header.append(('history','Background defined with the non-flaring spot.'))
    hdu.header.append(('history','Light curve is background subtracted'))
    hdu.header.append(('history','NAN values replaced with interpolation to nearest'))
    hdu.header.append(('history','Smoothed light curve'))
                      

    fits_cols = []
    ms = getMS(f)
    col       = fits.Column(name   = 'time',
                            format = 'J'   ,
                            unit   = 'ms'  ,
                            bscale = 1.0   ,
                            bzero  = 0.0   ,
                            array  = ms)

    fits_cols.append(col)

    col       = fits.Column(name   = 'Tb' ,
                            format = 'E'    ,
                            unit   = 'K'    ,
                            bscale = 1.0    ,
                            bzero  = 0.0    ,
                            array  = f['Tb'])
    fits_cols.append(col)
    
    col       = fits.Column(name   = 'cal'  ,
                            format = 'E'    ,
                            unit   = 'K/au' ,
                            bscale = 1.0    ,
                            bzero  = 0.0    ,
                            array  = f['cal'])
    fits_cols.append(col)

    col       = fits.Column(name   = 'sky'  ,
                            format = 'E'    ,
                            unit   = 'au' ,
                            bscale = 1.0    ,
                            bzero  = 0.0    ,
                            array  = f['sky'])
    fits_cols.append(col)

    col       = fits.Column(name   = 'fitsfil'                       ,
                            format = str(len(f['fitsname'][0]))+'A' ,
                            unit   = ''                              ,
                            array  = f['fitsname'])
    fits_cols.append(col)

    coldefs = fits.ColDefs(fits_cols)
    tbhdu   = fits.BinTableHDU.from_columns(coldefs)
    # Comments and other headers
    tbhdu.header.append(('comment','Time is in milliseconds since 0 UT',''))
    tbhdu.header.append(('comment','Calibration is in K per Arbitrary Unit',''))
    
    hduList = fits.HDUList([hdu,tbhdu])
    hduList.writeto(FITSfname,overwrite=True)

    return

def doInterpol(f):
    
    good = np.where(~ np.isnan(f['Tb']) & f['fitOK'] )

    s=[]
    for time in f['time']:
        s.append(time.hour*3.6E+03+time.minute*6.0E+01+time.second+time.microsecond*1.0E-06)
        
    s = np.asarray(s)
    
    Tb = f['Tb'][good]
    Fx = f['flux'][good]
    S  = s[good]
    
    tbi = interpolate.interp1d(S,Tb,kind='cubic')
    fxi = interpolate.interp1d(S,Fx,kind='cubic')

    x  = np.where(s <= S[-1])
    s  = s[x]
    dt = f['time'][x]
    tb = tbi(s)
    fx = fxi(s)
    
    return tb,fx,dt,good

def TimeProfile(base_dir,flr_src='box',wvt=False,Nsteps=10,Nmax=None):
    
    fitsfiles=glob.glob(base_dir+"/FLIR*.fits")
    fitsfiles.sort()
    if Nmax:
        fitsfiles=fitsfiles[:Nmax]
        
    # Lists for storing the results
    filelist = []
    Tblist = []
    mslist = []
    tlist  = []
    oklist = []
    cal    = []
    bkgr   = []
    spotTb = []
    spotoklist = []

    #Slices definitions
    slice_flr_y = slice(100,135,1)  # The flaring region for wavelet decomposition
    slice_flr_x = slice(177,212,1)

    slice_spot_y = slice(74,109,1)  # The reference spot for wavelet decomposition
    slice_spot_x = slice(151,186,1)
    
#    t_ref = dt.datetime.now() # start computations

    first=True
    i=1
    for fitsfile in fitsfiles[::Nsteps]:
        hdu=fits.open(fitsfile)

        # Get the Time & Cal factor
        au2tb,sky=getCal(hdu)
        #Get the Flux
        z = (hdu[0].data-sky)*au2tb
        wvt_scale = 2
        
        if wvt:
            if first:
                x0,sx0,y0,sy0               =  getSrcXY(z[slice_flr_y,slice_flr_x],wvt_scale)
                xspot,sx0spot,yspot,sy0spot =  getSrcXY(z[slice_spot_y,slice_spot_x],wvt_scale)
                recalibration               =  wvt2tb(z,slice_flr_y,slice_flr_x,wvt_scale,x0,sx0,y0,sy0)
                
            Tb,fitOK =  getFluxWVT(z[slice_flr_y,slice_flr_x],wvt_scale,x0,sx0,y0,sy0,method='fit')
            SpotTb,spotOK   =  getFluxWVT(z[slice_spot_y,slice_spot_x],wvt_scale,xspot,sx0spot,yspot,sy0spot,method='fit')
            print 'Frame {0:3d} out {1:5d}'.format(i,int(round(float(len(fitsfiles)/Nsteps))))
            i=i+1
                
        else:
            Tb = getFlux(z,flr_src)
            SpotTb = getRefLev(z,flr_src)
            fitOK = True
            x0 = -1
            y0 = -1

        if first:
            first = not first
            refTb = Tb
            refSpotTb = SpotTb
            
        filelist.append(fitsfile)
        Tblist.append((Tb-refTb)*recalibration)
        spotTb.append((SpotTb-refSpotTb)*recalibration)
        spotoklist.append(spotOK)
        oklist.append(fitOK)
        cal.append(au2tb)
        bkgr.append(sky)
        dt,ms = getTime(hdu)
        tlist.append(dt)

        hdu.close()
        
#    t_end = dt.datetime.now()  # End computations
#    print 'Total Time elapsed ', t_end-t_ref

    # Create a Dictionary with the data and information on how it was obtained
    flir={}
    flir.update({'time':np.asarray(tlist)})
    flir.update({'ms':getMS(flir)})
    flir.update({'Tb':np.asarray(Tblist)})
    flir.update({'cal':np.asarray(cal)})
    flir.update({'sky':np.asarray(bkgr)})
    flir.update({'recalibration':recalibration})
    flir.update({'ref_Tb':refTb})
    flir.update({'spot_Tb':np.asarray(spotTb)})
    flir.update({'spotOK':np.asarray(spotOK)})
    flir.update({'spot_refTb':np.asarray(refSpotTb)})
    flir.update({'fitsname':filelist})
    if wvt:
        flir.update({'method':'wavelet'})
        flir.update({'ref_X0':x0})
        flir.update({'ref_Y0':y0})
        flir.update({'ref_sX0':sx0})
        flir.update({'ref_sY0':sy0})
        flir.update({'fitOK':np.asarray(oklist)})
    else:
        flir.update({'method':'direct'})
        
    return flir
