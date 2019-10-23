'''
Library imports.
All of these libraries are included in Anaconda 5.1.0.
This software was developed and debugged within Anaconda 5.1.0.
'''

import sys
import multiprocessing as mp
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import serial, time, pandas, datetime, string, traceback
from scipy.optimize import curve_fit
from scipy.signal import gaussian, tukey
from scipy.interpolate import UnivariateSpline
from scipy.ndimage.filters import gaussian_filter
from configparser import SafeConfigParser
import PIL.ImageGrab

'''
Creates a configuration reader object and loads config.ini into it.
Generally, variables with ALL_CAPS names are read from the configuration file
and can be changed by changing the config file and restarting the program.
Ie. We don't expect these variables to change while the program is running.
'''

config = SafeConfigParser()
config.read("config.ini")

RED_COLOr=(255,0,0)
GREEN_COLOR=(0,255,0)
BLUE_COLOR=(0,0,255)
DGREEN_COLOR=(0,100,0)
DBLUE_COLOR=(0,0,128)
OLIVE_COLOR=(128,128,0)
ORANGE_COLOR=(255,165,0)
GOLD_COLOR=(184,134,11)
CYAN_COLOR=(0,128,128)
PURPLE_COLOR=(128,0,128)
PINK_COLOR=(255,20,147)
BROWN_COLOR=(139,69,19)

COLOR_SPECTRUM = eval(config.get("GUI", 'Spectrum_Plot_Color'))
COLOR_INTENSITY = eval(config.get("GUI", 'Intensity_Plot_Color'))
COLOR_POSITION = eval(config.get("GUI", 'Position_Plot_Color'))

SERIAL_PORT = config.get("DataAcquisition", "Serial_Port")
SERIAL_TIMEOUT = config.getfloat("DataAcquisition", "Serial_Timeout")
PROG_BAR_UPDATE = config.getint("DataAcquisition", "Progress_Bar_Update_Interval")
SECS_BTW_RETRY = config.getfloat("DataAcquisition", "Secs_Btw_Serial_Retry")

COUNTS_TO_DIST = config.getfloat("PhysicalConstants", "Stage_mm_per_Count")
TICK_MICROS = config.getfloat("PhysicalConstants", "Stage_Tick_Microseconds")
STAGE_MIN_STEP = config.getint("DataAcquisition", "Stage_Min_Step_Size")
#display customizations for spectrum plot (for increasing visibility 
#when the computer monitor is far away)
FONT_SIZE = config.getint("GUI", 'Spectrum_Plot_Font_Size')
DEFAULT_XMIN = config.getfloat("GUI", 'Spectrum_Default_Xmin')
DEFAULT_XMAX = config.getfloat("GUI", 'Spectrum_Default_Xmax')
SPECTRUM_LINE_WIDTH = config.getint("GUI", 'Spectrum_Plot_Line_Width')


axisFont = QtGui.QFont()
axisFont.setPointSize(FONT_SIZE)
labelStyle = {'font-size': '{}pt'.format(FONT_SIZE)}


def InteractWithTeensy(pipe_TeensyEnd):
    '''
    See documentation for a diagram showing the flow of this function.
    This process is designed to be run by a subprocess, 
    since it does not halt or contain a return statement.
    '''
    
    #Loop to reattempt opening serial port until it is opened.
    serialOpenSuccess = False
    while not serialOpenSuccess:
        try:
            ser = serial.Serial(port=SERIAL_PORT, baudrate=115200, 
                                bytesize=8, timeout=SERIAL_TIMEOUT, 
                                parity='N', stopbits=1)
            print("Using", ser.name)
            serialOpenSuccess = True
        except Exception as errorMessage:
            print("Error starting serial communications, \
            trying again in {}s:".format(SECS_BTW_RETRY))
            print(errorMessage)
            traceback.print_tb(errorMessage.__traceback__)
            serialOpenSuccess = False
            time.sleep(SECS_BTW_RETRY)
    
    #Initialize variables for holding read values and for timing.
    posBatch = []
    intensityBatch = []
    prevTime = time.clock()

    #Main loop that the subprocess will continuously run
    while True:
        #handle stage directions from GUI

        #if data was sent from GUI (ie. something waiting at pipe's Teensy end):
        if pipe_TeensyEnd.poll(): 
            #receive the sent data
            receivedData = pipe_TeensyEnd.recv() 

            #receivedData is formatted like ("string to write", 
            #                                 time delay to wait after executing command)
            ser.write(receivedData[0])
            time.sleep(receivedData[1])

            #while response from Teensy is in serial buffer:
            while ser.inWaiting(): 
                #read the entire buffer and print, send to GUI 
                msg=repr(ser.read(ser.inWaiting()).decode('ascii','ignore'))
                print(msg)
                pipe_TeensyEnd.send(msg)

            continue #go back to beginning of main loop

        #attempt to read 64 bytes
        byteBatch = ser.read(64)

        #handle Teensy going into configuration mode;
        #ie. Teensy does not produce 64 bytes
        if len(byteBatch) != 64:
            currentMode = 'Configuration'

            #if there is data in the batches, send it to GUI now
            if len(posBatch) != 0:
                posBatch, intensityBatch, prevTime = sendToPipe(
                    pipe_TeensyEnd, intensityBatch, posBatch, prevTime)
            
            #send mode indicator to GUI    
            pipe_TeensyEnd.send(currentMode)
            continue
        

        #handle teensy being in measurement mode
        newPosBatch = [int.from_bytes(byteBatch[index:index+2], 
                                      byteorder = "big") for index in range(0,64,4)]
        # print(['marker byte:',str(newPosBatch),',  ',str(np.median(newPosBatch))])
        
        #65535 and 65278 correspond to a buffer filled with 
        #[255, 255, ...] and [254, 254, ...] respectively. 
        #They indicate the digital trigger for the stage has changed 
        #(either rising or falling edge.)
        if np.median(newPosBatch) == 65535: 
            posBatch, intensityBatch, prevTime = sendToPipe(
                pipe_TeensyEnd, intensityBatch, posBatch, prevTime)
            # print('forward stage movement. Send data to PIPE')

        elif np.median(newPosBatch) == 65278: 
            #since the stage was travelling in an opposite direction 
            #as in the previous case, we reverse the data.
            # print('reverse stage movement. Send data to PIPE')
            intensityBatch.reverse()
            posBatch.reverse()

            posBatch, intensityBatch, prevTime = sendToPipe(
                pipe_TeensyEnd, intensityBatch, posBatch, prevTime)

        else:
            #otherwise, the buffer is filled with real data that 
            #is passed onto the intensity and position queues.
            # print('transforming bytes to data')
            posBatch.extend(newPosBatch)
            newIntensityBatch = [int.from_bytes(byteBatch[index:index+2], 
                                                byteorder = "big") for index in range(2,64,4)]
            intensityBatch.extend(newIntensityBatch)

        
        if len(posBatch) % PROG_BAR_UPDATE == 0:
            #the length of the position queue is reported 
            #every time it is divisible by a constant.
            #reduce the constant to get more frequent reporting 
            #(in effect, a smoother progress bar in the GUI.)
            pipe_TeensyEnd.send(len(posBatch))



def sendToPipe(pipeEnd, intensityBatch, posBatch, prevTime):
    '''
    Function to send intensity and position queues through the given pipe end.
    Also designed to compute time elapsed since the last time this function
    was called. 
    Returns two empty arrays and the time the function was called; 
    this is designed to be assigned to the intensity and position queues 
    (to empty them) and for the time to be passed back to the function 
    at the next time the function is called to provide the timing functionality.
    '''
    currentTime = time.clock()

    pipeEnd.send((intensityBatch, posBatch))
    pipeEnd.send(currentTime - prevTime)

    return [], [], currentTime



def clipMarginOnly(array, margin = 0.1):
    '''
    Takes: 
    Array
    Optional margin value (defaults to 0.1 if none supplied)

    Clips the first and last (margin / 2) percent of the array, 
    so that in total, margin percent of the array is removed. 

    Returns: clipped array

    '''
    n = len(array)
    startIndex = int(round(n * (margin/2)))

    if margin == 0:
        return array
    else:
        return array[startIndex:-startIndex]



def removeBackground(queue, samplePct = 0.05):
    '''
    Takes: 
    Array of numbers
    Optional percentage of array to sample

    Averages the first samplePct percent of queue, which is assumed to be a 
    baseline background level for the values in the queue. Subtracts this 
    background from every value in the array.

    Returns: array with background subtracted from every value in queue. 
    '''
    numToSample = int(samplePct * len(queue))
    if numToSample<1:
        numToSample=1
    background = sum(queue[:numToSample])/numToSample

    return [x - background for x in queue]




PROG_BAR_MIN_TIME = config.getfloat("GUI", "Progress_Bar_Min_Time")

def readPipe(pipe_GUIEnd, teensyModeWidget, consoleWidget): #progressBarWidget, 
    '''
    See documentation for flowchart of this function.
    The GUI thread runs this function to read the data supplied by the 
    Teensy interaction thread. GUI widgets are passed to this function
    so that their properties are updated. 
    '''

    #receive data sent from Teensy pipe end
    receivedData = pipe_GUIEnd.recv()

    #if received data is a tuple, this is a tuple with intensity and position 
    #data, and the immediate next received data is the time taken for this 
    #batch of data
    if isinstance(receivedData, tuple):
        intensityQueue = receivedData[0]
        posQueue = receivedData[1]

        timeTaken = pipe_GUIEnd.recv()
        assert isinstance(timeTaken, float)

        #set progress bar's 100% value to current length of intensity queue 
        #since we expect the next queue to be a similar length. However, if 
        #the time taken for this batch was very short, the bar displays a 
        #less distracting "busy" indicator instead.        
        # progressBarWidget.setMaximum(len(intensityQueue))
        # if timeTaken < PROG_BAR_MIN_TIME:
        #   progressBarWidget.setMaximum(0)

        #return data to the GUI update function.
        return intensityQueue, posQueue, timeTaken

    #if an integer is received, we expect this to be the current length of 
    #the position queue. The progress bar is updated accordingly.
    elif isinstance(receivedData, int):
        currentMode = 'Measurement'
        teensyModeWidget.setText(currentMode)
        # progressBarWidget.setValue(receivedData)
        
        #return empty arrays so that update function call 
        #ends quickly for program responsiveness.
        return [], [], 0

    #if we get this mode indicator string, we update the mode display in the 
    #GUI and change the progress bar to an indefinite "busy" indicator.
    elif receivedData == 'Configuration':
        currentMode = 'Configuration'
        teensyModeWidget.setText(currentMode)
        # progressBarWidget.setMaximum(0)
        
        #return empty arrays so that update function call 
        #ends quickly for program responsiveness.
        return [], [], 0

    #if we get any other strings, it is a reply from the stage controller 
    #that is passed via the Teensy to the computer. We print this message 
    #and output it to the GUI's console.
    elif isinstance(receivedData, str):
        consoleWidget.appendPlainText(receivedData)

        #return empty arrays so that update function call 
        #ends quickly for program responsiveness.
        return [], [], 0

def FWHM_fit(xdata,ydata):
    maxval=max(ydata)
    maxpos=np.argmax(ydata)
    FWHM01=(np.abs(ydata[:maxpos]-maxval/2)).argmin()
    FWHM02=(np.abs(ydata[maxpos:]-maxval/2)).argmin()+maxpos
    x0=[maxval,xdata[maxpos],(xdata[FWHM01]-xdata[FWHM02]),ydata[2]] #
    [popt, pcov] = curve_fit(gaus_fun, xdata, ydata, p0=x0)
    fitData= (gaus_fun(xdata, *popt))
    FWHM=np.abs( 2*np.sqrt(2*np.log(2))*popt[2] );
    return fitData, FWHM, popt[1]

def gaus_fun(x, *p):
    [a,x0,sigma,background]=p
    calc_data=background+a*np.exp(-(x-x0)**2/(2*sigma**2))
    return(calc_data)

def acorr_analysis(tsc,AC_raw):
    '''
    Takes: 
    Array that represents the amount of time it takes light to travel the 
    extra distances that the mirror moves. Therefore, this array contains 
    information about the position of the stage.
    Array that represents the intensity of the light hitting a sensor: 
    two-photon absorption on the detector corresponding to interferometric
    auto-correlation.

    Returns calculated intensity autocorrelation, converted into an array of 
    delay times [fs] and the corresponding intensities.
        
    Not doing interpolation of 'tsc' into linearly spaced array because 
    we need resault in the same time domain.
    '''
    
    tsc=tsc*1e15
    AC_raw=np.asarray(AC_raw)
    L= len(AC_raw) # int (( np.max(xsc1)-np.min(xsc1)) *10**13)*(2**16) #
    AC_raw=-AC_raw/max(abs(AC_raw))
    AC_fft= np.fft.fft(AC_raw)
    filterN=ACfilterEdit.value()

    chirp_fft=AC_fft

    # Here I calculate intensity AC from f=0 signal:
    AC_fft[filterN:-filterN]=0    
    AC_env=np.real(np.fft.ifft(AC_fft))
    AC_filtered_norm=AC_env/np.max(AC_env)

    [fitData,FWHMt,t0]=FWHM_fit(tsc,AC_filtered_norm)
    maxpos=np.argmax(AC_env)
    tsc=tsc-tsc[maxpos]

    '''
    As an estimate of linear dispersion I measure "FWHM" of 'w' signal. For that I:
    1. filter out signal at f=0,
    2. find position of maximum in freq domain - that is the 'w' signal,
    3. filter out everything except this signal,
    4. converst to time domain and take its positive part only - this will give envelope in freq domain,
    5. in freq domain take envelope (at f=0). This envelope is indicative of linear chirp and is free from noise.
    '''

    # chirp_fft[:filterN]=0
    # chirp_fft[-filterN:]=0
    # chirp_pos=np.argmax(np.abs(chirp_fft[:int(L/2)]))
    # chirp_fft[:chirp_pos-filterN*5]=0
    # chirp_fft[chirp_pos+filterN*5:-chirp_pos-filterN*5]=0
    # chirp_fft[-chirp_pos+filterN*5:]=0
    # chirp_filtered=np.real(np.fft.ifft(chirp_fft))
    # chirp_filtered[chirp_filtered<0]=0
    # chirp_env_fft=np.fft.fft(chirp_filtered)
    # chirp_env_fft[filterN:-filterN]=0
    # chirp_env=np.real(np.fft.ifft(chirp_env_fft))
    # chirp_filtered_norm=chirp_env/np.max(chirp_env)
    # [fitChirp,FWHM_chirp,t0_chirp]=FWHM_fit(tsc,AC_filtered_norm)
    FWHM_chirp=0

    '''
    use peak-detect function to estimate AC envelope
    '''
    # AC_peaks_array = peak_detect(AC_raw, (max(AC_raw) - min(AC_raw)) / 70) # [:,0]
    # AC_peaks=AC_peaks_array[:,1]
    # AC_peaks_ax=tsc[AC_peaks_array[:,0].astype(int)]
    # AC_peaks_norm=AC_peaks/np.max(AC_peaks)

    # [fitPeaks,FWHM_peaks,t0_peaks]=FWHM_fit(AC_peaks_ax,AC_peaks_norm)



    return(tsc,AC_filtered_norm,fitData,FWHMt)

def FFT_to_spectrum(xsc1,data1):
    '''
    Takes: 
    Array that represents the amount of time it takes light to travel the 
    extra distances that the mirror moves. Therefore, this array contains 
    information about the position of the stage.
    Array that represents the intensity of the light hitting a sensor. 
    This is expected to vary as the laser beam changes between constructive 
    and destructive interference.

    Returns Fourier transform results, converted into an array of 
    wavelengths [nm] and the corresponding intensities.
    '''

    L= len(data1) # int (( np.max(xsc1)-np.min(xsc1)) *10**13)*(2**16) # 
    tsc_lin=np.linspace(np.min(xsc1),np.max(xsc1),L)
    data_lin=np.interp(tsc_lin,xsc1,data1)

    NFFT = 2**L.bit_length()
    dat_fft= np.fft.fftshift(np.fft.fft(data_lin,n=NFFT)/L)
    df=1/(tsc_lin[2]-tsc_lin[1])
    fsc=df * ( np.linspace(0,1,NFFT)-0.5 )

    ax_wavenumbers = fsc / (SPEED_OF_LIGHT * 100) #2.9979e10 # scanned freq axis (cm-1)
    ax_nm = 1e7/ax_wavenumbers # in nm
    ax_lims=np.argmin(np.abs(ax_nm-200))
    ax_lims=np.append( ax_lims , np.argmin(np.abs(ax_nm-10000)) )
    ax_lims=np.sort(ax_lims)
        
    # Mertz phase compensation (FTIR_Griffiths_Ch4.pdf):
    dat_pwr=np.arctan(np.imag(dat_fft)/np.real(dat_fft)) 
    spectrum=np.abs(dat_fft*np.exp(-np.complex(0,1)*dat_pwr))

    # Wrong phase compensation:
    # dat_real=np.real(dat_fft)
    # dat_imag=np.imag(dat_fft)
    # dat_pwr=(dat_real**2+dat_imag**2)**0.5
        
    # dat_real*=dat_pwr
    # dat_imag*=dat_pwr
    # dat_phase=dat_real+dat_imag*1j
        
    # spectrum=np.abs(dat_fft*dat_phase)

    spectrum=spectrum[ax_lims[0]:ax_lims[1]]
    spectrum=spectrum-np.min(spectrum)
    spectrum_norm=spectrum/np.max(spectrum)


    [fitData,FWHMwv,wv0]=FWHM_fit(ax_nm[ax_lims[0]:ax_lims[1]],spectrum_norm)
    FWHMt_gaus=0.44/( (1e7/(wv0-FWHMwv/2)-1e7/(wv0+FWHMwv/2))*2.99792e10*1e-15)
    FWHMt_sec=0.315/( (1e7/(wv0-FWHMwv/2)-1e7/(wv0+FWHMwv/2))*2.99792e10*1e-15)
        
    start_pos=int(NFFT/2)+1
    return(ax_nm[ax_lims[0]:ax_lims[1]],spectrum_norm,fitData,FWHMwv,FWHMt_gaus)


def modifiedGaussian(x, baseline, amplitude, mean, deviation):
    '''
    Takes: 
    x value to evaluate function at
    Mean of Gaussian function (center; a horizontal translation constant)
    Amplitude of Gaussian function (height of function at mean; 
                                    a vertical stretch factor)
    Deviation (width of the “bump” of the Gaussian function; 
               a horizontal stretch factor)
    Baseline (value of Gaussian far away from mean; 
              a vertical translation constant)

    Returns value of this modified Gaussian function, evaluated at x. 
    Used for fitting to the fringes of a pulsed laser and finding 
    zero-path-difference position.
    '''
    return baseline + amplitude*np.exp(-((x-mean)/deviation)**2)



def applyWindow(queueToWindow):
    '''
    Takes: 
    An array of numbers, to be passed to Fourier transform.
    Parameter for the windowing function: standard deviation 
    for Gaussian and alpha for Tukey.

    Generates a windowing array that has the length of the input array. 
    The windowing array represents a windowing function that has been 
    horizontally scaled so that the first and last value is very small 
    and the center value is close to 1. 

    Returns a component wise multiplication of the windowing array and 
    queueToWindow. (ie. the first value of the windowing array is 
    multiplied with the first value of queueToWindow to make the first 
    value of the returned array.)
    '''

    #reads the drop down box to see which window to apply
    if FFTfilterSelector.currentText() == 'No FFT window':
        return queueToWindow
    elif FFTfilterSelector.currentText() == 'Gaussian window with SD':
        windowingArray = gaussian(len(queueToWindow), FFTparamInput.value())
    elif FFTfilterSelector.currentText() == 'Tukey window with alpha':
        windowingArray = tukey(len(queueToWindow), alpha = FFTparamInput.value())

    #component wise multiplication
    queueToWindow = [windowingArray[index] * value for index, value in enumerate(queueToWindow)]
    return queueToWindow




PARTITION_WIDTH = config.getint("DataProcessing", "Center_Finding_Partition_Width")

def getCenter(intensityArray):
    '''
    Considering PARTITION_WIDTH values at a time, finds a partition where 
    the difference between the maximum and minimum values is greatest. 
    The position halfway between this maximum and minimum value is assumed 
    to be the center.
    PARTITION_WIDTH - approximately one period of signal oscillation, or a little more.
    '''

    indices = []
    diffs = []


    #consider PARTITION_WIDTH values at a time
    for index in range(0, len(intensityArray), PARTITION_WIDTH):

        #get max/min in this interval
        maxVal = max(intensityArray[index:index+PARTITION_WIDTH])
        minVal = min(intensityArray[index:index+PARTITION_WIDTH])

        #get indices of these extrema
        maxIndex = intensityArray.index(maxVal, index, index + PARTITION_WIDTH)
        minIndex = intensityArray.index(minVal, index, index + PARTITION_WIDTH)

        #append average index
        indices.append((maxIndex + minIndex) / 2)
        #append range of partition
        diffs.append(maxVal-minVal)
        
    #return index that corresponds to the partition with largest range
    centerIndex = indices[diffs.index(max(diffs))]
    return centerIndex



def clipArraySymmetric(array, center):
    '''
    Takes:
    Array to be clipped
    Arbitrary center index of this array, to clip the array around

    Returns:
    Array clipped in a way such that the given center index becomes 
    the actual center index of this array.
    '''
    if center < (len(array) - center):
        return array[:2*center]
    else:
        return array[-2*(len(array) - center):]



DEFAULT_SMOOTHING_NAVG = config.getint("DataProcessing", "Moving_Avg_Batch_Size")
DEFAULT_SMOOTHING_STDEV = config.getfloat("DataProcessing", "Gaussian_Kernel_StDev")

def smoothPosArray(posArray, smoothing = 0, degree = 3, 
                   nAverage = DEFAULT_SMOOTHING_NAVG, 
                   stDev = DEFAULT_SMOOTHING_STDEV):
    '''
    Takes:
    Array of numbers, typically representing stage position data
    Optional smoothing parameter to pass to spline constructor
    Optional degrees of spline
    Optional number of points to average in moving average
    Optional standard deviation to use for Gaussian kernel smoothing

    Returns:
    xData: an array of integers from 0 to the length of posArray - 1 
    (including these endpoints.)
    Scipy UnivariateSpline object that fits xData and posArray as 
    x,y values, after posArray is smoothed with kernel smoothing and 
    a moving average.
    '''

    posArray = movingAverage(posArray, n = nAverage) #moving average smoothing
    posArray = gaussian_filter(posArray, stDev) #gaussian kernel smoothing
    xData = np.arange(len(posArray))
    mySpline = UnivariateSpline(xData, posArray, 
                                s = smoothing, k = degree) 
                                #s: smoothing factor, k: degrees of spline
    return xData, mySpline




SPEED_OF_LIGHT = config.getfloat("PhysicalConstants", "Speed_Of_Light")

def convertPosToLightTime(posArray): 
    '''
    Takes: array of recorded positions of the stage, in meters 
    (calculated from HeNe interference signal)

    Performs smoothing on posArray with smoothPosArray, as we assume that 
    the posArray represents a stage that is moving continuously.

    Returns: 
    Array of times equivalent to how long it would take time to travel the 
    smoothed distances
    Array with smoothed positions
    Array with derivative of smoothed positions
    '''
    xData, splineObject = smoothPosArray(posArray)
    return (splineObject(xData) / SPEED_OF_LIGHT *  1)




REF_WAVELENGTH = config.getfloat("PhysicalConstants", "Ref_Laser_Wavelength")

def intensityToPos(refIntensityArray):
    '''
    Converts reference intensity signal (expected to be a sine wave, 
    representing alternation between constructive and destructive 
    interference) to an array of distances.
    '''
    refIntensityArray = removeBackground(refIntensityArray, 1)

    peaks = peak_detect(refIntensityArray, 
                        (max(refIntensityArray) - min(refIntensityArray)) / 2)[:,0]

    posInterp = np.interp(np.arange(len(refIntensityArray)), 
                          peaks, 
                          [counter * REF_WAVELENGTH  for counter in range(len(peaks))]
                          )

    return posInterp, peaks



def getZeros(array):
    '''
    Given a set of discrete measurements of what is assumed is a continuous 
    signal, interpolates to find indices where zeros of this signal occur.
    Returns number of zeros found.
    '''
    xData = np.arange(len(array))
    refIntensityArray = removeBackground(array, 1)
    splineFit = UnivariateSpline(xData, refIntensityArray, s = 0, k = 3)
    splZeros = splineFit.roots()
    return len(splZeros)



def movingAverage(array, n = DEFAULT_SMOOTHING_NAVG):
    '''
    Takes moving average of the given array. 
    Number of points to put into average can be optionally specified.
    '''
    dataframe = pandas.DataFrame(data = np.array(array))
    rolling = dataframe.rolling(n, min_periods = 1, center = True).mean()

    return np.transpose(rolling.values)[0]



def peak_detect(y, delta, x = None):
     """ 
    Find local maxima in y.
    Inputs:
        - y : intensity data in which to look for peaks
        - delta : a point is considered a maximum peak if it has the 
        maximal value, and was preceded (to the left) by a value lower by DELTA.
        - x : correspond x-axis (optional)
    Outputs:
        - Array with two columns. Col1 = indices / the x-values of 
        peaks, Col2 = the peak values in y
    Citation:
        - Converted by Ed Kelleher from MATLAB script at 
        http://billauer.co.il/peakdet.html.
    """

     maxtab = []
     mintab = []

     if x is None:
         x = np.arange(len(y))

     y = np.asarray(y)
     mn, mx = np.Inf, -np.Inf
     mnpos, mxpos = np.NaN, np.NaN
     lookformax = True

     for i in np.arange(len(y)):
         this = y[i]
         if this > mx:
             mx = this
             mxpos = x[i]
         if this < mn:
             mn = this
             mnpos = x[i]

         if lookformax:
             if this < mx-delta:
                 maxtab.append((mxpos, mx))
                 mn = this
                 mnpos = x[i]
                 lookformax = False
         else:
             if this > mn+delta:
                 mintab.append((mnpos, mn))
                 mx = this
                 mxpos = x[i]
                 lookformax = True
     return np.array(maxtab) #, np.array(mintab). For now, only retun the PEAKS, not troughs


def convertParamsToCounts(start, speed, distance):
    '''
    Takes:
    Commanded start position of stage, in mm from one end of motion
    Commanded speed of stage
    Commanded distance for the stage to move in halfPeriod seconds, in mm

    Returns:
    Count corresponding to center position
    Number of steps to take
    Size each step should be, in counts
    Ticks of stage controller to delay, where each tick is 25.6 microseconds
    '''

    #convert physical values to counts of stage controller
    centerCounts = int(round(start / COUNTS_TO_DIST))
    distanceTotalCounts = distance / COUNTS_TO_DIST
    halfPeriod = distance / speed
    countsPerMicros = distanceTotalCounts / (halfPeriod * 10**6)

    #given starting step size, increment it as necessary so that 
    #ticksToDelay is nonzero (ie. we can wait a few ticks before 
    #commanding the next stage position)
    stepSize = STAGE_MIN_STEP
    ticksToDelay = 0
    while ticksToDelay == 0:
        microsPerCount = stepSize / countsPerMicros
        ticksToDelay = int(round(microsPerCount / TICK_MICROS))
        stepSize += 1
    stepSize -= 1

    return centerCounts, int(round(distanceTotalCounts / stepSize)), stepSize, ticksToDelay



def getMaxWavelength(wavelengths, intensity):
    '''
    Takes: 
    Array of wavelengths
    Array of intensities associated with wavelengths

    Returns: wavelength with the highest intensity
    '''
    #return wavelengths[intensity.tolist().index(max(intensity))]
    return wavelengths[np.argmax(intensity)]



def getHalfRange(xsc, intensity):
    '''
    Takes: 
    Array of wavelengths
    Array of intensities associated with wavelengths

    Returns: absolute difference between wavelengths that correspond to 
    half of maximum intensity.
    '''
    #maxIntensityIndex = intensity.tolist().index(max(intensity))
    maxIntensityIndex = np.argmax(intensity)

    #split wavelengths and intensities into two parts, 
    #before the maximum and after the maximum.
    xsc1 = xsc[:maxIntensityIndex]
    xsc2 = np.flipud(xsc[maxIntensityIndex:])

    intensities1 = intensity[:maxIntensityIndex]
    intensities2 = np.flipud(intensity[maxIntensityIndex:])

    #now, the two parts are expected to be one-to-one functions for a 
    #Gaussian shaped spectrum. We invert the function and read the 
    #wavelength at which an intensity of 0.5 would occur, and return 
    #the difference between the wavelength for the first and second parts.
    half1 = np.interp([0.5], intensities1, xsc1)[0]
    half2 = np.interp([0.5], intensities2, xsc2)[0]

    return np.abs(half1 - half2)



DELAY_MICROS = config.getfloat("DataAcquisition", "Stage_Pause_Microseconds")

def scancommand(pipeEnd):
    '''
    Takes: GUI pipe end (values sent in through the GUI end will be 
    received by the Teensy thread at the other end.)

    Also takes a symbolic constant, DELAY_MICROS, representing 
    microseconds to pause at the ends of the motion. 

    Writes encoded binary ASCII commands to the GUI pipe end, so that 
    the thread interacting with the Teensy can read them and use the 
    Teensy to command the stage controller. 
    '''
    stageConsole.appendPlainText("Instructions are being written to Teensy. Please be patient: ")

    centerCounts, NstepsCounts, stepCounts, ticksToDelay = convertParamsToCounts(
        stageStartInput.value(), 
        stageSpeedInput.value(), 
        stageDistInput.value())

    print("Converted values: center {}; nSteps {}; stepSize {}; ticksToDelay {}".format(
        centerCounts, NstepsCounts, stepCounts, ticksToDelay))

    center_poss=str(centerCounts)
    center_command= list(bytes("U 4 ", 'ascii'))
    for tt in center_poss:
        center_command=(center_command+[ord(tt)])
    center_command=(center_command+ [32,48,10,13])

    step_val=str(stepCounts)
    Nsteps_val=str(NstepsCounts)
        
    forward_command= list(bytes("U 5 ", 'ascii'))
    for tt in step_val:
        forward_command=(forward_command+[ord(tt)])
    forward_command=(forward_command+ [32])
    for tt in Nsteps_val:
        forward_command=(forward_command+[ord(tt)])
    forward_command=(forward_command+ [10,13])

    backward_command= list(bytes("U 7 -", 'ascii'))
    for tt in step_val:
        backward_command=(backward_command+[ord(tt)])
    backward_command=(backward_command+ [32])
    for tt in Nsteps_val:
        backward_command=(backward_command+[ord(tt)])
    backward_command=(backward_command+ [10,13])

    off_command = bytes("k0", 'ascii') + bytes((10, 13)) # stop the stage
    on_command = bytes("k1", 'ascii') + bytes((10, 13))

    pipeEnd.send((off_command, 1))
        
    pipeEnd.send((center_command, 2))
 
    pipeEnd.send((forward_command, 2))

    pipeEnd.send((bytes("U 6 0 {}".format(int(round(DELAY_MICROS/TICK_MICROS))), 'ascii') + bytes((10, 13)), 2))

    pipeEnd.send((backward_command, 2))

    pipeEnd.send((bytes("U 8 0 {}".format(int(round(DELAY_MICROS/TICK_MICROS))), 'ascii') + bytes((10, 13)), 2))

    runnonstop = bytes("V65535 {} 4 8".format(ticksToDelay), 'ascii') + bytes((10, 13)) # run the stage non-stop 
    pipeEnd.send((runnonstop, 2))

    pipeEnd.send((on_command, 1))



def setCenterCommand(pipeEnd):
    '''
    Takes: GUI pipe end.

    Moves the stage to the position indicated in the center input. 
    '''
    poss, Nsteps_val, step_val, ticksToDelay = convertParamsToCounts(stageStartInput.value(), 
                                                                     stageSpeedInput.value(), 
                                                                     stageDistInput.value())
    shift_code= [70]
    for tt in poss:
        shift_code=(shift_code+[ord(tt)])
    shift_code=(shift_code+ [10])
    shift_code=(shift_code+ [13])

    off_command = bytes("k0", 'ascii') + bytes((10, 13)) # stop the stage
    on_command = bytes("k1", 'ascii') + bytes((10, 13))

    pipeEnd.send((off_command, 1))

    runonce = bytes([86, 48, 10, 13]) # run once the stage
    pipeEnd.send((runonce, 1))

    shift_command = bytes(shift_code) 
    pipeEnd.send((shift_command, 2))

    pipeEnd.send((on_command, 1))


#Sends “k1” to turn on the stage. 
def startcommand(pipeEnd):
    stageConsole.appendPlainText("Starting stage:")
    on_command = bytes([107, 49, 10, 13])
    pipeEnd.send((on_command, 1))


#Sends “k0” to turn off the stage. 
def stopcommand(pipeEnd):
    stageConsole.appendPlainText("Stopping stage:")
    off_command = bytes([107, 48, 10, 13]) # stop the stage
    pipeEnd.send((off_command, 1))
    stageStartButton.setEnabled(True)
    stageSetCenterButton.setEnabled(True)


def saveData(filename, fname_stamp=1, *args):
    '''
    Takes: 
    Filename prefix (Fringe or Spectrum.)
    Arbitrary number of arrays, all of the same length.

    Writes input arrays as columns into a CSV file. The filename is 
    formatted with the prefix, the initial of the current windowing 
    function, and the datetime formatting specified in the user.
    '''
    filenameFormat='%Y%b%d_%H%M%S'
    n = len(args[0])
    if not all(len(array) == n for array in args):
        raise ValueError("Not all arrays passed to saveData are of equal length")
    
    #"%Y%b%d_%H%M%S.csv" : format for YMD-HHMMSS.csv
    if fname_stamp==0:
        outFileName = filename+'.csv'
    else:
        try:
            outFileName = datetime.datetime.now().strftime(
                "{}{}.csv".format(filename,
                                    # FFTfilterSelector.currentText()[0],
                                    filenameFormat))
        except Exception:
            print("Invalid string formatting, reverting to default")
            outFileName = datetime.datetime.now().strftime(
                "{}{}%Y%b%d_%H%M%S.csv".format(prefix,
                                               # FFTfilterSelector.currentText()[0],
                                               filenameFormat))
    print("Saving to {}".format(outFileName))
    outFile = open(outFileName, 'w')
    for index in range(n):
        lineToWrite = ''

        #iterate through all arrays passed to function 
        #and concantenate the string for this line.
        for array in args:
            lineToWrite += str(array[index])
            lineToWrite += ','

        #add newline character and write the line to the file.
        lineToWrite += '\n'
        outFile.write(lineToWrite)
    
    #save the file by closing it.    
    outFile.close()

    stageConsole.appendPlainText("Data saved to {}".format(outFileName))



def sendCommand(pipeEnd):
    '''
    Encodes whatever is in the console command input box into binary ASCII 
    and sends it to the Teensy interaction thread. Instructs Teensy to wait 
    one second after writing this command. 
    '''
    command = stageCommandInput.text()

    if command == "d0":
        for letter in string.ascii_uppercase + 'abcdkmn':
            pipeEnd.send(( bytes(letter, 'ascii') + bytes((10, 13)), 1)) #add carriage return

    pipeEnd.send(( bytes(command, 'ascii') + bytes((10, 13)), 1)) #add carriage return

    stageCommandInput.setText("")



MS_WAIT_READ_SETTINGS = config.getint("GUI", "ms_Before_Reading_Settings")

def sendReadInstruction():
    '''
    Essentially acts as a macro for sending the commands “U4”, “U5”, 
    and “V” to the Teensy. By querying these settings from the stage 
    controller, we can read the settings previously commanded to the 
    controller.

    Returns nothing. After 4 seconds, triggers readStageSetting.
    '''
    stageCommandInput.setText("U4")
    stageCommandSubmit.click()
    stageCommandInput.setText("U5")
    stageCommandSubmit.click()
    stageCommandInput.setText("V")
    stageCommandSubmit.click()

    timerRead = QtCore.QTimer.singleShot(MS_WAIT_READ_SETTINGS, readStageSetting)

def readStageSetting():
    '''
    Reads the three most recent lines from the console output, which 
    is expected to be the stage controller’s response to the commands
    “U4”, “U5”, and “V”, which was sent by sendReadInstruction. Using 
    the information from these responses, fills in the command boxes 
    for stage center, half-period, and scan distance. 

    Returns nothing, but sets values in the three boxes above.
    '''

    #parsing replies from controller into integers; we know what 
    #output formatting to expect so the indices are specific
    ErrorLabel.setVisible(False)
    try:
        consoleText = stageConsole.toPlainText().split("\n")
        u4line = consoleText[-3]
        u5line = consoleText[-2]
        vline = consoleText[-1]

        centerCounts = int(u4line.split()[2])
        if centerCounts < 0:
            centerCounts += 2**16
        stepSize = int(u5line.split()[2])
        nSteps = int(u5line.split()[3][:-9])
        ticksToDelay = int(vline.split()[2])

        distance = nSteps * stepSize * COUNTS_TO_DIST
        halfPeriod = nSteps * ticksToDelay * TICK_MICROS / 10**6
        speed = distance / halfPeriod
        
        stageStartInput.setValue(centerCounts * COUNTS_TO_DIST)
        stageDistInput.setValue(distance)
        stageSpeedInput.setValue(speed)

        stageStartButton.setDisabled(True)
        stageSetCenterButton.setDisabled(True)
        stageApplyButton.setEnabled(True)

    except Exception:
        ErrorLabel.setVisible(True)
        stageConsole.appendPlainText("Error while parsing controller response. \
        Power-cycle the stage controller or manually check the output for stage settings.")
        print("Error while parsing controller response. Power-cycle the stage \
        controller or manually check the output for stage settings.")
    

#sets background of graphs to be white and foreground (ie. text) 
#to be black
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

#initialize a top level application. Then, the top level topWidget is 
#placed into the application, and we give a top level layout to it. 
#We add a plot widget to the top level layout.
app = QtGui.QApplication(sys.argv)
topWidget = QtGui.QWidget()
plotWidget = pg.GraphicsLayoutWidget()
layout = QtGui.QGridLayout()
topWidget.setLayout(layout)
layout.addWidget(plotWidget,1,0,1,4) #place in row 4, col 0, spans 1 row, 4 cols
topWidget.setWindowTitle('FTIR Controller')
topWidget.show()

#Plotting control UI group: controls settings for what/how data is 
#displayed and what processing to apply to the data.
plotCtrlBox = QtGui.QGroupBox("Plotting controls")
plotCtrlLayout = QtGui.QGridLayout()
plotCtrlBox.setLayout(plotCtrlLayout)
layout.addWidget(plotCtrlBox, 0,1)
layout.setColumnStretch(1, 3) #this box has a relative width of "1" as well.


#widget to display what mode the Teensy is in (configuration/measurement.)
modeLabel = QtGui.QLabel("Teensy Mode:")
plotCtrlLayout.addWidget(modeLabel, 0,0,1,1)
modeDisp = QtGui.QLineEdit()
modeDisp.setStyleSheet("background-color: green")
modeDisp.setReadOnly(True)
plotCtrlLayout.addWidget(modeDisp,0,1,1,1)

#create a checkbox and link its activation/deactivation to the plot 
#display manager above.
showIntensityCheckbox = QtGui.QCheckBox("&Raw data")
showIntensityCheckbox.setDisabled(False)
plotCtrlLayout.addWidget(showIntensityCheckbox, 1,0)
showIntensityCheckbox.setChecked(True)

#create a checkbox and link its activation/deactivation to the plot 
#display manager above.
showSpecCheckbox = QtGui.QCheckBox("&Spectrum")
showSpecCheckbox.setDisabled(False)
plotCtrlLayout.addWidget(showSpecCheckbox, 2,0)
showSpecCheckbox.setChecked(False)

wvlimsCheckbox = QtGui.QCheckBox("&lims:")
plotCtrlLayout.addWidget(wvlimsCheckbox, 2,1)
wvlimsCheckbox.setChecked(True)

wv1Edit = QtGui.QSpinBox()
wv1Edit.setRange(0,5000)
plotCtrlLayout.addWidget(wv1Edit,2,2,1,1)
wv1Edit.setValue(1000)

wv2Edit = QtGui.QSpinBox()
wv2Edit.setRange(0,5000)
plotCtrlLayout.addWidget(wv2Edit,2,3,1,1)
wv2Edit.setValue(2000)

#create a checkbox and link its activation/deactivation to the plot 
#display manager above.
showACorrCheckbox = QtGui.QCheckBox("&Auto-corr.")
showACorrCheckbox.setDisabled(False)
plotCtrlLayout.addWidget(showACorrCheckbox, 3,0,1,2)
showACorrCheckbox.setChecked(False)

fslimsCheckbox = QtGui.QCheckBox("&lims:")
plotCtrlLayout.addWidget(fslimsCheckbox, 3,1)
fslimsCheckbox.setChecked(True)

fs1Edit = QtGui.QSpinBox()
fs1Edit.setRange(-5000,0)
plotCtrlLayout.addWidget(fs1Edit,3,2)
fs1Edit.setValue(-300)

fs2Edit = QtGui.QSpinBox()
fs2Edit.setRange(0,5000)
plotCtrlLayout.addWidget(fs2Edit,3,3)
fs2Edit.setValue(300)

# ACfilterLabel = QtGui.QLabel("AC filter")
# dataCollLayout.addWidget(ACfilterLabel, 1,0)
ACfilterLabel = QtGui.QLabel("ACorr filter:")
plotCtrlLayout.addWidget(ACfilterLabel, 3,5)

ACfilterEdit = QtGui.QSpinBox()
ACfilterEdit.setRange(0,1000)
plotCtrlLayout.addWidget(ACfilterEdit,3,6)
ACfilterEdit.setValue(100)

#the update function checks this box near the beginning and returns 
#early if it is checked, so that nothing is updated to the GUI.
freezeCheckbox = QtGui.QCheckBox("&Freeze")
plotCtrlLayout.addWidget(freezeCheckbox, 1,2)

def CWmodeManager():
    '''
    Changes the labels of one box in the status group depending on if 
    CW mode is activated. Also shows/hides right axis label (if CW 
    mode is on, p2 and p2a display the same units, which is intensity.)
    '''
    if CWmodeCheckbox.isChecked():
        dataPtsLabel.setText("Signal zeros / ref")
        p2.getAxis("right").showLabel(show = False)
    else:
        dataPtsLabel.setText("Data points to FT")
        p2.getAxis("right").showLabel(show = True)

CWmodeCheckbox = QtGui.QCheckBox("&CW mode")
plotCtrlLayout.addWidget(CWmodeCheckbox, 1,1)
CWmodeCheckbox.stateChanged.connect(CWmodeManager)


def showIntensityManager():
    '''
    Creates global variables (pyqtgraph PlotDataItem objects) used for 
    plotting points. When “show intensity” checkbox is checked, the 
    plots are re-created. When the box is unchecked, the plots are 
    destroyed.
    '''
    global p2, p2a, plotOnePeriodIntensity, plotOnePeriodPos
    if showIntensityCheckbox.isChecked():
        p2 = plotWidget.addPlot(row = 0, col = 0, 
                                title = 'Intensity / Position Data')
        plotOnePeriodIntensity = p2.plot(
            autoDownsample = True, pen = COLOR_INTENSITY, clipToView = True) #creates plotDataItem
        plotOnePeriodIntensity.setDownsampling(auto = True, method = "peak")

        p2.showAxis("left")
        p2.setLabel('left', 'Signal intensity')
        p2.showAxis("bottom")
        if showACorrCheckbox.isChecked():
            p2.setLabel('bottom', 'delay, fs')
        else:
            p2.setLabel('bottom', 'Index')
        p2.showAxis("right")
        p2.setLabel('right', 'Additional path length', units = 'm')

        #creates a secondary plot that is overlaid on top of p2
        #this is used to display position.
        p2a = pg.ViewBox()
        p2.scene().addItem(p2a)
        p2.getAxis("right").linkToView(p2a)
        p2a.setXLink(p2)

        #updates p2a as p2 is resized so that they are always 
        #the same size and overlaid.
        updateP2View()
        p2.vb.sigResized.connect(updateP2View)

        plotOnePeriodPos = pg.PlotDataItem(
            autoDownsample = True, pen = COLOR_POSITION, clipToView = True) #creates plotDataItem
        plotOnePeriodPos.setDownsampling(auto = True, method = "peak")
        p2a.addItem(plotOnePeriodPos)

    else:
        #if show intensity checkbox is not checked: remove the plots.
        p2.scene().removeItem(p2a)
        plotWidget.removeItem(p2)


def updateP2View():
    '''
    updates p2a as p2 is resized so that they are always the same size 
    and overlaid. This is done by reading p2's display dimensions and 
    setting p2a to those dimensions.
    '''
    p2a.setGeometry(p2.vb.sceneBoundingRect())
    p2a.linkedViewChanged(p2.vb, p2a.XAxis)


showIntensityCheckbox.stateChanged.connect(showIntensityManager)
showIntensityManager()

def showSpectrumManager():
    '''
    Creates global variables (pyqtgraph PlotDataItem objects) used for 
    plotting points. When “show spectrum” checkbox is checked, the 
    plots are re-created. When the box is unchecked, the plots are 
    destroyed.
    '''
    global p3, plotSpectrum, plotSpectrumFit
    if showSpecCheckbox.isChecked():
        #plot object for displaying computed spectrum
        p3 = plotWidget.addPlot(row = 0, col = 1) #creates plotItem and places it in next available slot
        p3.setTitle("Spectrum", size = '{}pt'.format(FONT_SIZE))
        p3.showGrid(x=True, y=True)
        plotSpectrum = p3.plot(autoDownsample = True, 
                               pen = pg.mkPen('b', width = 4),
                               clipToView = True) #creates plotDataItem
        plotSpectrum.setDownsampling(auto = True, method = "subsample")#"subsample")
        # QtCore.Qt.SolidLine / QtCore.Qt.DotLine / QtCore.Qt.DashLine
        # width=5
        plotSpectrumFit= p3.plot(autoDownsample = True, 
                               pen = pg.mkPen('r', width = 2, style=QtCore.Qt.DashLine),
                               clipToView = True) #creates plotDataItem
        plotSpectrumFit.setDownsampling(auto = True, method = "subsample")#"subsample")
        p3.showAxis("left")
        p3.setLabel('left', 'Amplitude', **labelStyle)
        p3.getAxis("left").tickFont = axisFont
        p3.getAxis("left").setStyle(tickTextOffset = 10, tickTextHeight = 20, 
                                    tickTextWidth = 100)


        p3.showAxis("bottom")
        p3.getAxis("bottom").tickFont = axisFont
        p3.getAxis("bottom").setStyle(tickTextOffset = 10, tickTextHeight = 20)
        p3.setLabel('bottom', 'Wavelength (nm)', **labelStyle)
    else:
        #if show intensity checkbox is not checked: remove the plots.
        try:
            plotWidget.removeItem(p3)
        except:
            # no spectrum plot exist yet
            pass

showSpectrumManager()

def showACorrManager():
    '''
    Creates global variables (pyqtgraph PlotDataItem objects) used for 
    plotting points. When “show spectrum” checkbox is checked, the 
    plots are re-created. When the box is unchecked, the plots are 
    destroyed.
    '''
    global p4, plotACorr, plotACorrFit
    if showACorrCheckbox.isChecked():
        #plot object for displaying computed spectrum
        p4 = plotWidget.addPlot(row = 0, col = 2) #creates plotItem and places it in next available slot
        p4.setTitle("Auto-correlation", size = '{}pt'.format(FONT_SIZE))
        p4.showGrid(x=True, y=True)
        plotACorr = p4.plot(autoDownsample = True, 
                               pen = pg.mkPen('b', width = 4), 
                               clipToView = True) #creates plotDataItem
        plotACorr.setDownsampling(auto = True, method = "subsample")#"subsample")
        # QtCore.Qt.SolidLine / QtCore.Qt.DotLine / QtCore.Qt.DashLine
        # width=5
        plotACorrFit= p4.plot(autoDownsample = True, 
                               pen = pg.mkPen('r', width = 2, style=QtCore.Qt.DashLine),
                               clipToView = True) #creates plotDataItem
        plotACorrFit.setDownsampling(auto = True, method = "subsample")#"subsample")
        p4.showAxis("left")
        p4.setLabel('left', 'Amplitude', **labelStyle)
        p4.getAxis("left").tickFont = axisFont
        p4.getAxis("left").setStyle(tickTextOffset = 10, tickTextHeight = 20, 
                                    tickTextWidth = 100)


        p4.showAxis("bottom")
        p4.getAxis("bottom").tickFont = axisFont
        p4.getAxis("bottom").setStyle(tickTextOffset = 10, tickTextHeight = 20)
        p4.setLabel('bottom', 'Delay (fs)', **labelStyle)
    else:
        #if show intensity checkbox is not checked: remove the plots.
        try:
            plotWidget.removeItem(p4)
        except:
            # no auto-correlation plot exist yet
            pass
showACorrManager()


def plotApplyXRange(plotObject, xMin, xMax):
    '''
    Sets plotObject's x range to xMin to xMax. Unfortunately doesn't 
    seem to always work, but zooming can always be done manually.
    '''
    plotObject.setXRange(xMin, xMax)

def logPlotManager():
    '''
    If the log plot checkbox is checked, enable the log plot and 
    disable the left axis label on the spectrum plot. Otherwise, 
    do the opposite.
    Attempts to keep the x-axis the same, but this doesn't seem 
    to work.
    '''
    currentXBounds = p3.viewRange()[0]
    if logSpecCheckbox.isChecked():
        p3.getAxis("left").showLabel(show = False)
        p3.setLogMode(x = False, y = True)
    else:
        p3.getAxis("left").showLabel(show = True)
        p3.setLogMode(x = False, y = False)
    plotApplyXRange(p3, currentXBounds[0], currentXBounds[1])

logSpecCheckbox = QtGui.QCheckBox("&Log scale")
plotCtrlLayout.addWidget(logSpecCheckbox, 2,4)
logSpecCheckbox.stateChanged.connect(logPlotManager)

logACorrCheckbox = QtGui.QCheckBox("&Log scale")
plotCtrlLayout.addWidget(logACorrCheckbox, 3,4)
logACorrCheckbox.stateChanged.connect(logPlotManager)

GAUSS_MIN_STDEV = config.getfloat("DataProcessing", "Gauss_Window_Min_StDev")
GAUSS_MAX_STDEV = config.getfloat("DataProcessing", "Gauss_Window_Max_StDev")
GAUSS_DEF_STDEV = config.getfloat("DataProcessing", "Gauss_Window_Def_StDev")
GAUSS_STDEV_STEP = config.getfloat("DataProcessing", "Gauss_Window_StDev_Step")
TUKEY_MIN_ALPHA = config.getfloat("DataProcessing", "Tukey_Window_Min_Alpha")
TUKEY_MAX_ALPHA = config.getfloat("DataProcessing", "Tukey_Window_Max_Alpha")
TUKEY_DEF_ALPHA = config.getfloat("DataProcessing", "Tukey_Window_Def_Alpha")
TUKEY_ALPHA_STEP = config.getfloat("DataProcessing", "Tukey_Window_Alpha_Step")
MARGIN_MIN = config.getfloat("DataProcessing", "Margin_Min")
MARGIN_MAX = config.getfloat("DataProcessing", "Margin_Max")
MARGIN_DEFAULT = config.getfloat("DataProcessing", "Margin_Default_Pct")
MARGIN_STEP = config.getfloat("DataProcessing", "Margin_Step")

def FFTfilterManager():
    '''
    Responds to changes in the FFT windowing function dropdown 
    selector. Sets ranges/enables the parameter input box as 
    appropriate.
    '''
    # if FFTfilterSelector.currentText() == 'No FFT window':
    #     FFTparamInput.setEnabled(False)
    # elif FFTfilterSelector.currentText() == 'Gaussian window with SD':
    #     FFTparamInput.setEnabled(True)
    #     FFTparamInput.setRange(GAUSS_MIN_STDEV,GAUSS_MAX_STDEV)
    #     FFTparamInput.setValue(GAUSS_DEF_STDEV)
    #     FFTparamInput.setSingleStep(GAUSS_STDEV_STEP)
    # elif FFTfilterSelector.currentText() == 'Tukey window with alpha':
    FFTparamInput.setEnabled(True)
    FFTparamInput.setRange(TUKEY_MIN_ALPHA,TUKEY_MAX_ALPHA)
    FFTparamInput.setValue(TUKEY_DEF_ALPHA)
    FFTparamInput.setSingleStep(TUKEY_ALPHA_STEP) 

#dropdown box with options for what windowing function to apply to 
#signal before passing to FFT.
FFTfilterSelector = QtGui.QComboBox()
FFTfilterSelector.addItems(['No FFT window', 
                            'Gaussian window with SD', 
                            'Tukey window with alpha'])
FFTfilterSelector.setCurrentIndex(2) #default: choose Tukey window

#spinbox so stdev/alpha can be specified for Gaussian/Tukey windows.
FFTparamInput = QtGui.QDoubleSpinBox()

#we can only call FFTfilterManager after FFTparamInput is defined
FFTfilterSelector.currentIndexChanged.connect(FFTfilterManager)
FFTfilterManager()




#spinbox to input margin to clip data by. 
marginLabel = QtGui.QLabel("% to clip data")
marginInput = QtGui.QDoubleSpinBox()
marginInput.setRange(0,1)
marginInput.setValue(MARGIN_DEFAULT)
marginInput.setSingleStep(0.05)
# plotCtrlLayout.addWidget(marginInput,4,1)



#input and label for setting the formatting of the filenames used.
saveDataBtn = QtGui.QPushButton("Save (time-stamp if empty)")
saveDataBtn.setCheckable(True)
saveDataBtn.setChecked(False)
plotCtrlLayout.addWidget(saveDataBtn, 0,5,1,2)
#it seems that labels are by default greedy with their space, so this 
#makes it more in line with the other UI elements.

filenameInput = QtGui.QLineEdit()
filenameInput.setText("")#%Y%b%d_%H%M%S")
plotCtrlLayout.addWidget(filenameInput, 1,5,1,2)




#Teensy control UI group: contains controls for the stage.
stageCtrlBox = QtGui.QGroupBox("Stage controls")
stageCtrlLayout = QtGui.QGridLayout()
stageCtrlBox.setLayout(stageCtrlLayout)
layout.addWidget(stageCtrlBox, 0,3)
layout.setColumnStretch(3,3) #this group box has a relative width of "3"
                             #since it contains a console widget.

#buttons to start/stop the stage
stageStartButton = QtGui.QPushButton("St&art")
stageStartButton.setDisabled(True)
stageCtrlLayout.addWidget(stageStartButton, 0,0)
stageStartButton.clicked.connect(lambda: startcommand(pipe_GUIEnd))

stageStopButton = QtGui.QPushButton("St&op")
stageCtrlLayout.addWidget(stageStopButton, 1,0)
stageStopButton.clicked.connect(lambda: stopcommand(pipe_GUIEnd))

ErrorLabel = QtGui.QLabel(" Error! \n\n Cycle stage power")
ErrorLabel.setStyleSheet("background-color: red")
ErrorLabel.setVisible(False)
stageCtrlLayout.addWidget(ErrorLabel, 0, 3, 3, 2)

OverloadLabel_1 = QtGui.QLabel("  Signal <0 \n \n  Adjust Teensy pots!")
OverloadLabel_1.setStyleSheet("background-color: blue")
OverloadLabel_1.setVisible(False)
plotCtrlLayout.addWidget(OverloadLabel_1, 0, 3, 3, 2)

OverloadLabel_2 = QtGui.QLabel("  Signal overload! \n\n  Adjust Teensy pots!")
OverloadLabel_2.setStyleSheet("background-color: red")
OverloadLabel_2.setVisible(False)
plotCtrlLayout.addWidget(OverloadLabel_2, 0, 3, 3, 2)

OverloadLabel_3 = QtGui.QLabel("  Reference <0 \n\n  Adjust Teensy pots!")
OverloadLabel_3.setStyleSheet("background-color: rgb(127,255,0)")
OverloadLabel_3.setVisible(False)
plotCtrlLayout.addWidget(OverloadLabel_3, 0, 3, 3, 2)

OverloadLabel_4 = QtGui.QLabel("  Reference overload! \n\n  Adjust Teensy pots!")
OverloadLabel_4.setStyleSheet("background-color: rgb(255,0,255)")
OverloadLabel_4.setVisible(False)
plotCtrlLayout.addWidget(OverloadLabel_4, 0, 3, 3, 2)


#labels and spinboxes to control parameters for stage movement
DEFAULT_STAGE_START = config.getfloat("DataAcquisition", "Stage_Default_Start_Pos")
DEFAULT_STAGE_DIST = config.getfloat("DataAcquisition", "Stage_Default_Scan_Dist")
DEFAULT_STAGE_SPEED = config.getfloat("DataAcquisition", "Stage_Default_Speed")

stageStartLabel = QtGui.QLabel("Starting pos (mm)")
stageCtrlLayout.addWidget(stageStartLabel, 0,1)
stageStartInput = QtGui.QDoubleSpinBox()
stageStartInput.setRange(0, 10)
stageStartInput.setSingleStep(0.1)
stageStartInput.setValue(DEFAULT_STAGE_START)
stageCtrlLayout.addWidget(stageStartInput,0,2)

stageDistLabel = QtGui.QLabel("Distance to scan (mm)")
stageCtrlLayout.addWidget(stageDistLabel, 1,1)
stageDistInput = QtGui.QDoubleSpinBox()
stageDistInput.setRange(0,5)
stageDistInput.setSingleStep(0.1)
stageDistInput.setValue(DEFAULT_STAGE_DIST)
stageCtrlLayout.addWidget(stageDistInput,1,2)

stageSpeedLabel = QtGui.QLabel("Stage speed (mm/s)")
stageCtrlLayout.addWidget(stageSpeedLabel, 2,1)
stageSpeedInput = QtGui.QDoubleSpinBox()
stageSpeedInput.setRange(0.1, 10)
stageSpeedInput.setSingleStep(0.1)
stageSpeedInput.setValue(DEFAULT_STAGE_SPEED)
stageCtrlLayout.addWidget(stageSpeedInput,2,2)

#console for stage communications
stageConsole = QtGui.QPlainTextEdit("Communications from stage controller \
will be displayed below.\nPress Alt to see keyboard shortcuts.")
stageConsole.setReadOnly(True)
stageCtrlLayout.addWidget(stageConsole, 0, 3, 3, 2)

#buttons to read stage settings, set center setting only 
#(stationary stage), or start scanning the stage
stageReadSettingsButton = QtGui.QPushButton("R&ead settings")
stageCtrlLayout.addWidget(stageReadSettingsButton, 3,0)
stageReadSettingsButton.clicked.connect(sendReadInstruction)

stageSetCenterButton = QtGui.QPushButton("Move t0 to:")
stageSetCenterButton.setDisabled(True)
stageCtrlLayout.addWidget(stageSetCenterButton, 3,1)
stageSetCenterButton.clicked.connect(lambda: setCenterCommand( pipe_GUIEnd ) )

stageApplyButton = QtGui.QPushButton("&Scan stage")
stageApplyButton.setDisabled(True)
stageCtrlLayout.addWidget(stageApplyButton, 3,2)
stageApplyButton.clicked.connect(lambda: scancommand( pipe_GUIEnd ) )

#line edit box for sending custom commands to stage
stageCommandInput = QtGui.QLineEdit()
stageCtrlLayout.addWidget(stageCommandInput, 3, 3)
stageCommandInput.returnPressed.connect(lambda: sendCommand( pipe_GUIEnd ) )

stageCommandSubmit = QtGui.QPushButton("Send")
stageCtrlLayout.addWidget(stageCommandSubmit, 3, 4)
stageCommandSubmit.clicked.connect(lambda: sendCommand( pipe_GUIEnd ) )

showSpecCheckbox.stateChanged.connect(showSpectrumManager)
showACorrCheckbox.stateChanged.connect(showACorrManager)

print('done initializing graphs')



def updateWithMode():
    '''
    enable stage control box if Teensy is in configuration mode only
    '''
    mode = modeDisp.text()
    if mode == 'Measurement':
        stageCtrlBox.setEnabled(False)
        modeDisp.setStyleSheet("background-color: rgb(250,128,114)")
        stageCtrlBox.setTitle("Stage controls - Teensy must be in Configuration mode")
    elif mode == 'Configuration':
        stageCtrlBox.setEnabled(True)
        modeDisp.setStyleSheet("background-color: rgb(0,255,0)")
        stageCtrlBox.setTitle("Stage controls")

modeDisp.textChanged.connect(updateWithMode)



def update():
    '''
    main update loop that is executed as fast as possible.
    See documentation for more details.
    '''
    try:
        #read data from the pipe.
        yIntensityQueue, refIntensityQueue, timeTaken = readPipe(
            pipe_GUIEnd, modeDisp, stageConsole) # progressBar, 
        # print('reading PIPE!!!')
        if refIntensityQueue:
            level_Sig1=np.round(min(yIntensityQueue)/655.35)
            level_Sig2=np.round(max(yIntensityQueue)/655.35)
            level_Ref1=np.round(min(refIntensityQueue)/655.35)
            level_Ref2=np.round(max(refIntensityQueue)/655.35)
            print(level_Sig1,level_Sig2,level_Ref1,level_Ref2)
            if level_Sig1==0:
                OverloadLabel_1.setVisible(True)
            elif level_Sig2==100:
                OverloadLabel_2.setVisible(True)
            elif level_Ref1==0:
                OverloadLabel_3.setVisible(True)
            elif level_Ref2==100:
                OverloadLabel_4.setVisible(True)
            else:
                OverloadLabel_1.setVisible(False)
                OverloadLabel_2.setVisible(False)
                OverloadLabel_3.setVisible(False)
                OverloadLabel_4.setVisible(False)

                #if readPipe returns something that is not a batch of data or 
                #the freeze checkbox is checked, we just have to check if data 
                #should be saved, and return quickly so the program remains 
                #responsive between updates.
                if timeTaken == 0 or freezeCheckbox.isChecked():

                    #we can save these variables which were retained from the 
                    #last time update() was executed.
                    # if saveDataBtn.isChecked():
                    #     fname_stamp="%Y%b%d_%H%M%S"
                    #     if showACorrCheckbox.isChecked():
                    #       saveData('ACorr_', fname_stamp, 
                    #                filenameInput.text(), 
                    #                plotOnePeriodIntensity.xData, 
                    #                plotOnePeriodPos.yData, 
                    #                plotOnePeriodIntensity.yData)
                    #       saveDataBtn.setChecked(False)
                    #     else:
                    #       saveData('Fringes_', fname_stamp, 
                    #                filenameInput.text(), 
                    #                plotOnePeriodIntensity.xData, 
                    #                plotOnePeriodPos.yData, 
                    #                plotOnePeriodIntensity.yData)
                    #       saveDataBtn.setChecked(False)

                    # if saveSpectrumCheckbox.isChecked():
                    #     saveData('Spectrum_',  fname_stamp,
                    #              filenameInput.text(), 
                    #              plotSpectrum.xData, 
                    #              plotSpectrum.yData)
                    #     saveSpectrumCheckbox.setChecked(False)
                    
                    return

                #set the timer display for information.
                # timeDisp.setText('{} s'.format(round(timeTaken, 3)))

                #record the original length of the queue before processing.
                unclippedLength = len(yIntensityQueue)

                #clip the queues according to specified margin.
                yIntensityQueue = clipMarginOnly(yIntensityQueue, marginInput.value())
                refIntensityQueue = clipMarginOnly(refIntensityQueue, marginInput.value())

                #remove background level from intensity queue.
                yIntensityQueue = removeBackground(yIntensityQueue, 0.05)

                #convert fringes of reference laser to distance measurement.
                yPosQueue, posZeros = intensityToPos(refIntensityQueue)

                #if not in CW mode:
                if not CWmodeCheckbox.isChecked():

                    #we assume the signal is of a mode-locked laser and find 
                    #the zero-path-difference position.
                    centerIndex = int(round(getCenter(yIntensityQueue)))

                    #given the center position, we clip the queues symmetric 
                    #around that center.
                    yIntensityQueue = clipArraySymmetric(yIntensityQueue, centerIndex)
                    yPosQueue = clipArraySymmetric(yPosQueue, centerIndex)
                
                #otherwise, we are in CW mode, so we don't clip the intensity 
                #data. Instead we count zeros for the reference intensity and 
                #measurement intensity signal.    
                else:
                    
                    #remove background level from the reference laser as well.
                    refIntensityQueue = removeBackground(refIntensityQueue, 0.05)
                    sigIntensityZeros = getZeros(yIntensityQueue)

                #in any case, we want to plot the data in a way that the 
                #center is fixed in the plot.
                centerX = np.arange(-len(yIntensityQueue)//2, len(yIntensityQueue)//2)

                #apply a windowing function to the intensity queue.
                yIntensityQueue = applyWindow(yIntensityQueue)

                #plot intensity/position only if the plots are showing.
                if showIntensityCheckbox.isChecked():
                    # if showACorrCheckbox.isChecked():
                        #convert positions into corresponding light-times.
                    if CWmodeCheckbox.isChecked():
                        plotOnePeriodIntensity.setData(centerX, yIntensityQueue)
                        plotOnePeriodPos.setData(centerX, refIntensityQueue)
                    else:
                        lightTimeSeries = convertPosToLightTime(yPosQueue)
                        plotOnePeriodIntensity.setData(centerX, yIntensityQueue)
                        plotOnePeriodPos.setData(centerX, lightTimeSeries)
                    p2.setTitle("Signal = {:1.0f}...{:1.0f}, Ref={:1.0f}...{:1.0f}".format(
                                    level_Sig1, level_Sig2, 
                                    level_Ref1, level_Ref2 ) )

                # ##FFT starts here
                if showSpecCheckbox.isChecked():
                    #convert positions into corresponding light-times.
                    lightTimeSeries = convertPosToLightTime(yPosQueue)

                    #convert into wavelength domain via FFT.
                    wavelength, amplitude, fitData, FWHMwv, FWHMt_gaus = FFT_to_spectrum(lightTimeSeries, yIntensityQueue)

                    #plot the spectrum.
                    plotSpectrum.setData(wavelength, amplitude)
                    plotSpectrumFit.setData(wavelength, fitData)

                    #get peak and FWHM and display it via title of spectrum plot.
                    peakWavelength = getMaxWavelength(wavelength, amplitude)
                    p3.setTitle("Max = {} nm, FWHM = {} nm, pulse ~ {} fs".format(
                        round(peakWavelength, 1), round(FWHMwv, 1), round(FWHMt_gaus, 1))
                                )
                    p3.setLabel('bottom', 'Wavelength (nm)', **labelStyle)
                    if wvlimsCheckbox.isChecked():
                        p3.setXRange(wv1Edit.value(),wv2Edit.value())
                    else:
                        p3.setXRange(DEFAULT_XMIN,DEFAULT_XMAX)

                if showACorrCheckbox.isChecked():

                    # perform FFT and calculate Auto-correlation
                    tax, acorr, fitData, FWHMt = acorr_analysis(lightTimeSeries, yIntensityQueue)

                    #plot the spectrum.
                    plotACorr.setData(tax, acorr)
                    plotACorrFit.setData(tax, fitData)
                    # plotACorr.setData(tax[AC_peaks_array[:,0].astype(int)], AC_peaks_array[:,1])
                    # plotACorrFit.setData(tax[AC_peaks_array[:,0].astype(int)], fitPeaks)

                    p4.setTitle("FWHM = {} fs, pulse ~ {} fs".format( np.round(FWHMt, 1), np.round(FWHMt/np.sqrt(2), 1)))
                    # p4.setTitle("FWHM = {} fs, pulse ~ {} fs".format( np.round(FWHM_peaks, 1), np.round(FWHM_peaks/np.sqrt(2), 1)))
                    p4.setLabel('bottom', 'time delay (fs)', **labelStyle)
                    if fslimsCheckbox.isChecked():
                        p4.setXRange(fs1Edit.value(),fs2Edit.value())
                    else:
                        p4.setXRange(DEFAULT_XMIN,DEFAULT_XMAX)
                
                if filenameInput.text():
                    filename=filenameInput.text()
                    fname_stamp=0
                else:
                    fname_stamp=1
                    filename=''
                if saveDataBtn.isChecked():
                    img = PIL.ImageGrab.grab()
                    if filename=='':
                        img.save(datetime.datetime.now().strftime("{}{}.jpg".format('%Y%b%d_%H%M%S')))
                    else:
                        img.save(str(filename+".jpg"))
                    filename1=('Raw_'+filename)#, fname_stamp, filenameInput.text())   
                    print('filename:'+filename1)         
                    saveData(filename1, fname_stamp,
                             plotOnePeriodIntensity.xData, 
                             plotOnePeriodPos.yData, 
                             plotOnePeriodIntensity.yData)
                    saveDataBtn.setChecked(False) 
                    if showSpecCheckbox.isChecked():
                        filename2=('Spectrum_'+filename)#+fname_stamp+'_'filenameInput.text())
                        print('filename:'+filename2)
                        saveData(filename2,fname_stamp, 
                                 plotSpectrum.xData, 
                                 plotSpectrum.yData,)
                    if showACorrCheckbox.isChecked():
                        filename3=('ACorr_'+filename)#, fname_stamp, filenameInput.text())
                        print('filename:'+filename3)
                        saveData(filename3,fname_stamp, 
                                 plotACorr.xData, 
                                 plotACorr.yData,)
                return

    #except block to catch errors during the above try block.
    except Exception as errorMessage:
        print("Error during general update was caught:")
        print(errorMessage)
        traceback.print_tb(errorMessage.__traceback__)
        stageConsole.appendPlainText("Error during general update was caught. \
        Check the console output for details.")
        return

    # ##FFT starts here
    # try:
    #     if refIntensityQueue:

    # #catch errors in FFT try block above.
    # except Exception as errorMessage:
    #     print("Error during FFT was caught:")
    #     print(errorMessage)
    #     traceback.print_tb(errorMessage.__traceback__)
    #     stageConsole.appendPlainText("Error during FFT was caught. Check the console output for details.")

    #     return 

#create a timer to run the update function as fast as possible.
timer1 = QtCore.QTimer()
timer1.timeout.connect(update)
timer1.start(0)

#only start the above if program is running on its own 
#(ie. not imported.)
if __name__ == "__main__":

    #improves Windows handling of subprocesses?
    mp.freeze_support()

    #create a pipe object to communicate between two processes.
    pipe_TeensyEnd, pipe_GUIEnd = mp.Pipe()


    #start a new process for communicating to Teensy.
    teensyProcess = mp.Process(target = InteractWithTeensy, args = (pipe_TeensyEnd,), daemon = True)
    teensyProcess.start()
    time.sleep(2) #allow serial time to start

    #execute the QT app.
    app.exec_()
 