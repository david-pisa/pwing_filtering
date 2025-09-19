import numpy as np
import scipy
import scipy.fft as scifft

def removepeakvalues(vlfdata, fs=40000.):
    debug = False
    showrempeakswitching=False
    removepeakvalues=True
    '''   If==1, resets data to zero when a sudden transient occur 
                    %   and the peak value in the signal AMPLITUDE exceeds
                    %   the given "rempeakvalue" .
    '''                  
    #if not rempeakvalue:
    rempeakvalue=.15
    '''
    % The value depends on the data and on the reason you use this filter.
    %                   % Essentially this filter sets the signal value to zero when
    %                   % the rms value of the signal exceeds the "rempeakvalue" as measured 
    %                   % from the prevailing bottom level signal value. In reality 
    %                   % this very difficult filter is also very complicated.
    %
    %                   % The "rempeakvalue"is either fixed for the analysis or
    %                   % changes with conditions forming ADAPTIVE FILTER. The
    %                   % given value is then the STARTING VALUE in that filter
    '''
                 
    # %%%%% Defining frequency band where sferic signal level is measured %%%           
                    

    rempvmaxfreq=15800 # defines the maximum frequency used in sferics definition control computation
    rempvminfreq=800  # defines the minimum frequency used in sferics definition control computation
    '''
    % The data used in sferic removal bandpass
    % filtered measured signal.

    % The band defined above must be free from strong disturbing signals 
    % e.g. nonfiltered VLF transmitters. 
    % Sferics ARE MEASURED in the given band,but the created DATA RESET (data samples to zeros) 
    % covers the WHOLE band.
    % IF YOU MEASURE HIGH FREQUNCY HISS e.g.IN THE BAND
    % 28.5-32 kHz (rather clean band often), THEN YOU HAVE TO SET THE FREQUENCY BAND
    % FOR SFERIC REMOVAL TO THE SAME FREQUENCY BAND.
    ''' 
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%  Sferic filter commands for automatic control   (adaptive filter)                
                    
    TrackSfericLevel=True    # if==1, tracks data loss level slowly and changes rempeakvalue in order 
                          # to keep Sfericlosspercent near the given value.
                        
    Automaticstart=True    #   if ==1, then the program tries to develope from the given "rempeakvalue" 
                        #   a better estimate at the start of the analysis in order to get faster the
                        #   final tracking value
                        
    saverempeakvalue=True    #   if==1, the removepeakvalue is saved in every dump to _spec file under the name "rempeakvalue"               
                       
    Sfericlosspercent=30  # Program keeps the given value (data loss in percent of time) by adjusting slowly  
                      # the "rempeakvalue" 
    Rempvlimit=.01   # lowest accepted rempeakvalue, which servo can give,limit needed for stability reasons
    
    corrstep = 0.0001
    
    trackcounterlimit = 500
 
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%
    datalen = len(vlfdata)
    #exptwo = np.ceil(np.log(datalen) / np.log(2.))
    #FFTlen=int(2**exptwo) 
    '''
    % Defining time divisions used in signal detection and background (BGR) finding 
    % rounding of switching edges in sferic filter.
    % "Backround" level is the minimum level of signal
    % occurring at short time periods between sferics.
    '''                    
    detectingdivision=int(datalen / fs / 0.0025) # = 2.5 ms; for KAN = 2000 => 2.5 ms
    if debug:
        print(f'Detecting division={detectingdivision} intervals ({(datalen / detectingdivision) / fs * 1000} ms)')
    ''' Divides "blockduration" long datablock to "detectingdivision" pieces 
    %                      % and a single "rms" amplitude level estimate is defined for every piece.                                             
    '''
    BGRdivision= int(datalen / fs / 0.05)# = 50 ms; for KAN = 100 => 51 ms      
    if debug:
        print(f'Detecting BGRdivision={BGRdivision} intervals ({(datalen / BGRdivision) / fs * 1000} ms)')
    '''
    % divides "blockduration" long datablock to BGRdivision pieces and defines
    % signal minimum level estimate for every piece using values obtained in
    % previous "detectingdivision" computations.
    '''                     
                                             
    roundingbeeta=10    #parameter controls smoothening of edges in the signal on-off/off-on switching
                        #% in "removepeakvalues" filter, suitable values are e.g. 10-50. 
                        #% Higher values make edges smoother. Test performance with the data.

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    # Minimize the effects of sferic removal in spectral power
                    #
    correctPRloss=True     # if ==1 the powerloss in apectra due to peak amplitude (sferics) removal is more or less corrected
                   # and very high loss spectra are setted to zero. Resetting is controlled by
                   #"PRlossresetlimit" parameter given below.

    lossestim=True      # if==1 causes "mean",if not 1 causes "median" computation, when compensating power losses
                    # factor to "about" no losses level is estimated. KEEP value 1, it is a safe selection
                    
    PRlossresetlimit=0.50 # Active if "correctPRloss=1". The whole spectrum is resetted if THE REMAINING POWER
                    # IS LESS THAN THE GIVEN "PRlossresetlimit" (no loss=1, total loss = 0);
                    # The computation takes into account the window used in the spectral computations 
                    # and also the necessary tapering of the data needed in sferic removal reset boundaries. 
                    # The resetted spectra are do not affect the values in time averages formed
                    # later during data compression. Those are correctly normalized.
                    # If you dont like gaps put very low value (eg.0.01) to
                    # "PRlossresetlimit", but then badly erronous spectra can enter into your integrated estimates 
    
    #datalen = len(vlfdata)
    #exptwo = np.ceil(np.log(datalen) / np.log(2.))
    #3FFTlen=int(2**exptwo) #   % note: for one minute blockduration the exponent is 23, 
    # which is interesting fft size
    
    Nvlfdata=vlfdata#np.zeros((datalen,), dtype='complex')
    #padlen=int((FFTlen-datalen)/2)
    #Nvlfdata[padlen:padlen+datalen]=vlfdata # padding the data                   
    PLfft = scipy.fft.fft(Nvlfdata)
    Freq = scipy.fft.fftfreq(len(Nvlfdata), 1./fs)
    dF = Freq[1]             
    #PLdeltafreq = 0.1490                    
    #
    #Removing the high signal values of sferics 
    #
    fmax=int(rempvmaxfreq/dF)
    fmin=int(rempvminfreq/dF)
    valdiff = fmax-fmin
    if debug:
        print(f'Check sferics at frequencies: {Freq[fmin]}-{Freq[fmax]} Hz')
    fdiff=fmax-fmin
    PLfftwindow=scipy.signal.windows.kaiser(valdiff,10)
    #PLcenter=len(PLfft)//2
    RPfft=np.zeros((len(PLfft),), dtype='complex')
    RPfft[fmin:fmax] = PLfft[fmin:fmax] * PLfftwindow
    RPfft[-fmax:-fmin] = PLfft[-fmax:-fmin] * PLfftwindow
    #RPfft[PLcenter-valmaxlen:PLcenter+valmaxlen+1] = \
    #    PLfft[PLcenter-valmaxlen:PLcenter+valmaxlen+1]
    #RPfft[PLcenter-valminlen:PLcenter+valminlen]=np.zeros((1,2*valminlen))
    
    #RPfft[PLcenter-valmaxlen:PLcenter-valminlen] *= PLfftwindow
    #RPfft[PLcenter+valminlen:PLcenter+valmaxlen] *= PLfftwindow

    RPvlfdata=scipy.fft.ifft(RPfft)# %computes VLF signal from rempvminfreq to rempvmaxfreq
    RPvlfdata=RPvlfdata-np.mean(RPvlfdata)# %setting again meanvalue to zero   
    #RPvlfdata=RPvlfdata[padlen:padlen+datalen+1]
    RPvlfdata=np.real(RPvlfdata*np.conjugate(RPvlfdata))# amplitude squared data
    #%%%%
    #
    # RPvlfdata is the POWER of the received signal in the frequency band rempvminfreq:rempvmaxfreq
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Below one defines the indices for dividing data to "detectingdivision" pieces
    # the indices are not exactly at equal intervals !!!!!!, funny
    rawstep=datalen/detectingdivision
    dind=(np.ceil(np.arange(0,detectingdivision + 1, dtype='int32') * rawstep)).astype('int')
    #for div in range(0,detectingdivision):        # creating division indices
    #    dind[div]=np.ceil(((div-1)*rawstep))
    #if dind[detectingdivision-1] != datalen
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for div in range(0,detectingdivision):
        RPvlfdata[dind[div]:dind[div+1]] = np.sqrt(np.mean(RPvlfdata[dind[div]:dind[div+1]])) * np.ones((dind[div+1]-dind[div]))
    if debug:
        print(f'RMS mean:{np.mean(RPvlfdata)},min:{np.min(RPvlfdata)},max:{np.max(RPvlfdata)}')
    #%%%%%%%%%%%%%%%%%%% BACGROUND noise level estimate formation
    #
    rawstep=datalen/BGRdivision
    dind=(np.ceil(np.arange(0,BGRdivision + 1, dtype='int32') * rawstep)).astype('int')
    #for div in range(0,BGRdivision+1):#        % creating division indices
    #    dind[div]=np.ceil(((div-1)*rawstep))
    #dind[BGRdivision] = datalen
    #%%%%%%%%%%%%%%%%%%%%%
    # creating the minimum values for every BGRdivision
    BRPvlfdatava=np.zeros((datalen))
    for div in range(0, BGRdivision):
        BRPvlfdatava[dind[div]:dind[div+1]] = np.min(RPvlfdata[dind[div]:dind[div+1]])
    #%%%%%%%%%%%%%%%%%%%%%
    
    '''
    % If we use adaptive filter mode, whichs modifies the rempeakvalue
    % after every "totaltime",and tries to keep loss percent at constatn level,
    % we use the very first data block to define 
    % possibly a suitable value for rempeakvalue
    % here starts the real computation of the lightning filtering
    % the rempeakvalue is the given in the control parameters of the program
    % Now we first try to find the best value using the very first
    % applying the rempeakvalue by making the referencevalue
    %disp('olen taalla A')
    '''
    trackcounter=0
    trackaccuracy=10
    while trackaccuracy>1.:#  %The value is given
        refvalue=np.sqrt(rempeakvalue**2 + 4*(BRPvlfdatava*BRPvlfdatava));
        peakfiltvector=0.5*np.sign(-RPvlfdata+refvalue)+0.5
        datalossestimate=100*(1-(np.sum(np.real(peakfiltvector))/datalen))#  %prosentteja                                                           
        corrdir=np.sign(datalossestimate-Sfericlosspercent)# %i f positive (1) then one has to increase 
                                                        # rempeakvalue else (-1) decrease
        corrsize=abs((datalossestimate-Sfericlosspercent)*corrstep)# %value of correctionstep
        trackaccuracy=abs(Sfericlosspercent-datalossestimate)# % goal minus present value 
        rempeakvalue=rempeakvalue+corrdir*corrsize#    % new rempeakvalue computed
        if debug:
            print(f'refvalue:{np.mean(refvalue)},datalosse:{datalossestimate}%,corrsize:{corrsize},corrdir:{corrdir},rempeakvalue:{rempeakvalue}')
        if rempeakvalue < Rempvlimit:
            if debug:
                print(f'Rempvlimit reached!')
            rempeakvalue=Rempvlimit# % sets lowest value to that defined by user
            trackaccuracy=0.1# %quarantees while koop end
            
        trackcounter=trackcounter+1
        if trackcounter>=trackcounterlimit:
            if debug:
                print(f'rempeakvalue search failed, trackcountervalue exceeds limit {trackcounterlimit}')
            return vlfdata

    if debug:
        print(f'Automatic rempeakvalue in the start = {rempeakvalue}')

    #refvalue=np.sqrt(rempeakvalue**2 + 4*(BRPvlfdatava*BRPvlfdatava))
    #peakfiltvector(1,padlen+1:end-padlen)=0.5*sign(-RPvlfdata+refvalue)+0.5;
    #peakfiltvector=0.5*np.sign(-RPvlfdata+refvalue)+0.5#  %control figure
      
    #if TrackSfericLevel: # if==1, tracks data loss level slowly and changes rempeakvalue in order to keep it at the given value.
    #    if s==1:                   
    ontime=np.sum(np.real(peakfiltvector))
    #        #REFvalue=refvalue;
    #    else:
    #        ontime +=np.sum(real(peakfiltvector),2)
    #
    # REFvalue=REFvalue+refvalue;
    # REFvalue=REFvalue/smax;
    
    meandataloss=100*(1-ontime/datalen)# % mean data loss in 1:smax periods
    if debug:
        print(f'mean data loss = {meandataloss:.1f} percent for ')
    
    # corrdir=sign(meandataloss-Sfericlosspercent)# %if positive (1) then one has to 
    # increase rempeakvalue else (-1) decrease
    # corrsize=abs(meandataloss-Sfericlosspercent)*10 #  %*30
    # trackspeedfactor=np.max(1,np.sqrt(np.sqrt(rempeakvalue/50)))# %3,rempeakvalue/50
    #if corrsize < 5:
    #    corrsize=0
    #rempeakvalue=rempeakvalue+corrdir*(corrsize+trackspeedfactor)#    % new rempeakvalue computed
    #if rempeakvalue < Rempvlimit:
    #    rempeakvalue=Rempvlimit
    #    print(f'NEW rempeakvalue = {rempeakvalue:d}')  
    # %%%%%%%%%%%%
    #peakfiltvector=np.array([np.zeros((padlen)), peakfiltvector, np.zeros((padlen))])
    #peakfft=scipy.fftpack.fft(np.real(peakfiltvector))
    #peakwindow=scipy.signal.windows.kaiser(PLfftlen,roundingbeeta)
    #peakfft=peakwindow*(scipy.fftpack.fftshift(peakfft))
    #peakfiltvector=np.real(scipy.fftpack.ifft(scipy.fftpack.fftshift(peakfft)))
    # %%%%%%%%%
    if correctPRloss:
        #Losspeakvector=real(peakfiltvector[padlen:-padlen])
        Losspeakvector=np.real(peakfiltvector)

    
    Nvlfdata=np.real(peakfiltvector)*Nvlfdata;
    vlfdata=Nvlfdata#[padlen:end-padlen+1] # trim leading and trailing pads to the 2^N length
    
    return vlfdata
