import numpy as np
import scipy
import scipy.signal
import scipy.fft as scifft

def PLHRfilter(sig, fs=40000.):
    debug = False
    '''
    DO NOT BELIEVE THAT IT ALWAYS FILTERS ALL LINES AS YOUR WISH. IT CAN BE
    DISTURBED BY YOUR OWN NOISE, SUDDEN CHANGES IN PLHR FREQUENCIES,
    GEOPHYSICAL REASONS (!!!), ELF TRANSMITTER HARMONICS AND NUMEROUS NOT CLEARLY
    RECOGNIZED REASONS......
    '''
    filtermode=1      # if==1 50Hz interval (all harmonics),if==2 100Hz interval (odd harmonics) up to PLHRmaxline
                  # filtermode =2 is usually a better selection
    PLHRmaxline=8500  # Maximum powerline line frequency to be filtered   
    checkband=7       # bandwidth in Hz reserved around nominal harmonic frequencies for 
                  # measuring the true frequencies

    PLchecklines=[150,300,550,572,575,739,750,950,1350,1450,1550,1650,1750, \
              1850,2150,2250,2350,2400,2450,2550,2650,2750,2850,2950, \
              3150,3250,3350,3450,3550,3750,3850]
         
    maxmeanratio=2.5    # parameter, which tests "PLHR checkline SNR". It should be somewether e.g. between 1.4 to 2.5
                    # The bigger "maxmeanratio", the higher SNR is demanded before the individual lines are used in
                    # determination of the PLHR frequencies.
    prevcorrfactor=1   # Helps sometimes getting proper start in the beginning of measurement and in noisy cases                    
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#  SETTING THE BANDWIDTH and shape OF THE NOTCH FIRLTERS IN POWERLINE FILTERING
                    

    taperingband=10    # Total bandwidth (Hz)used in individual notch filters, Suitable values e.g. between 10-30 Hz (!)
    rejectbeeta=20    # defines tapering shape (beeta parameter in kaiser window, which  cotrols the shape),
                   # suitable values 10-100, depends also on taperingband ;
                    
    correctPLHRfilterlosses=1  # if==1  corrects the power losses caused by powerline filtering.                 
                          # NOTE: SHOULD NORMALLY BE ALWAYS ==1
    '''
    %
    % Checking the real powerline frequency and making correction to the
    % nominal ones. Without this the powerline harmonic filtering is nonsense.
    % The assumption is after this that the mean frequency obtained from the
    % computation is really constant enough in timescale of "blockduration"
    % parameter given in the analysis control, and short term frequency
    % variations do not wander outside the notch filter rejection frequencies.
    % Notch filters is firned below 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''   
# defining powerline harmonic frequencies to be filtered
    startfreq=50
    if filtermode==1:
        deltaPLH=50
    elif filtermode==2:
        deltaPLH=100
    if np.max(PLchecklines) < PLHRmaxline:
        PLHRmaxline = np.max(PLchecklines)
    
    Pfft = scifft.fft(sig)
    Pfft2 = np.real(Pfft * np.conjugate(Pfft))
    Freq = scifft.fftfreq(len(sig), d=1./fs)
    dF = Freq[1]
    dF_half = dF / 2.
    taperstep = 2*int((0.5*(taperingband)/dF))
    if debug:
        print(f'FFTLEN={len(sig)},dF={dF},taperstep:{taperstep}')

    PLch=len(PLchecklines) # number of powerline harmonic frequencies in Hz
    checkbandcenterind=np.zeros(PLch, dtype='int')
    maxcheckbandind=np.zeros(PLch, dtype='int')
    
    PLharmonic=np.zeros(PLch)
    Pcorrfactor=np.zeros(PLch)
    
    halfbandstep=np.round(0.5*checkband/dF)
    
    checkbandpower=np.zeros((PLch,int(2*halfbandstep+1)))
    maxcheckpower=np.zeros(PLch)
    maxcheckfind=np.zeros(PLch)
    Pcorrfactor = list()
       
    for i, f in enumerate(PLchecklines):
        ind = np.where((Freq >= f - dF_half ) & (Freq < f + dF_half))[0]
        if len(ind) == 0:
            print(f'Warning: Frequency {f} out of bounds or not found.')
            continue
        f0 = (ind - taperstep // 2)[0]
        f1 = (ind + taperstep // 2 + 1)[0]
        if debug:
            print(f'Checking line at f={f} ({f0}-{f1}) Hz')
        #checkbandpower[i, 0:(f1-f0)] = Pfft2[f0:f1]
        maxcheckpower[i] = np.max(Pfft2[f0:f1])
        maxcheckfind[i] = f0 + np.argmax(Pfft2[f0:f1])
        if maxcheckpower[i]/np.mean(Pfft2[f0:f1]) > maxmeanratio:
            #Pcorrfactor.append(checkbandcenterind(PLcheck)-halfbandstep+maxcheckbandind(PLcheck))*PLdeltafreq/PLchecklines(PLcheck)# forming correction factor
            PLrejectwindow=1.-(scipy.signal.windows.kaiser((f1-f0),rejectbeeta))
            Pfft[f0:f1] *= PLrejectwindow
            Pfft[-f1:-f0] *= PLrejectwindow
    
    vlfdata = scifft.ifft(Pfft)
    '''        
    corrfactor=np.median(np.array(Pcorrfactor))
    if check < PLch/3:
        corrfactor=prevcorrfactor
    else:
        prevcorrfactor=corrfactor
    
    #CPLfreq=np.zeros(PLmaxind)
    #for k=1:PLmaxind
    #    CPLfreq(k)=corrfactor*PLfreq(k);
    CPLfreq = corrfactor * PLfreq
    
    #if filterZEVS==1
    #PLmaxind=PLmaxind+1;    
    #CPLfreq(PLmaxind)=82; %russian ZEVS frequency;
    #end

    taperstep = 2*fix((0.5*(taperingband)/PLdeltafreq));
    PLrejectind=zeros(1,PLmaxind);
    for k in range(0,PLmaxind):
        PLstep=taperstep;
        PLrejectind(k)=round(CPLfreq(k)/PLdeltafreq);

#%%%%%%%%%%%

    PLrejectloss=np.ones((PLmaxind), dtype='dlouble')
    PLHfiltWEIGTH = np.ones((len(PLfft), dtype='double') # the notch filter comb for PLH filtering will be formed to this vector

    for PLind=1:PLmaxind       
        PLHfiltWEIGTH(PLfftlen/2-PLrejectind(PLind)-taperstep/2+1:PLfftlen/2-PLrejectind(PLind)+taperstep/2)=PLrejectwindow; 
        PLHfiltWEIGTH(PLfftlen/2+PLrejectind(PLind)-taperstep/2+1:PLfftlen/2+PLrejectind(PLind)+taperstep/2)=PLrejectwindow;  
   
    # return to filtered time domain data

    PLfft=PLfft * PLHfiltWEIGTH # This command removes powerlineharmonics (in spectral domain)
    vlfdata=ifft(fftshift(PLfft)) # This command produces powerline free VLF data
    vlfdata=vlfdata[padlen:datalen+padlen] # and this removes zero paddings
    #PLHfiltWEIGTH=PLHfiltWEIGTH(1,padlen+1:datalen+padlen); % this removes padding from
    compdatalen=length(vlfdata) # the filter vector
    specind=ones(1,floor(PLHRmaxline/50)+1);
    PLHcorr=ones(1,floor(PLHRmaxline/50)+1);
    # If one wants correct the power levels in spectral channels containing
    # filtered power line, then one can try the following and hope that it
    # improves the data


#%%%%%%%%%%%%%%%%%%%%

    if correctPLHRfilterlosses:
        maxlinefgate=np.ceil(PLHRmaxline/deltafreq);   
        fstepfactor=deltafreq/PLdeltafreq
        CorrFactor=np.ones((fftlen/2,1))
        for sumind=1:maxlinefgate:
            CorrFactor(sumind)=sqrt(fstepfactor)/sqrt(sum((PLHfiltWEIGTH(end/2+1+(sumind-1)*fstepfactor:end/2+1+(sumind*fstepfactor)-1)).*...
            (PLHfiltWEIGTH(end/2+1+(sumind-1)*fstepfactor:end/2+1+(sumind*fstepfactor)-1))));

        PLHcorrfact=(1+( 0.3536*(CorrFactor-1))) #  %note:0.3536=1/(2*sqrt(2))
    '''
    return vlfdata