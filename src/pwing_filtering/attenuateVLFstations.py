import numpy as np
import scipy
import scipy.signal
import scipy.fft as scifft

def attenuateVLFstations(sig, fs=40000.):
    '''
    # some strong VLF transmitted stations are
    % attenuated for decreasing some unwanted
    % effects in other filterings or spectral
    % computations. Note that VLF station disturbes
    % the data via SIDELOBES at MUCH wider freguency range 
    % than the nominal modulation bandwidth
    % NOTE: In this progam version
    %       the "attenuateVLFstations" filter
    %       prepares the system ready
    %       also for other spectral domain
    %       filterings!
                             
    % NOTE: active VLF stations are
    % different at different times.
    %
    % Russian "classical alpha" stations are 11905,12649,and 14881
    % ..but.. there are some other very RARELY
    % used frequencies,too, in the same band
    % and have been seen in our data ..!!!!!!
    %                               
    % also miscellaneous disturbing man made narrow band
    % signals can occasionally be eliminated by adding their frequency to the  
    % VLF station list
    %
    % Control for band rejection filters
    '''
    debug = False
    VLFfreqlowlim=14990    # Hz, assumed lowest frequency for modulated VLF transmitter stations.
                           # Above which the listed VLF stations are
                           # notch filtered using the following parameters:
                           # bandwidth centered at nominal freqency where
    #                      # after filtering the signal level is exactly zero                      
    vzero1=0 #    % Hz
    vtaplen1=300 #  % Hz        % Tapering bandwidth with rounded edges
    vbeeta1=4
     
     
    # Below "VLFfreqlowlim" the stations are handled as very narrown band tansmissions
    # In practice those belong to the Russian ALPHA navigation system and usually there are three active 
    # frequecies. The Russian Alpha stations may occasionally transmit at many  more frequencies than 
    # they usually do.
    #  
    
    vzero2=3 #  % for AlPHA stations - or similar
    vtaplen2=120 #
    vbeeta2=3 #
    #%%%%%%%%
    
    # At lowest frequencies there often occasionally are quite narrow "unidentifed# spectral lines, which 
    # need special handling (power line control lines, power plants,
    # your own generator, computers, etc.)
    
    SPECIALfreqhighlim=4000  # Hz,Below this frequency the disturbing lines are handled with the following 
                             # notch filtercoefficient
    
    vzero3=0 
    vtaplen3=4
    vbeeta3=4
    #
    #
    # You may need to make ready a few different VLFstationlists,depending on your computation demands
    #
    # VLFstationlist=[11905,12649,14881,18060,19570,20250,...
    #    20900,21750,22100,21750,22100,25200,26700,27340,28010,35600]; 
    #      lists attenuated VLF frequencies, Note:16300 seems to be sometimes
    #      disturbing, an Indian station.
    # VLFstationlist=[11905,12500,12649,12990,13280,13580,14881,15620,16300,16400,18060,19570,20250...
    #    20900,21400,21750,22100,21400,21750,22100,23400,24000,25200,26700,27340,28010,34170,35600,37500];
    #
    # VLFstationlist=[11905,12090,12500,12649,14881,16400,18120,19620,20290,...
    #    21140,21370,21780,22120,23420,24000,24840,25200,26700,29700,35570,37500,38010];
 
    VLFstationlist = [150,250,300,350,400,543,550,572,650,739,848, \
                       944,1143,1146,1154,1240,1249,1450,1545,1850, \
                       2050,2346,2356,2394,2947,3309,3166,3462,3471, \
                       3757,11890,11905,11920,12090,12630,12649,12670, \
                       14500,14860,14881,14900,15840,16250,16400,16550, \
                       17950,18080,18340,19580,19880,20250,20600,20900,\
                       21070,21130,21440,21760,22100,22600,23370,23490,\
                       23590,23690,24000,24810,25200,26700,29700,30070,\
                       30170,35600,37500,38000]
    
    # take signal as complex series and calculate spectrum from the full length
    Pfft = scifft.fft(sig)
    #Pfft2 = np.real(Pfft * np.conjugate(Pfft))
    Freq = scifft.fftfreq(len(sig), d=1./fs)
    Freqmax = np.max(Freq)
    dF = Freq[1]
    #dF_half = dF / 2.
    
    for f in VLFstationlist:
        if f >= Freqmax:
            if debug:
                print(f'Max. freq={Freq[-1]} exceeded ({f})')
            break
        find = int(f / dF)
        if debug:
            print(f'VLF transmitter at f={f} Hz')
        if f > VLFfreqlowlim:
            zerovec = None # Hz
            winlen = 300 # Hz
            beeta = 4 # Kaiser koeficient
        elif f > SPECIALfreqhighlim:
            zerovec = 3 # Hz
            winlen = 120 # Hz
            beeta = 3
        else:
            zerovec = None # Hz
            winlen = 4 # Hz
            beeta = 4
        winsize = int(np.round(winlen / dF))
        if zerovec:
            zerosize = int(np.round(zerovec / dF))
        else:
            zerosize = 0
        kaiser = 1. - scipy.signal.windows.kaiser(winsize, beeta)
        if zerosize:
            win = np.array(np.concatenate((kaiser[0:winsize//2], np.zeros((zerosize), dtype='double'), 
                              kaiser[winsize//2:])))
        else:
            win = kaiser
        if debug:
            print(f'Find:{find}, winsize={winsize}, zerosize={zerosize}')
        # adaptive peak max seeker
        # tfind = np.argmax(np.abs(Pfft[find - winsize // 2: find + winsize // 2 + 1]))
        # if tfind != winsize // 2:
        #    if debug:
        #        print(f'Stronger peak found @ {Freq[find + tfind - winsize // 2]} for VLF @ f={f}')
        #    find = find + tfind - winsize // 2
                          
        f0 = int(find - winsize // 2 - zerosize // 2)
        f1 = int(find + winsize // 2 + zerosize // 2)
        if debug:
            print(f'F boundaries at {f0} and {f1}')
        winsize = len(win)
        Pfft[f0:f0+winsize] *= win
        Pfft[-f0-winsize:-f0] *= win

    vlfdata = scifft.ifft(Pfft)
    return vlfdata
    '''   
    for vlfind=1:vlfdim(1);
        win=1-kaiser(2*vtaplen(vlfind),vbeeta)';
        vzerovect=zeros(1,vzero(vlfind));
        winA=win(1:end/2);
        winB=win(end/2+1:end);
        LA=length(winA);
        LB=length(winB);
        LZ=length(vzerovect)/2;
    
    PLweigth(PLfftlen/2-vadd(vlfind)-LA-LZ+1:PLfftlen/2-vadd(vlfind)-LZ)=winA.*...
        PLweigth(PLfftlen/2-vadd(vlfind)-LA-LZ+1:PLfftlen/2-vadd(vlfind)-LZ);    
    PLweigth(PLfftlen/2-vadd(vlfind)-LZ+1:PLfftlen/2-vadd(vlfind)+LZ)=vzerovect.*...
        PLweigth(PLfftlen/2-vadd(vlfind)-LZ+1:PLfftlen/2-vadd(vlfind)+LZ);
    PLweigth(PLfftlen/2-vadd(vlfind)+LZ+1:PLfftlen/2-vadd(vlfind)+LZ+LB)=winB.*...
    PLweigth(PLfftlen/2-vadd(vlfind)+LZ+1:PLfftlen/2-vadd(vlfind)+LZ+LB);
    PLweigth(PLfftlen/2+vadd(vlfind)-LA-LZ+1:PLfftlen/2+vadd(vlfind)-LZ)=winA.*...
        PLweigth(PLfftlen/2+vadd(vlfind)-LA-LZ+1:PLfftlen/2+vadd(vlfind)-LZ);
    PLweigth(PLfftlen/2+vadd(vlfind)-LZ+1:PLfftlen/2+vadd(vlfind)+LZ)=vzerovect.*...
        PLweigth(PLfftlen/2+vadd(vlfind)-LZ+1:PLfftlen/2+vadd(vlfind)+LZ);
    PLweigth(PLfftlen/2+vadd(vlfind)+LZ+1:PLfftlen/2+vadd(vlfind)+LZ+LB)=winB.*...
        PLweigth(PLfftlen/2+vadd(vlfind)+LZ+1:PLfftlen/2+vadd(vlfind)+LZ+LB);
    '''
        