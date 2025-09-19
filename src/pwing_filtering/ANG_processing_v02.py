import os
import glob
import numpy as np
from pathlib import Path
import cdflib
import cdflib.xarray
from multiprocessing import Pool
from PLHRfilter import *
from attenuateVLFstations import *
from removepeakvalues import *
import struct as st
import scipy as sci
import xarray as xr

def read_pwing_data(fpath):
    with open(fpath, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
        d = st.unpack('<BBBBBBHBHBI3s13B', fileContent[0:32])
        dt = np.datetime64(f'{2000+d[0]}-{d[1]:02d}-{d[2]:02d}T{d[3]:02d}:{d[4]:02d}:{d[5]:02d}.{d[6]:03d}', 'us')
        fs = float(d[7]) # there is a bug, it should be 40e3
        if fs != 40000.:
            fs = fs
        exp0 = d[8]
        n_ch = d[9]
        station = d[11]
        nrec = len(fileContent[32:])//2
        data_raw = np.array(st.unpack(f'{nrec}h', fileContent[32:]), dtype='float')
        data_v = np.reshape(data_raw, (n_ch, nrec//3), 'F')
        data_v *= 20. / 65536. # raw -> V 
        # flip North-South -> South-North, then an incident angle can be called azimuth
        data_v[0, :] *= -1.
    file.close()   
    return (dt, fs, n_ch, station, data_v)

def get_B_dipole(r, lat):
    B0 = 3.12e-5 # T
    lat = np.pi/2. - np.deg2rad(lat)
    return B0/r**3 * np.sqrt(1 + 3 * np.cos(lat)**2)

def get_smx(wave, stime, fftlen=1024, nmax=32768):
    fs = 40000. # Hz
    Bx = np.real(wave)
    By = np.imag(wave)
    nsize = len(wave)
    smx = []
    time = []
    for w in range(0, nsize, nmax // 2):
        if w + nmax > nsize:
            break
        Bxx = Bx[w:w+nmax]
        Byy = By[w:w+nmax]
        fftstep = fftlen // 2
        Jx = np.zeros((fftlen//2, int(nmax / fftstep - 1)), dtype='complex')
        Jy = np.zeros((fftlen//2, int(nmax / fftstep - 1)), dtype='complex')
        Jxy = np.zeros((fftlen//2, int(nmax / fftstep - 1)), dtype='complex')
        for s in range(0, nmax, fftlen // 2):
            if s+fftlen > nmax:
                break
            Freq = sci.fft.fftfreq(fftlen, 1./fs)
            Jx[:, s // fftstep] = (sci.fft.fft(Bxx[s:s+fftlen] * sci.signal.windows.hann(fftlen) * 2.) / (Freq[1] * fftlen))[0:fftlen//2]
            Jy[:, s // fftstep] = (sci.fft.fft(Byy[s:s+fftlen] * sci.signal.windows.hann(fftlen) * 2.) / (Freq[1] * fftlen))[0:fftlen//2]
            Jxy[:, s // fftstep] = (np.conjugate(Jy[:, s // fftstep]) * Jx[:, s // fftstep])[0:fftlen//2]
       
        av_Jxx = np.real(np.mean(Jx*np.conjugate(Jx), axis=1))
        av_Jyy = np.real(np.mean(Jy*np.conjugate(Jy), axis=1))
        av_Jxy = np.mean(Jxy, axis=1)
        sm = np.zeros((fftlen//2,2,2), dtype='complex')
        sm[:, 0, 0] = av_Jxx
        sm[:, 1, 1] = av_Jyy
        sm[:, 0, 1] = av_Jxy
        sm[:, 1, 0] = np.conjugate(av_Jxy)
        smx.append(sm)
        time.append(stime + np.timedelta64(int((w + nmax //2) / fs * 1e6), 'us'))
        
    return np.array(smx), np.array(Freq[0: fftlen // 2]), time
        
def get_stokes(sm):
    if sm.ndim == 4:
        Jxx = sm[:, :, 0, 0]
        Jyy = sm[:, :, 1, 1]
        Jxy = sm[:, :, 0, 1]
    elif sm.ndim == 3:
        Jxx = sm[:, 0, 0]
        Jyy = sm[:, 1, 1]
        Jxy = sm[:, 0, 1]
    elif sm.ndim == 2:
        Jxx = sm[0, 0]
        Jyy = sm[1, 1]
        Jxy = sm[0, 1]
    else:
        print(f'Not allowed SMX dimension (expected 2,3,4 and given {sm.ndims}')
    #+= Stokes' parameters, Taubenschuss and Santolik, 2019 Springer
    # S = BxBx* + ByBy*
    # Q = BxBx* - ByBy*
    # U = 2 * Re(BxBy*)
    # V = 2 * Im(BxBy*)
    # tau = 0.5 * np.arctan(U/Q)
    # beta = 0.5 * np.arcsin(V/np.sqrt(Q**2+U**2+V**2))
    S = np.real(Jxx + Jyy)
    Q = np.real(Jxx - Jyy)
    U = 2. * np.real(Jxy)
    V = 2. * np.imag(Jxy)
    #-=
    tau = 0.5 * np.arctan(U/Q)
    ellip = np.tan(0.5 * np.arcsin(V/np.sqrt(Q**2 + U**2 + V**2)))
    Dp2 = np.sqrt(Q**2 + U**2 + V**2) / S
    Dl2 = np.sqrt(Q**2 + U**2) / S
    Dc2 = V / S
    #
    #ellip *= -1.
    #Dc2 *= -1.
    #
    return ellip, tau, Dp2, Dl2, Dc2

def add_istp_var(cdf, var_name, rec_vary, data, catdesc, fieldnam, units, vartype,
                 fillval, depend_0, display_type, validmin=None, validmax=None, format_type=None, scaletyp=None):
    cdf.write_var({"Variable": var_name, 
                   "Data_type": "",
                   "Num_elements": 1,
                   "Rec_vary": rec_vary}, var_data=data)
    cdf.write_attribute(var_name, 'CATDESC', catdesc)
    cdf.write_attribute(var_name, 'FIELDNAM', fieldnam)
    cdf.write_attribute(var_name, 'VAR_TYPE', vartype)
    cdf.write_attribute(var_name, 'UNITS', units)
    cdf.write_attribute(var_name, 'FILLVAL', fillval)
    cdf.write_attribute(var_name, 'DEPEND_0', depend_0)
    cdf.write_attribute(var_name, 'DISPLAY_TYPE', display_type)
    if validmin is not None:
        cdf.write_attribute(var_name, 'VALIDMIN', validmin)
    if validmax is not None:
        cdf.write_attribute(var_name, 'VALIDMAX', validmax)
    if format_type is not None:
        cdf.write_attribute(var_name, 'FORMAT', format_type)
    if scaletyp is not None:
        cdf.write_attribute(var_name, 'SCALETYP', scaletyp)

def save_to_cdf(out_path, times, freq, BSUM, ell, tau, Dp2, Dl2, Dc2, station, fs):
    ds = xr.Dataset(
    {
        "BSUM": xr.DataArray(
            BSUM,
            dims=("time", "freq"),
            coords={"freq": freq, "time": times},
            attrs={
                "CATDESC": "Sum of B-field Power",
                "FIELDNAM": "BSUM",
                "VAR_TYPE": "data",
                "UNITS": "V^2/Hz",
                "FILLVAL": np.nan,
                "DEPEND_0": "time",
                "DISPLAY_TYPE": "spectrogram",
                "VALIDMIN": 0.0,
                "VALIDMAX": float(np.nanmax(BSUM)),
                "FORMAT": "F10.2",
                "SCALETYP": "log"
            },
        ),
        "ellipticity": xr.DataArray(
            ell,
            dims=("time", "freq"),
            coords={"freq": freq, "time": times},
            attrs={
                "CATDESC": "Ellipticity",
                "FIELDNAM": "Ellip.",
                "VAR_TYPE": "data",
                "UNITS": "1",
                "FILLVAL": np.nan,
                "DEPEND_0": "time",
                "DISPLAY_TYPE": "spectrogram",
                "VALIDMIN": -1.0,
                "VALIDMAX": 1.0,
                "FORMAT": "F7.3",
                "SCALETYP": "linear"
            },
        ),
        "tau": xr.DataArray(
            tau,
            dims=("time", "freq"),
            coords={"freq": freq, "time": times},
            attrs={
                "CATDESC": "Tau angle",
                "FIELDNAM": "Tau",
                "VAR_TYPE": "data",
                "UNITS": "deg",
                "FILLVAL": np.nan,
                "DEPEND_0": "time",
                "DISPLAY_TYPE": "spectrogram",
                "VALIDMIN": 0.0,
                "VALIDMAX": 180.0,
                "FORMAT": "F8.2",
                "SCALETYP": "linear"
            },
        ),
        "Dp2": xr.DataArray(
            Dp2,
            dims=("time", "freq"),
            coords={"freq": freq, "time": times},
            attrs={
                "CATDESC": "Degree of polarization",
                "FIELDNAM": "DOP",
                "VAR_TYPE": "data",
                "UNITS": "1",
                "FILLVAL": np.nan,
                "DEPEND_0": "time",
                "DISPLAY_TYPE": "spectrogram",
                "VALIDMIN": 0.0,
                "VALIDMAX": 1.0,
                "FORMAT": "F6.3",
                "SCALETYP": "linear"
            },
        ),
    },
    coords={
        "freq": xr.DataArray(
            freq,
            dims="freq",
            attrs={
                "CATDESC": "Frequency",
                "FIELDNAM": "Frequency",
                "VAR_TYPE": "support_data",
                "UNITS": "Hz",
                "FILLVAL": -1e31,
                "VALIDMIN": float(np.nanmin(freq)),
                "VALIDMAX": float(np.nanmax(freq)),
                "FORMAT": "E8.2"
            }
        ),
        "time": xr.DataArray(
            times,
            dims="time",
            attrs={
                "CATDESC": "Time tags (UTC)",
                "FIELDNAM": "Time",
                "VAR_TYPE": "support_data",
                "UNITS": "ms",
                "FILLVAL": -1e31,
                "FORMAT": "E12.3"
            }
        ),
    },
    attrs={
        "station": station,
        "station_CATDESC": "Station Name",
        "station_FIELDNAM": "Station",
        "fs": fs
    }
    )

    # Export to NASA CDF using cdflib's xarray interface
    cdflib.xarray.xarray_to_cdf(ds, out_path)

def make_filtered_signal(ang):
    dt64 = ang[0]
    fs = 4e4#ouj[1]
    data = ang[-1]
    ts = 590 # Lenght in seconds 9 min 50 secs
    period = 1.0 / fs#; Sampling Period
    nsize = data.shape[1] # If 100 kHz then nsize = 58998784 If 40 kHz then nsize=23600000
    time = [dt64 + np.timedelta64(int(s / fs * 1e6), 'us') for s in range(0, nsize)]

    #Maximum number of data points
    nmax = 65535 * 4# dt = 1.6384 s
    # Ch1 - NS direction
    # Ch2 - EW direction
    signal = data[0,:] + 1j * data[1, :]#[complex(data[0, i], data[1, i]) for i in range(0, nsize)]
    # Getting Tukey Window
    #tukey_win = sci.signal.windows.tukey # Getting the tukey window
    #Reading the data to format: 3601,2,16384
    ni = nsize / nmax # Number of iterations here sampling 40 kHz 23600000/16384 = 1440 but changes if sampling is 100 khz
    nfreq = 0 # Initializing new frequency for 16 point smoothing
    fftlen = 512 # df = 78.125 Hz
    #Separating data into 2 channels
    datalen = int(len(signal))
    datatime = datalen / fs
    step = 5 # s
    nstep = int(np.floor(step * fs))
    sigi = np.zeros((datalen), dtype='complex')
    for s in range(0, datalen, nstep):
        # wrap signal edges by 500 pts
        edges = sci.signal.windows.kaiser(1001, 10)
        wind = np.ones((nstep))
        wind[0:500] = edges[0:500]
        wind[-500:] = edges[501:]
        dat = signal[s:s+nstep]
        dat *= wind
        #print(dat.shape)
        a = PLHRfilter(dat)
        b = attenuateVLFstations(a)
        c = removepeakvalues(b)
        sigi[s:s+nstep] = c
    return sigi    

def process_single_file(ang_file):
    fs = 40000. # Hz
    data = read_pwing_data(ang_file)
    dt64 = data[0]
    sigi = make_filtered_signal(data)
    SM, freq, times = get_smx(sigi, dt64, fftlen=8192, nmax=65535)
    ell, tau, Dp2, Dl2, Dc2 = get_stokes(SM)
    tau = np.rad2deg(tau); tau += 90.; tau[np.where(tau < 0)] += 180.
    BSUM = np.real(SM[:,:,0,0] + SM[:,:,1,1])
    out_path = f"/data/sgo/ANG/cdf/{Path(ang_file).stem}.cdf"
    save_to_cdf(out_path, times, freq, BSUM, ell, tau, Dp2, Dl2, Dc2, 'ANGELI', fs)
    print(f"Processed and saved: {out_path}")
    return out_path

def process_multiple_files_parallel(ang_files, num_workers=1):
    with Pool(processes=num_workers) as pool:
        cdf_files = pool.map(process_single_file, ang_files)
    return cdf_files

# --- Running the pipeline ---
ang_file_list = glob.glob('/data/sgo/ANG/2025/*.ANG')
cdf_file_list = process_multiple_files_parallel(ang_file_list, num_workers=6)
#cdf_file = process_single_ang(ang_file_list)