#!/usr/bin/python -u

import wave
import numpy
import numpy.fft
import math
import pylab
import sys
import struct
import matplotlib
import argparse
import scipy.signal
import time
import alsaaudio
import random
import matplotlib.pyplot as plt
import thread

matplotlib.rcParams['lines.markersize'] = 6
matplotlib.rcParams['lines.markeredgewidth'] = 0
matplotlib.rcParams['font.size'] = 10

freq_to_notes = {}
notes_to_freq = {}
notes = []
C2_freq = 65.406
note_sounds = {}

_channels = 2
_sample_size = 2             
_frame_size = _channels * _sample_size 
_frame_rate = 44100                  
_period_size = _frame_rate / 10

def compute_note_freqs():
    note_freq_step = 2.0 ** (1.0 / 12) 
    shifted_freq = C2_freq
    i = 0
    for octave in range(2, 7):
        for note in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
            full_note = note + str(octave) 
            notes.append(full_note)
            freq_to_notes[shifted_freq] = full_note
            notes_to_freq[full_note] = shifted_freq
            note_sounds[full_note] = get_note_sound(full_note)
            #print "%4d" % i, "%4s" % freq_notes[shifted_freq], "%.3f" % shifted_freq
            shifted_freq *= note_freq_step
            i += 1

def play_tones(times, tones, play_choice):
    print "PLAY"
    pcm = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK)
    pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    pcm.setchannels(_channels)
    pcm.setrate(_frame_rate)
    pcm.setperiodsize(_period_size)
    
    #for i in range(0, 100):
    #    pcm.write(note_sounds["C4"])
    #return

    tstep = int(len(times) / (times[-1] / (float(_period_size) / float(_frame_rate))))
    start_t = time.clock()
    prev_note = get_nearest_note(tones[0], 0, None)
    for i in range(1, len(times), tstep):
        note = get_nearest_note(tones[i], tones[i - tstep], prev_note)
        prev_note = note
        print "%-8.3f" % times[i], "%-3s" % note, "%8.0f" % notes_to_freq[note], "%-8.0f" % tones[i]

        if play_choice == "a": pcm.write(get_freq_sound(tones[i]))
        elif play_choice == "n": pcm.write(note_sounds[note])

    
def get_note_sound(note):
    freq = notes_to_freq[note]
    return get_freq_sound(freq)

def get_freq_sound(freq):
    step = 2.0 * math.pi / (_frame_rate / freq)
    max_val = 2.0 * math.pi
    wave_data = ""
    for x in numpy.arange(0, max_val, step):
        y = int((math.sin(x) + 1) * 32767.5)
        for c in range(_channels): wave_data += (chr(y & 0xff) + chr((y>>8) & 0xff))
    nwaves = int(_period_size * _frame_size / len(wave_data))
    return wave_data * nwaves

# def get_freq_sound(f):
#     t_max = math.floor(float(_period_size) / float(_frame_rate) * float(f)) / float(f)
#     step = 1.0 / _frame_rate
#     wave_data = ""
#     nharmonics = 5
#     for t in numpy.arange(0.0, t_max, step):
#         x = 0.0
#         for i in numpy.arange(1, nharmonics + 1, 1.0):
#             x += (math.sin(2.0 * math.pi * t * f * i)) / i

#         y = int((x + 1.0) * 32767.5)
#         for c in range(_channels): wave_data += (chr(y & 0xff) + chr((y>>8) & 0xff))
#     return wave_data

def get_nearest_note(freq, prev_freq, prev_note):
    max_err = freq * 0.01
    index = int(round(12.0 * math.log(freq / C2_freq, 2)))
    note = notes[index]
    if abs(freq - notes_to_freq[note]) <= max_err: return note
    index = int(math.floor(12.0 * math.log(freq / C2_freq, 2)))
    return notes[index]

    
def parabolic(f, x):
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)


def read_data(fname):
    s = wave.open(fname)
    nframes = s.getnframes()
    frames = s.readframes(nframes)
    framerate = s.getframerate()
    sample_width = s.getsampwidth()
    nchannels = s.getnchannels()
    s.close()

    print "Channels", nchannels
    print "Sample width", sample_width
    print "Read in", fname
    print "Frame rate", framerate
    print "Number frames", nframes
    print "Duration %.3fs" % (float(nframes) / framerate) 

    if sample_width != 2:
        print "Sample width needs to be 2"
        sys.exit(0)

    data = []
    for i in range(0, len(frames), 2 * nchannels):
        ch_tot = 0.0
        for c in range(nchannels):
            ch_tot += struct.unpack("<h", frames[i + 2 * c] + frames[i + 1 + 2 * c])[0]
        data.append(ch_tot / 2)
    
    return data, framerate


def plot_wave(data, framerate):
    tot_len = len(data)

    t = numpy.arange(0, float(tot_len), 1)
    t = t / framerate
    plt.plot(t, data[:tot_len])
    #plt.xlim((0.076, 0.0965))
    plt.axhline(y=0, color="black")

def get_tones_fft(data, nbits, framerate, db_lim):
    wsize = 2**nbits
    tot_len = len(data)
    tones = []
    times = []
    step_size = wsize / 128
    if step_size == 0: step_size = 1
    for i in range(0, tot_len, step_size):
        seg = data[i:i + wsize]
        wind_data = seg * scipy.signal.blackmanharris(len(seg))
        amplitudes = numpy.abs(numpy.fft.rfft(wind_data))
        max_ampl_index = numpy.argmax(amplitudes)
        max_power = 10.0 * numpy.log10((amplitudes[max_ampl_index] / float(wsize)) ** 2)
        if max_power > db_lim:
            true_freq = parabolic(numpy.log(amplitudes), max_ampl_index)[0]
            true_freq = float(framerate) * true_freq / len(wind_data)
            tones.append(true_freq)
            times.append(float(i) / framerate)
    return times, tones

def get_tones_zcross(data, framerate, db_lim, max_peak_ratio):
    max_wavelen = 880
    min_wavelen = 40
    prev_max = 0
    prev_max_i = 0
    curr_max = 0.0
    curr_max_i = 0
    av_wavelen = 0.0
    tones = []
    times = []
    lim_max = 10.0 ** (db_lim / 10.0)
    for i in range(0, len(data) - 1):
        if data[i] >= curr_max:
            curr_max = data[i]
            curr_max_i = i
        if data[i] <= 0 and data[i + 1] > 0: 
            if curr_max / (prev_max + 0.0001) < max_peak_ratio: continue
            wavelen = curr_max_i - prev_max_i
            if wavelen < min_wavelen: continue
            if wavelen <= max_wavelen and prev_max != 0 and curr_max >= lim_max: 
                tones.append(float(framerate) / wavelen)
                times.append(float(prev_max_i) / framerate)
            shift = True
        elif i - prev_max_i > max_wavelen:
            shift = True
        else:
            shift = False
        if shift:
            prev_max = curr_max
            prev_max_i = curr_max_i
            curr_max = data[i]
            curr_max_i = i 
            
    return times, tones

def get_tones_acorr(data, framerate, db_lim, noise):
    gradients = numpy.empty(len(data) - 1)
    for i in range(len(data) - 1):
        if data[i + 1] >= data[i]: gradients[i] = 1
        else: gradients[i] = -1
        if i > 0 and gradients[i - 1] != gradients[i]: gradients[i - 1] = 0
    min_wavelen = 40       # max freq of 1100hz
    max_wavelen = 880      # min freq of 50hz
    tones = []
    times = []
    ai = 0
    lim_max = 10.0 ** (db_lim / 10.0)
    max_ai = len(gradients) - 2.0 * max_wavelen
    while (ai < max_ai):
        for wi in range(min_wavelen, max_wavelen):
            noise_count = int(wi * noise)
            misses = 0
            peak = False
            for bi in range(ai, ai + wi): 
                if not peak and gradients[bi] == 0: peak = True
                if gradients[bi] == gradients[bi + wi]: continue
                misses += 1
                if misses >= noise_count: break
            if peak and misses < noise_count:
                ampl = max(data[ai:ai + wi])
                if ampl >= lim_max:
                    tones.append(float(framerate) / wi)
                    times.append(float(bi + wi) / framerate)
                break
        ai += wi * 2

    return times, tones

def median_smooth(times, tones, smooth_interval):
    if smooth_interval > times[-1]: smooth_interval = times[-1]
    t = 0.0
    ti = 0
    smooth_tones = []
    smooth_times = []
    while (t <= times[-1] - smooth_interval):
        for i in range(1, len(times) - ti):
            if times[ti + i] - t >= smooth_interval:
                sorted_seg = sorted(tones[ti:ti + i])
                smooth_tones.append(sorted_seg[i / 2])
                smooth_times.append(times[ti + i /2])
                break
        ti += 1
        t = times[ti]
    return smooth_times, smooth_tones

def annotate(times, tones):
    prev_note = get_nearest_note(tones[0], 0, None)
    for i in range(1, len(times), 35):
        note = get_nearest_note(tones[i], tones[i - 100], prev_note)
        prev_note = note
        #print times[i], tones[i], note
        plt.text(times[i], tones[i] + 5, note)


    
def main():
    parser = argparse.ArgumentParser(description="Find pitch in wav file.")
    parser.add_argument(dest= "fnames", metavar="file", nargs=1,
                        help="A wav file to be processed")
    parser.add_argument("-p", action="store_true", dest="plot_wave", default=False,
                        help="Plot wave")       
    parser.add_argument("-o", action="store_true", dest="print_data", default=False,
                        help="Print data")       
    parser.add_argument("-w", type=int, dest="nbits", default=13, 
                        help="Number of bits in FFT window (default %(default)d)")
    parser.add_argument("-d", type=int, dest="db_lim", default=25,
                        help="Decibel limit below which sound is ignored (default %(default)d)")
    parser.add_argument("-f", type=float, dest="max_freq", default=400, 
                        help="Max frequency to plot (default %(default)d)")
    parser.add_argument("-n", type=float, dest="noise", default=0.25, 
                        help="Noise level for ACORR (default %(default)f)")
    parser.add_argument("-s", type=float, dest="post_smooth_int", default=0.1, 
                        help="Interval for median-smoothing in secs (default %(default)f)")
    parser.add_argument("-r", type=float, dest="max_peak_ratio", default=0.95,
                        help="Max. peak ratio for ZCROSS (default %(default)f)")
    parser.add_argument("-z", type=float, dest="pre_smooth_wind", default=100, 
                        help="Window for pre-smoothing (use 0 g, 100 for v) (default %(default)d)")
    parser.add_argument("-m", dest="methods", default="ZCROSS",
                        help="Methods: ZCORR,ACORR,FFT (default %(default)s)")
    parser.add_argument("-P", dest="play", default=None,
                        help="Play sample (a - actual, n - nearest notes)")
    
    options = parser.parse_args()

    compute_note_freqs()

    #play_tones(None, None)
    #return

    print "Parameters: "
    for line in parser.format_help().split("\n"):
        if line == "": continue
        line = line.strip()
        if line[0] != "-": continue
        flag = line[0:2]
        if flag == "-h": continue
        if line[len(flag) + 1:len(flag) + 2] == " ": continue
        opt = line.split()[1]
        print " ", flag, "%-8s" % vars(options)[opt.lower()], line[len(flag) + 1 + len(opt):].strip()

    start_t = time.clock()
    data, framerate = read_data(options.fnames[0])
    print "Read %d data points in %.3fs" % (len(data), time.clock() - start_t)

    if options.pre_smooth_wind > 0:
        smoothed = numpy.convolve(data, numpy.ones(options.pre_smooth_wind) / options.pre_smooth_wind, 
                                  mode="valid")
    else:
        smoothed = data
    if options.plot_wave: 
        plt.subplot(2, 1, 0)
        plot_wave(data, framerate)
        plt.subplot(2, 1, 1)
        plot_wave(smoothed, framerate)
        plt.show()

    g_freqs = [82.4, 110.0, 146.8, 196.0, 246.9, 329.6, 1046.5]

    fig = plt.figure()

    plt.subplots_adjust(hspace=0.3, wspace=0.15, left=0.05, right=0.95, 
                        top=0.95, bottom=0.1)
    methods = options.methods.split(",")
    max_t = float(len(data)) / framerate
    for i, method in enumerate(methods):
        start_t = time.clock()
        if method == "ZCROSS": 
            times, tones = get_tones_zcross(smoothed, framerate, options.db_lim, options.max_peak_ratio)
        elif method == "FFT": 
            times, tones = get_tones_fft(smoothed, options.nbits, framerate, options.db_lim)
        elif method == "ACORR": 
            times, tones = get_tones_acorr(smoothed, framerate, options.db_lim, options.noise)
        else:
            print "Method", method, "is not supported"
            return
        print method, "computation took %.3fs" % (time.clock() - start_t)
        if options.post_smooth_int > 0: 
            times, tones = median_smooth(times, tones, options.post_smooth_int)
        plt.subplot(len(methods), 1, i, title=method, ylim=(0, options.max_freq), xlim=(0, max_t))
        plt.plot(times, tones, marker="*", ls='', alpha=0.5, color="red")
        annotate(times, tones)
        for f in g_freqs: plt.axhline(y=f, color="black")
        if options.print_data:
            print method
            for i in range(0, len(times)): print "%8d" % i, "%.4f" % times[i], "%8.0f" % tones[i]

    if options.play != None: thread.start_new_thread(play_tones, (times, tones, options.play))
    plt.show()



if __name__ == "__main__": main()

 

 
