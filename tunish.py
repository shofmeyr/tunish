#!/usr/bin/python -u

import wave
import numpy
import math
import sys
import struct
import matplotlib
import argparse
import time
import matplotlib.pyplot as plt

import playtones
import notefreqs

matplotlib.rcParams['lines.markersize'] = 6
matplotlib.rcParams['lines.markeredgewidth'] = 0
matplotlib.rcParams['font.size'] = 10

_FRAME_RATE = 44100
_SAMPLE_WIDTH = 2


def read_data(fname):
    start_t = time.clock()
    s = wave.open(fname)
    if s.getframerate() != _FRAME_RATE:
        sys.exit("Frame rate needs to be", _FRAME_RATE)
    if s.getsampwidth() != _SAMPLE_WIDTH:
        sys.exit("Sample width needs to be", _SAMPLE_WIDTH)
    nchannels = s.getnchannels()
    print >> sys.stderr, "  Number channels", nchannels

    nframes = s.getnframes()
    frames = s.readframes(nframes)
    s.close()

    data = []
    for i in range(0, len(frames), 2 * nchannels):
        ch_tot = 0.0
        for c in range(nchannels):
            ch_tot += struct.unpack("<h", frames[i + 2 * c] + frames[i + 1 + 2 * c])[0]
        data.append(ch_tot / 2)
    
    print >> sys.stderr, "Read", fname +",", len(data), "data points", \
        "(%.1fs)" % (float(nframes) / _FRAME_RATE), "in %.1fs" % (time.clock() - start_t)

    return data


def plot_wave(data):
    tot_len = len(data)
    t = numpy.arange(0, float(tot_len), 1)
    t = t / _FRAME_RATE
    plt.plot(t, data[:tot_len])
    #plt.xlim((0.076, 0.0965))
    plt.axhline(y=0, color="black")


def get_tones_zcross(data, db_lim, max_peak_ratio):
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
                tones.append(float(_FRAME_RATE) / wavelen)
                times.append(float(prev_max_i) / _FRAME_RATE)
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


def annotate(times, tones, period, print_data):
    note_freqs = notefreqs.NoteFreqs()
    prev_note = note_freqs.get_nearest_note(tones[0])
    prev_t = times[0]
    prev_tone = tones[0]
    for i in range(1, len(times)):
        if tones[i] == 0: continue
        if times[i] < prev_t + period: continue
        #if abs(prev_tone - tones[i]) <= 0.02: continue
        note = note_freqs.get_nearest_note(tones[i])
        if prev_note == note: continue
        plt.text(prev_t, prev_tone + 5, prev_note)
        if print_data:
            print "%10.3f" % prev_t, "%5s" % prev_note, "%8.3f" % (times[i] - prev_t), \
                "%8.0f" % prev_tone, "%8.0f" % note_freqs.get_note_freq(prev_note)
        prev_note = note
        prev_t = times[i]
        prev_tone = tones[i]
    
def main():
    #sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 0)

    parser = argparse.ArgumentParser(description="Find pitch in wav file.")
    parser.add_argument(dest= "fnames", metavar="file", nargs=1,
                        help="A wav file to be processed")
    parser.add_argument("-p", action="store_true", dest="plot_wave", default=False,
                        help="Plot wave")       
    parser.add_argument("-o", action="store_true", dest="print_data", default=False,
                        help="Print data")       
    parser.add_argument("-d", type=int, dest="db_lim", default=25,
                        help="Decibel limit for noise (default %(default)d)")
    parser.add_argument("-f", type=float, dest="max_freq", default=400, 
                        help="Max frequency to plot (default %(default)d)")
    parser.add_argument("-s", type=float, dest="post_smooth_int", default=0.1, 
                        help="Median-smoothing interval in secs (default %(default).2f)")
    parser.add_argument("-r", type=float, dest="max_peak_ratio", default=0.95,
                        help="Max. peak ratio (default %(default).2f)")
    parser.add_argument("-z", type=float, dest="pre_smooth_wind", default=100, 
                        help="Pre-smoothing window (use 0 for g, 100 for v) " + \
                            "(default %(default)d)")
    parser.add_argument("-t", type=float, dest="period", default=0.1,
                        help="Output period in secs (default %(default).2f)")
    
    options = parser.parse_args()

    print >> sys.stderr, "Parameters: "
    for line in parser.format_help().split("\n"):
        if line == "": continue
        line = line.strip()
        if line[0] != "-": continue
        flag = line[0:2]
        if flag == "-h": continue
        if line[len(flag) + 1:len(flag) + 2] == " ": continue
        opt = line.split()[1]
        print >> sys.stderr, " ", flag, "%-8s" % vars(options)[opt.lower()], \
            line[len(flag) + 1 + len(opt):].strip()

    data = read_data(options.fnames[0])

    if options.plot_wave: 
        plot_wave(data)
        plt.show()

    start_t = time.clock()
    if options.pre_smooth_wind > 0:
        data = numpy.convolve(data, numpy.ones(options.pre_smooth_wind) / 
                              options.pre_smooth_wind, mode="valid")
    max_t = float(len(data)) / _FRAME_RATE
    times, tones = get_tones_zcross(data, options.db_lim, options.max_peak_ratio)
    print >> sys.stderr, "Pitch detection took %.3fs" % (time.clock() - start_t)

    if options.post_smooth_int > 0: 
        times, tones = median_smooth(times, tones, options.post_smooth_int)

    plt.ylim((0, options.max_freq))
    plt.plot(times, tones, marker="*", ls='', alpha=0.5, color="red")
    # guitar frequencies
    for f in [82.4, 110.0, 146.8, 196.0, 246.9, 329.6, 1046.5]: 
        plt.axhline(y=f, color="black")

    annotate(times, tones, options.period, options.print_data)

    plt.show()


if __name__ == "__main__": main()

 

 
