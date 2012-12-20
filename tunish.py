#!/usr/bin/python -u

import wave
import numpy
import math
import sys
import struct
import matplotlib
import argparse
import time
import alsaaudio
import random
import matplotlib.pyplot as plt
import thread

matplotlib.rcParams['lines.markersize'] = 6
matplotlib.rcParams['lines.markeredgewidth'] = 0
matplotlib.rcParams['font.size'] = 10

_freq_to_notes = {}
_notes_to_freq = {}
_notes = []
_note_sounds = {}

_NCHANNELS = 2
_SAMPLE_WIDTH = 2
_FRAME_RATE = 44100

def compute_note_freqs(period):
    C2_freq = 65.406
    note_freq_step = 2.0 ** (1.0 / 12) 
    shifted_freq = C2_freq
    i = 0
    for octave in range(2, 7):
        for note in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
            full_note = note + str(octave) 
            _notes.append(full_note)
            _freq_to_notes[shifted_freq] = full_note
            _notes_to_freq[full_note] = shifted_freq
            _note_sounds[full_note] = get_note_sound(full_note, period)
            #print "%4d" % i, "%4s" % freq_notes[shifted_freq], "%.3f" % shifted_freq
            shifted_freq *= note_freq_step
            i += 1

def play_tones(times, tones, play_choice, period):
    print "PLAY"
    pcm = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK)
    pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    pcm.setchannels(_NCHANNELS)
    pcm.setrate(_FRAME_RATE)
    pcm.setperiodsize(int(_FRAME_RATE * period))
    
    tstep = int(len(times) / (times[-1] / period))
    start_t = time.clock()
    prev_note = get_nearest_note(tones[0], 0, None)
    for i in range(1, len(times), tstep):
        note = get_nearest_note(tones[i], tones[i - tstep], prev_note)
        prev_note = note
        print "%-8.3f" % times[i], "%-3s" % note, "%8.0f" % _notes_to_freq[note], \
            "%-8.0f" % tones[i]

        if play_choice == "a": 
            pcm.write(get_freq_sound(tones[i], period))
        elif play_choice == "n": 
            pcm.write(_note_sounds[note])

    
def get_note_sound(note, period):
    freq = _notes_to_freq[note]
    return get_freq_sound(freq, period)


def get_freq_sound(freq, period):
    step = 2.0 * math.pi / (_FRAME_RATE / freq)
    max_val = 2.0 * math.pi
    wave_data = ""
    for x in numpy.arange(0, max_val, step):
        y = int((math.sin(x) + 1) * 32767.5)
        for c in range(_NCHANNELS): wave_data += (chr(y & 0xff) + chr((y>>8) & 0xff))
    nwaves = int(period * _FRAME_RATE * _NCHANNELS * _SAMPLE_WIDTH / len(wave_data))
    return wave_data * nwaves


def get_nearest_note(freq, prev_freq, prev_note):
    C2_freq = 65.406
    max_err = freq * 0.01
    index = int(round(12.0 * math.log(freq / C2_freq, 2)))
    note = _notes[index]
    if abs(freq - _notes_to_freq[note]) <= max_err: return note
    index = int(math.floor(12.0 * math.log(freq / C2_freq, 2)))
    return _notes[index]

    
def read_data(fname):
    start_t = time.clock()
    s = wave.open(fname)
    if s.getframerate() != _FRAME_RATE:
        print "Frame rate needs to be", _FRAME_RATE
        sys.exit(0)
    if s.getsampwidth() != _SAMPLE_WIDTH:
        print "Sample width needs to be", _SAMPLE_WIDTH
        sys.exit(0)
    if s.getnchannels() != _NCHANNELS:
        print "Number channels needs to be", _NCHANNELS
        sys.exit(0)

    nframes = s.getnframes()
    frames = s.readframes(nframes)
    s.close()

    data = []
    for i in range(0, len(frames), 2 * _NCHANNELS):
        ch_tot = 0.0
        for c in range(_NCHANNELS):
            ch_tot += struct.unpack("<h", frames[i + 2 * c] + frames[i + 1 + 2 * c])[0]
        data.append(ch_tot / 2)
    
    print "Read", fname +",", len(data), "data points", \
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
    parser.add_argument("-P", dest="play", default=None,
                        help="Play sample (a - actual, n - nearest notes)")
    parser.add_argument("-t", type=float, dest="period", default=0.1,
                        help="Output period in secs (default %(default).2f)")
    
    options = parser.parse_args()

    compute_note_freqs(options.period)

    print "Parameters: "
    for line in parser.format_help().split("\n"):
        if line == "": continue
        line = line.strip()
        if line[0] != "-": continue
        flag = line[0:2]
        if flag == "-h": continue
        if line[len(flag) + 1:len(flag) + 2] == " ": continue
        opt = line.split()[1]
        print " ", flag, "%-8s" % vars(options)[opt.lower()], \
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
    print "Pitch detection took %.3fs" % (time.clock() - start_t)
    plt.ylim((0, options.max_freq))
    if options.post_smooth_int > 0: 
        times, tones = median_smooth(times, tones, options.post_smooth_int)
    plt.plot(times, tones, marker="*", ls='', alpha=0.5, color="red")
    annotate(times, tones)
    # guitar frequencies
    for f in [82.4, 110.0, 146.8, 196.0, 246.9, 329.6, 1046.5]: 
        plt.axhline(y=f, color="black")

    if options.print_data:
        for i in range(0, len(times)): 
            print "%8d" % i, "%.4f" % times[i], "%8.0f" % tones[i]

    if options.play != None: 
        thread.start_new_thread(play_tones, (times, tones, options.play, options.period))
    plt.show()


if __name__ == "__main__": main()

 

 
