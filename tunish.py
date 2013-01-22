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
matplotlib.rcParams['grid.linestyle'] = '-'
matplotlib.rcParams['grid.color'] = '#aaaaaa'
matplotlib.rcParams['grid.linewidth'] = 0.5
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['figure.figsize'] = [16.8, 8.4]


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


def get_note_list(times, tones, period, err_skip_level, max_freq, min_freq):
    """
    This function takes a sequuence of tones and converts them into a string of notes.
    The idea is to select the notes as close as possible to the tones, without 
    changing the intervals between the tones too much, i.e. we don't want to push 
    two tones that are close together to different notes by rounding.
    The algorithm here is we first try to shift the tones to minimize the error, then
    we go tone by tone, rounding to the nearest note, and dropping those tones that are 
    too far from a note. To find the best shift, we have to try multiple shifts, measure
    the error in each, and then select the best at the end.
    """
    note_freqs = notefreqs.NoteFreqs()
    min_err = 10000
    min_err_shift = 0
    for shift in numpy.arange(1.0, 2.1, 0.1):
        tot_err = 0.0
        prev_t = times[0]
        for i in range(0, len(times)):
            if tones[i] == 0: continue
            if tones[i] > max_freq or tones[i] < min_freq: continue
            if times[i] < prev_t + period: continue
            tone = tones[i] * shift
            note, err = note_freqs.get_nearest_note(tone)
            tot_err += err
            prev_t = times[i]
        tot_err /= len(times)
        if tot_err < min_err:
            min_err = tot_err
            min_err_shift = shift
            

    prev_note = note_freqs.get_nearest_note(tones[0] + min_err_shift)[0]
    prev_t = times[0]
    prev_tone = tones[0]
    note_times = []
    notes = []
    orig_tones = []
    note_durations = []
    note_tones = []
    err_skipped = 0
    for i in range(1, len(times)):
        if tones[i] == 0: continue
        if tones[i] > max_freq: continue
        if times[i] < prev_t + period: continue
        note, err = note_freqs.get_nearest_note(tones[i] + min_err_shift)
        if prev_note == note: continue
        if err > err_skip_level:
            err_skipped += 1
            continue
        note_times.append(prev_t)
        notes.append(prev_note)
        orig_tones.append(prev_tone)
        note_durations.append(times[i] - prev_t)
        note_tones.append(note_freqs.get_note_freq(prev_note))
        prev_note = note
        prev_t = times[i]
        prev_tone = tones[i]
    print >>sys.stderr, "Skipped", err_skipped, "error points out of", (len(notes) + err_skipped)
    return note_times, notes, note_durations, orig_tones, note_tones


def main():
    parser = argparse.ArgumentParser(description="Find pitch in wav file.")
    parser.add_argument(dest= "fnames", metavar="file", nargs=1,
                        help="A wav file to be processed")
    parser.add_argument("-p", action="store_true", dest="plot_wave", default=False,
                        help="Plot wave")       
    parser.add_argument("-o", action="store_true", dest="print_notes", default=False,
                        help="Print list of notes")       
    parser.add_argument("-d", type=int, dest="db_lim", default=25,
                        help="Decibel limit for noise (default %(default)d)")
    parser.add_argument("-F", type=float, dest="max_freq", default=325, 
                        help="Max frequency for tones (default %(default)d)")
    parser.add_argument("-f", type=float, dest="min_freq", default=90, 
                        help="Min frequency for tones (default %(default)d)")
    parser.add_argument("-s", type=float, dest="post_smooth_int", default=0.1, 
                        help="Median-smoothing interval in secs (default %(default).2f)")
    parser.add_argument("-r", type=float, dest="max_peak_ratio", default=0.95,
                        help="Max. peak ratio (default %(default).2f)")
    parser.add_argument("-z", type=float, dest="pre_smooth_wind", default=100, 
                        help="Pre-smoothing window (default %(default)d)")
    parser.add_argument("-t", type=float, dest="period", default=0.125,
                        help="Output period in secs (default %(default).2f)")
    parser.add_argument("-e", type=float, dest="err_skip_level", default=0.02,
                        help="Mismatch level for tone skip (default %(default).2f)")
    
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

    plt.semilogy(times, tones, marker="o", ls='', alpha=0.5, color="red")

    times, notes, durations, tones, note_tones = get_note_list(times, tones, options.period, 
                                                               options.err_skip_level, 
                                                               options.max_freq, options.min_freq)

    note_freqs = notefreqs.NoteFreqs()
    all_notes = note_freqs.get_all_notes()
    all_freqs = [note_freqs.get_note_freq(note) for note in all_notes]
    gtabs = [note_freqs.get_gtab_note(note) for note in all_notes]

    for i in range(len(all_notes)):
        if all_freqs[i] < options.min_freq - 10: continue
        if all_freqs[i] > options.max_freq + 10: continue
        plt.text(times[-1] + 1.1, all_freqs[i], gtabs[i], color="blue")
                     
    plt.yticks(all_freqs, all_notes)
    plt.xticks(numpy.arange(0, times[-1], 1))
    plt.axis([0, times[-1] + 1, options.min_freq - 10, options.max_freq + 10])
    plt.grid(True, "major")
    plt.xlabel("Time (s)")
    plt.ylabel("Note")
    plt.title(options.fnames[0])

    if options.print_notes:
        # notes = all_notes
        # times = numpy.arange(0, len(notes) * 0.125, 0.125)
        # durations = [0.125] * len(notes)
        # tones = all_freqs
        # note_tones = all_freqs

        for i in range(len(times)):
        #for i in range(28):
            print "%10.3f" % times[i], "%5s" % notes[i], "%8.3f" % durations[i], \
                "%8.0f" % tones[i], "%8.0f" % note_tones[i], note_freqs.get_gtab_note(notes[i])
        print "fin"

    plt.show()




if __name__ == "__main__": main()

 

 
