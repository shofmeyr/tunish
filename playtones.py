#!/usr/bin/python -u

import numpy
import math
import sys
import argparse
import time
import alsaaudio
import notefreqs

class PlayTones:
    _note_freqs = None
    _note_sounds = {}

    def __init__(self, nchannels=2, sample_width=2, frame_rate=44100, period=0.1):
        self.nchannels = nchannels
        self.sample_width = sample_width
        self.frame_rate = frame_rate
        self.period = period
        self._note_freqs = notefreqs.NoteFreqs()

        self.pcm = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK)
        self.pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        self.pcm.setchannels(self.nchannels)
        self.pcm.setrate(self.frame_rate)
        self.pcm.setperiodsize(int(self.frame_rate * self.period))

        for note in self._note_freqs.get_all_notes():
            self._note_sounds[note] = self.get_note_sound(note)

    def play_tone(self, tone):
        self.pcm.write(self.get_freq_sound(tone))

    def play_note(self, note):
        self.pcm.write(self._note_sounds[note])
        
    def get_note_sound(self, note):
        freq = self._note_freqs.get_note_freq(note)
        return self.get_freq_sound(freq)

    def get_freq_sound(self, freq):
        step = 2.0 * math.pi / (self.frame_rate / freq)
        max_val = 2.0 * math.pi
        wave_data = ""
        for x in numpy.arange(0, max_val, step):
            y = int((math.sin(x) + 1) * 32767.5)
            for c in range(self.nchannels): wave_data += (chr(y & 0xff) + chr((y>>8) & 0xff))
        nwaves = int(self.period * self.frame_rate * self.nchannels * self.sample_width / len(wave_data))
        return wave_data * nwaves

def main():
    parser = argparse.ArgumentParser(description="Play notes in file. Format: float_time_in_secs note")
    parser.add_argument("-P", dest="play", default="n",
                        help="Play sample (a - actual, n - nearest notes)")
    parser.add_argument("-t", type=float, dest="period", default=0.1,
                        help="Output period in secs (default %(default).2f)")
    parser.add_argument('-f', '--input-file', type=argparse.FileType('r'), default='-', dest="f")

    options = parser.parse_args()

    play_tones = PlayTones(nchannels=2, sample_width=2, frame_rate=44100, period=options.period)

    prev_t = None
    while True:
        line = sys.stdin.readline()
        if line == "": continue
        t, note, freq = line.strip().split()[0:3]
        t = float(t)
        freq = float(freq)
        if prev_t != None:
            if t < prev_t + options.period: continue
            if t > prev_t + options.period: time.sleep(t - prev_t - options.period)
        prev_t = t
        if options.play == "a": play_tones.play_tone(freq)
        elif options.play == "n": play_tones.play_note(note)

if __name__ == "__main__": main()

 

 
