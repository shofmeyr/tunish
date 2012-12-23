#!/usr/bin/python -u

import numpy
import math
import sys
import argparse
import time
import alsaaudio
import notefreqs
import fluidsynth
import numpy

class PlayTones:
    _note_freqs = None
    #_note_sounds = {}

    def __init__(self, nchannels=2, sample_width=2, frame_rate=44100, period=0.1, instrument=74):
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

        self.mixer = alsaaudio.Mixer()
        
        #for note in self._note_freqs.get_all_notes():
        #    self._note_sounds[note] = self.get_note_sound(note)
        self.synth = fluidsynth.Synth()
        sfid = self.synth.sfload("/usr/share/sounds/sf2/FluidR3_GM.sf2")
        # these come from the general MIDI Level 1 Instrument Patch Map
        # http://www.midi.org/techspecs/gm1sound.php#instrument
        choir_ahhs = 53
        voice_oohs = 54
        syn_choir = 55
        stell_guitar = 26
        violin = 41
        viola = 42
        cello = 43
        flute = 74
        self.synth.program_select(0, sfid, 0, instrument - 1)

    def play_tone(self, tone, duration):
        self.mixer.setvolume(20)
        for i in range(0, int(duration / self.period)):
            self.pcm.write(self.get_freq_sound(tone))

    def play_note(self, note, duration):
        self.mixer.setvolume(100)
        s = []
        # self.synth.noteon(0, 60, 127)
        # s = numpy.append(s, self.synth.get_samples(int(44100.0 * self.period / 2)))
        # self.synth.noteoff(0, 60)
        # s = numpy.append(s, self.synth.get_samples(int(44100.0 * self.period / 2)))
        # self.pcm.write(fluidsynth.raw_audio_string(s))

        self.synth.noteon(0, self._note_freqs.get_note_midi(note), 127)
        s = numpy.append(s, self.synth.get_samples(int(44100 * duration)))
        self.synth.noteoff(0, self._note_freqs.get_note_midi(note))
        #s = numpy.append(s, self.synth.get_samples(int(44100 * self.period)))
        s = numpy.append(s, self.synth.get_samples(1))
        self.pcm.write(fluidsynth.raw_audio_string(s))
        #self.pcm.write(self._note_sounds[note])
        
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
        nwaves = int(self.period * self.frame_rate * self.nchannels * 
                     self.sample_width / len(wave_data))
        return wave_data * nwaves

def main():
    parser = argparse.ArgumentParser(description="Play notes in file. Format: float_time_in_secs note")
    parser.add_argument("-P", dest="play", default="n",
                        help="Play sample (a - actual, n - nearest notes)")
    parser.add_argument("-t", type=float, dest="period", default=0.125,
                        help="Output period in secs (default %(default).2f)")
    parser.add_argument("-i", type=int, dest="instrument", default=1,
                        help="Instrument  (default %(default)d)")
    parser.add_argument('-f', '--input-file', type=argparse.FileType('r'), default='-', dest="f")

    options = parser.parse_args()

    play_tones = PlayTones(nchannels=2, sample_width=2, frame_rate=44100, period=options.period,
                           instrument=options.instrument)

    prev_t = None
    while True:
        print "\b\b\b\b\b\b\b\b\b\b",
        line = sys.stdin.readline()
        if "fin" in line: return
        if line == "": continue
        t, note, duration, freq = line.strip().split()[0:4]
        t = float(t)
        duration = float(duration)
        freq = float(freq)
        if options.play == "a": play_tones.play_tone(freq, duration)
        elif options.play == "n": play_tones.play_note(note, duration)
        print "%8.3f" % t,

if __name__ == "__main__": main()

 

 
