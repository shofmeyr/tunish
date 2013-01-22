#!/usr/bin/python -u

import numpy
import math
import sys
import argparse
import time

class NoteFreqs:
    def __init__(self):
        self._notes_to_freq = {}
        self._notes_to_midi = {}
        self._notes = []
        self._E2_FREQ = 82.407

        note_freq_step = 2.0 ** (1.0 / 12) 
        shifted_freq = self._E2_FREQ
        i = 0
        for octave in range(2, 7):
            if octave == 2: notes = ["E", "F", "F#", "G", "G#", "A", "A#", "B"]
            else: notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            for note in notes:
                full_note = note + str(octave) 
                self._notes.append(full_note)
                self._notes_to_freq[full_note] = shifted_freq
                self._notes_to_midi[full_note] = i + 28
                shifted_freq *= note_freq_step
                i += 1

    def get_note_freq(self, note):
        return self._notes_to_freq[note]

    def get_gtab_note(self, note):
        index = self._notes.index(note)
        if index < 15: return "%d/%d" % (int(index / 5) + 1, index % 5)
        elif 15 <= index < 19: return "4/%d" % (index - 15)
        elif 19 <= index < 24: return "5/%d" % (index - 19)
        else: return "6/%d" % (index - 24)

    def get_all_notes(self):
        return self._notes
    
    def get_note_midi(self, note):
        return self._notes_to_midi[note]

    def get_nearest_note(self, freq):
        if freq < self._E2_FREQ: freq = self._E2_FREQ
        index = int(round(12.0 * math.log(freq / self._E2_FREQ, 2)))
        note = self._notes[index]
        err = abs(self.get_note_freq(note) - freq) / freq
        return note, err



 

 
