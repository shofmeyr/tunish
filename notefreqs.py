#!/usr/bin/python -u

import numpy
import math
import sys
import argparse
import time

class NoteFreqs:
    _notes_to_freq = {}
    _notes_to_midi = {}
    _notes = []
    _C2_FREQ = 65.406

    def __init__(self):
        note_freq_step = 2.0 ** (1.0 / 12) 
        shifted_freq = self._C2_FREQ
        i = 0
        for octave in range(2, 7):
            for note in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
                full_note = note + str(octave) 
                self._notes.append(full_note)
                self._notes_to_freq[full_note] = shifted_freq
                self._notes_to_midi[full_note] = i + 36
                shifted_freq *= note_freq_step
                i += 1

    def get_note_freq(self, note):
        return self._notes_to_freq[note]

    def get_all_notes(self):
        return self._notes

    def get_note_midi(self, note):
        return self._notes_to_midi[note]

    def get_nearest_note(self, freq):
        if freq < self._C2_FREQ: freq = self._C2_FREQ
        max_err = freq * 0.01
        index = int(round(12.0 * math.log(freq / self._C2_FREQ, 2)))
        note = self._notes[index]
        if abs(freq - self._notes_to_freq[note]) <= max_err: return note
        index = int(math.floor(12.0 * math.log(freq / self._C2_FREQ, 2)))
        return self._notes[index]



 

 
