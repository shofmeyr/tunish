#!/usr/bin/python -u

import alsaaudio
import wave
import numpy
import argparse

def main():
    parser = argparse.ArgumentParser(description="Record to wav file")
    parser.add_argument("-o", dest="output_file", default="record.wav",
                        help="Output wav file (default %(default)s)")
    parser.add_argument("-d", dest="db_lim", type=int, default=25,
                        help="dB limit (default %(default)d)")
    options = parser.parse_args()

    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
    inp.setchannels(1)
    inp.setrate(44100)
    inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    inp.setperiodsize(1024)

    w = wave.open(options.output_file, 'w')
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(44100)

    ampl_lim_max = 10.0 ** (options.db_lim / 10.0)

    while True:
        l, data = inp.read()
        a = numpy.fromstring(data, dtype='int16')
        amplitude = numpy.abs(a).mean()
        if amplitude > ampl_lim_max: w.writeframes(data)

if __name__ == "__main__": main()

