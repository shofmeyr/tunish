#!/usr/bin/python -u

import time
import numpy
import fluidsynth
import alsaaudio

fl = fluidsynth.Synth()
sfid = fl.sfload("/usr/share/sounds/sf2/FluidR3_GM.sf2")

pcm = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK)
pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
pcm.setchannels(2)
pcm.setrate(44100)
pcm.setperiodsize(4410)

#for i in [16, 25, 41, 42, 43, 53, 54, 55, 74, 76]:
for i in [52, 53, 54, 55, 56]:
    s = []

    # Initial silence is 1 second
    s = numpy.append(s, fl.get_samples(44100 * 1))

    fl.program_select(0, sfid, 0, i - 1)

    fl.noteon(0, 60, 127)
    fl.noteon(0, 67, 127)
    fl.noteon(0, 76, 127)

    # Chord is held for 2 seconds
    s = numpy.append(s, fl.get_samples(int(44100 * 2)))

    fl.noteoff(0, 60)
    fl.noteoff(0, 67)
    fl.noteoff(0, 76)

    # Decay of chord is held for 1 second
    s = numpy.append(s, fl.get_samples(int(44100 * 1)))

    samps = fluidsynth.raw_audio_string(s)

    print len(samps)
    print 'Starting playback'

    pcm.write(samps)
                                                                               
fl.delete()

