#!/usr/bin/env python

from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import chirpletringmod

#import os.path
from scikits.audiolab import Sndfile
from scikits.audiolab import Format

ch = chirpletringmod.Chirpletringmod(framesize=256)
#ch = chirpletringmod.Chirpletringmod(framesize=256, signalrange=(100,1000), proberange=(1000,1450)) # just for testing resynth

egsourcea = 2
egsourceb = (len(ch.breakpoints) - egsourcea) - 1

print("Example chirps used:")
print("   %i: from %g to %g" % (egsourcea, ch.breakpoints[egsourcea][0] * ch.bintofreq, ch.breakpoints[egsourcea][1] * ch.bintofreq))
print("   %i: from %g to %g" % (egsourceb, ch.breakpoints[egsourceb][0] * ch.bintofreq, ch.breakpoints[egsourceb][1] * ch.bintofreq))

if True:
	# single one plot:
	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.plot(real(ch.atoms[egsourcea]), 'k-')
	ax = fig.add_subplot(212)
	ax.plot(real(ch.atoms[egsourceb]), 'k-')
	#plt.title('Single atoms')
	#ax.set_axis_off()
	plt.savefig("plots/testchirps_real.pdf", papertype='A4', format='pdf')

	# plot the FFT magnitudes of them (plot FFT of the WHOLE complex thing at first, for analytic check. for diagram, we want real part only)
	ffta = abs(fft.fft(real(ch.atoms[egsourcea])))
	fftb = abs(fft.fft(real(ch.atoms[egsourceb])))
	fig = plt.figure()
	ax = fig.add_subplot(121)
	letsplot = 80
	ax.plot(ffta[:letsplot], list(range(letsplot)), 'k-')
	ax = fig.add_subplot(122)
	ax.plot(fftb[:letsplot], list(range(letsplot)), 'k-')
	plt.savefig("plots/testchirps_fft.pdf", papertype='A4', format='pdf')

"""
# write out an audio file containing a series of exact chirps
path = "audio/catchirp.wav"
sf = Sndfile(path, "w", Format(), 1, ch.sr)
#for index in range(0,len(ch.atoms),len(ch.atoms)/10):
for index in [egsourcea, egsourceb]:
	print "writing chirp #%i" % index
	## single one plot:
	#fig = plt.figure()
	#ax = fig.add_subplot(111)
	#ax.plot(real(ch.atoms[index]))
	#plt.title('Single atom for writing')
	#plt.show()
	sf.write_frames(real(ch.atoms[index]) * 10.)
sf.close()

# colormap plot:
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(real(ch.atoms), aspect='auto', cmap=cm.gray)
plt.title('Whole dictionary')
#plt.show()

####################################################################################
# Check performance when doing ringmod->fft->abs, by directly analysing a frame made as a real slice through a complex atom. the result should be strong DC resonance in our own bin
eg = ch.atoms[egsourcea].real * 0.7 + ch.atoms[egsourcea].imag * 0.3
#eg = eg + random.normal(0, 0.1, ch.framesize) # add some noise (optional)
ega = abs(ch.analyseframe(eg))
ch.plotchirpogram(ega, cmap=cm.gray_r) #, title='Results of analysing atom #%i' % egsourcea, cmap=cm.gray_r)
plt.savefig("plots/testchirps_chirpo1.pdf", papertype='A4', format='pdf')

eg = ch.atoms[egsourceb].real * 0.7 + ch.atoms[egsourceb].imag * 0.3
#eg = eg + random.normal(0, 0.1, ch.framesize) # add some noise (optional)
ega = abs(ch.analyseframe(eg))
ch.plotchirpogram(ega, cmap=cm.gray_r) #, title='Results of analysing atom #-%i' % egsourceb)
plt.savefig("plots/testchirps_chirpo2.pdf", papertype='A4', format='pdf')

####################################################################################
# check the binrange stuff
binrestrict = [ch.withinbandbinrange(ega, i) for i in range(shape(ega)[0])]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([x[0] for x in binrestrict], '*-')
ax.plot([x[1] for x in binrestrict], '*-')
plt.title('Upper and lower bin-restrictions for detection')

#plt.show()

#ch.frametopresults(ega, 15)    # post the winners

"""

####################################################################################
# analysing an atom, then resynthesising it, should recreate it
print("--------------------------------------------------------")
print(ch.breakpoints)

oneatom  = real(ch.atoms[len(ch.atoms)/2 - 4])
oneatom  = real(ch.atoms[4])
oneatom  = real(ch.make_complex_chirp(22, 22))
oneatom  = real(ch.make_complex_chirp(22, 31))
oneatom  = real(ch.make_complex_chirp(35, 18))
oneatom  = real(ch.make_complex_chirp(35, 18, 2))
oneatom  = real(ch.make_complex_chirp(15, 38, 1))
oneatom *= ch.window
ega = abs(ch.analyseframe(oneatom))
ch.plotchirpogram(ega, cmap=cm.gray_r) #, title='Results of analysing atom #%i' % egsourcea, cmap=cm.gray_r)

frameana = ch.analyseframeplusfeatures(oneatom, 0.5, 'ch', storeraw=False)
print(frameana)
remade   = ch.resynth([frameana['peaks']])

maxval_orig   = max(abs(oneatom))
maxval_remade = max(abs(remade ))
print("orig peak %g, remade peak %g (ratio %g)" % (maxval_orig, maxval_remade, maxval_remade / maxval_orig))
maxval = max(maxval_orig, maxval_remade)
fig = plt.figure()
#plt.title('Atom; resynthesised; diff')
ax = fig.add_subplot(311)
ax.plot(oneatom, 'k-')
plt.ylabel('Atom')
plt.ylim(-maxval, maxval)
ax = fig.add_subplot(312)
ax.plot(remade, 'k-')
plt.ylabel('Resynth')
plt.ylim(-maxval, maxval)
ax = fig.add_subplot(313)
ax.plot(oneatom-remade, 'k-')
plt.ylabel('Diff')
plt.ylim(-maxval, maxval)


plt.show()

