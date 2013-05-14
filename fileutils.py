#!/usr/bin/env python

from numpy import *

import chirpletringmod

import os.path
from scikits.audiolab import Sndfile
from scikits.audiolab import Format
from copy import deepcopy

def analysefile(path, hopsize=0.5, mode='ch', numtop=1, framesize = 1024, chrm_kwargs=None, maxdursecs=None):
	"""Analyses an audio file from disk, dividing into lapped frames and returning an array holding [raw, peaks, slopecent] for each frame.
	Can also do plain FFT-type analysis as an alternative."""
	if (mode != 'ch') and (mode != 'fft'):
		raise ValueError('Mode %s not recognised' % mode)
	if not os.path.isfile(path):
		raise ValueError("path %s not found" % path)
	sf = Sndfile(path, "r")
	if sf.channels != 1:
		raise Error("ERROR in chirpletringmod: sound file has multiple channels (%i) - mono audio required." % sf.channels)
	#print sf.format
	if maxdursecs!=None:
		maxdurspls = maxdursecs * sf.samplerate
	else:
		maxdurspls = sf.nframes

	if chrm_kwargs != None:
		chrm_kwargs = deepcopy(chrm_kwargs)
		chrm_kwargs['samplerate'] = sf.samplerate
		chrm_kwargs['framesize']  = framesize
	else:
		chrm_kwargs = {'samplerate':sf.samplerate, 'framesize':framesize}

	ch = chirpletringmod.Chirpletringmod(**chrm_kwargs)

	ihop = int(hopsize * ch.framesize)
	unhop = ch.framesize - ihop
	numspecframes = sf.nframes / ihop
	print "File contains %i spectral frames" % numspecframes
	storeraw = numspecframes < 500
	frames = []
	moretocome = True
	data = zeros(ch.framesize, float32)
	while(moretocome):
		try:
			nextdata = sf.read_frames(ihop, dtype=float32)
		except RuntimeError:
			#print "sf.read_frames runtime error, assuming EOF"
			moretocome = False
		if len(nextdata) != ihop:
			print "data truncated, detected EOF"
			moretocome = False
			nextdata = hstack((nextdata, zeros(ihop - len(nextdata))))
		data = hstack(( data[ihop:],  nextdata ))

		frames.append(ch.analyseframeplusfeatures(data, hopsize, mode, numtop, storeraw))

		if len(data) >= maxdurspls:
			break

	sf.close()
	return {'ch':ch, 'frames':frames, 'srate':sf.samplerate, 'hopsize':hopsize, 'framesize':ch.framesize}   # the ch knows srate and framesize, why are we duplicating?

def writeresynth(analysis, outpath, hopsize=0.5):
	"Writes a CSV file in format [duration,startfreq0,endfreq0,amp0,...]"
	dur = float(analysis['framesize'])*hopsize / analysis['srate']
	fp = open(outpath, "w")
	fp.write('dur')
	for i in range(len(analysis['frames'][0]['peaks'])):
		fp.write(',startfreq%i,endfreq%i,amp%i' % (i,i,i))
	fp.write('\n')
	for frame in analysis['frames']:
		fp.write('%g' % dur)
		for apeak in frame['peaks']:
			fromto = apeak['fromto']
			mag    = apeak['mag']
			fp.write(',%g,%g,%g' % (fromto[0], fromto[1], mag))
		fp.write('\n')
	fp.close()

def writeresynthaudio(ch, frames, outpath, hopsize=0.5):
	"Actually resynthesises the audio and writes that out"
	audiodata = ch.resynth([frame['peaks'] for frame in frames], hopsize)
	sf = Sndfile(outpath, "w", Format(), 1, ch.sr)
	sf.write_frames(audiodata)
	sf.close()

def decompose(inpath, outdir='output', niters=3, framesize=1024, hopsize=0.5, writefiles=True, wintype='hann'):
	"""Given a path to an input file, runs a pursuit iteration to decompose the signal into atoms.
	Writes out quite a lot of files - for each iteration, partial resynth, total resynth, residual.
	Also returns the aggregated peaks and the residual."""
	if not os.path.isfile(inpath):
		raise ValueError("path %s not found" % inpath)
	sf = Sndfile(inpath, "r")
	if sf.channels != 1:
		raise Error("ERROR in chirpletringmod: sound file has multiple channels (%i) - mono audio required." % sf.channels)

	ch = chirpletringmod.Chirpletringmod(samplerate=sf.samplerate, framesize=framesize, wintype=wintype)
	signal = sf.read_frames(sf.nframes, dtype=float32)
	sf.close()

	outnamestem = "%s/%s" % (outdir, os.path.splitext(os.path.basename(inpath))[0])

	resynthtot = zeros(len(signal))
	aggpeaks = []
	residual = signal
	print("chf.decompose: original signal energy %g" % sum(signal ** 2))
	for whichiter in range(niters):
		print("----------------------------------------")
		print("iteration %i" % whichiter)

		iterdata = ch.decompose_oneiter(residual, hopsize=hopsize)
		"""Given an input signal, decomposes it a bit like one round of matching-pursuit or suchlike, with the added constraint of
		one detection per frame. Returns the peaks found, the resynthesised version, and the residual."""
		#return {'peaks':framespeaks, 'resynth':resynth, 'residual':residual}

		resynthtot += iterdata['resynth']
		aggpeaks.extend(iterdata['peaks'])

		if writefiles:
			sf = Sndfile("%s_%i_resynth.wav" % (outnamestem, whichiter), "w", Format(), 1, ch.sr)
			sf.write_frames(iterdata['resynth'])
			sf.close()

			sf = Sndfile("%s_%i_resynthtot.wav" % (outnamestem, whichiter), "w", Format(), 1, ch.sr)
			sf.write_frames(resynthtot)
			sf.close()

			sf = Sndfile("%s_%i_residual.wav" % (outnamestem, whichiter), "w", Format(), 1, ch.sr)
			sf.write_frames(iterdata['residual'])
			sf.close()
		residual = iterdata['residual'] # fodder for next iter

		print("resynth    signal energy %g" % sum(iterdata['resynth'] ** 2))
		print("resynthtot signal energy %g" % sum(resynthtot ** 2))
		print("residual   signal energy %g" % sum(residual   ** 2))

	return {'ch':ch, 'peaks':aggpeaks, 'residual':residual}

