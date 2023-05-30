#!/usr/bin/env python

from numpy import *
import numpy as np   # transitioning over to this form
from scipy.stats import scoreatpercentile
import scipy.version  # for checking its version, gah

import chirpletringmod

import os.path
import librosa
from copy import deepcopy
import tempfile
import subprocess
import shutil

def analysefile(path, hopsize=0.5, mode='ch', numtop=1, sr=None, framesize = 1024, chrm_kwargs=None, maxdursecs=None, storeraw=False, addphase=False, ch=None, startpossecs=0):
	"""Analyses an audio file from disk, dividing into lapped frames and returning an analysis for each frame.
	Can also do plain FFT-type analysis as an alternative.

import fileutils as chf
import numpy as np
fname = "blackbird-long-edited_ZOOM0001_LR_Leiden_20220522_2102_blackbird_rooftop_long"
indir = "/home/dans/birdsong/dan_recordings/"
outdir = "/home/dans/Documents/tilnat/aprojects/timeseries_bbrb"
maxdursecs = 60
#maxdursecs = None
ana = chf.analysefile(f"{indir}/{fname}.wav", chrm_kwargs={'signalrange':(1000,10000), 'probedepth':1000}, maxdursecs=maxdursecs)

srate = ana['srate']
hopspls = int(ana['framesize'] * ana['hopsize'])
hopsecs = float(hopspls) / srate
framesizesecs = float(ana['framesize']) / srate
print(ana['frames'][0]['peaks'][0].keys())        # ['bin', 'fromto', 'mag', 'atom']
with open(f'{outdir}/ana_%s.csv' % fname, 'wt') as fp:
	fp.write('time,freq,slope,mag\n')
	for whichframe, frame in enumerate(ana['frames']):
		peak = frame['peaks'][0]
		freq = np.mean(peak['fromto'])
		if True: #freq>1000 and peak['mag'] > 0.1:
			fp.write('%g,%g,%g,%g\n' % (whichframe * hopsecs, freq, (peak['fromto'][1]-peak['fromto'][0])/framesizesecs, peak['mag']))

	"""
	if (mode != 'ch') and (mode != 'fft'):
		raise ValueError('Mode %s not recognised' % mode)
	if not os.path.isfile(path):
		raise ValueError("path %s not found" % path)

	sr = librosa.get_samplerate(path)
	#audiodata, sr = librosa.load(path, mono=True, sr=sr, offset=startpossecs, duration=maxdursecs)

	if ch==None:   # by default we are intending to create our own internal ch instance. Or you may have one to pass in from outside.
		if chrm_kwargs != None:
			chrm_kwargs = deepcopy(chrm_kwargs)
			chrm_kwargs['samplerate'] = sr
			chrm_kwargs['framesize']  = framesize
		else:
			chrm_kwargs = {'samplerate':sr, 'framesize':framesize}

		ch = chirpletringmod.Chirpletringmod(**chrm_kwargs)

	ihop = int(hopsize * ch.framesize)

	audiodatastream = librosa.stream(path, mono=True, block_length=1, frame_length=framesize, hop_length=ihop, fill_value=0, offset=startpossecs, duration=maxdursecs)

	if storeraw == None:
		storeraw = True
	frames = []
	for y_block in audiodatastream:
		data = y_block
		storeraw_tmp = storeraw || (storeraw is None and len(frames) < 500)
		frames.append(ch.analyseframeplusfeatures(data, hopsize, mode, numtop, storeraw_tmp, addphase))

	return {'ch':ch, 'frames':frames, 'srate':sr, 'hopsize':hopsize, 'framesize':ch.framesize}   # the ch knows srate and framesize, why are we duplicating?

"""
import fileutils as chf
import plots
import matplotlib.pyplot as plt
from copy import deepcopy
item = chf.analysefile('/home/dan/birdsong/nips4b/examples/nips4b2013_birds_file_0001.wav', storeraw=True)
itemr = deepcopy(item)
chf.noisereduce(itemr, 99)

plt.figure()
fig = plots.plot_chirped(item, 44100.0)
plt.title("Raw")
plt.figure()
fig = plots.plot_chirped(itemr, 44100.0)
plt.title("Noisereduced")
plt.show()

"""
def noisereduce(ana, percentile=50, storeraw=False, downsample=False):
	"""Given the data from analysefile(), this applies a median-based stationary noise removal.
	It applies the procedure in-place -- if you need the original, take a deepcopy of it.
	Note that some of the peaks will have magnitude zero as a result of this - we do not delete them from the data structure.
	The "downsample" argument is used to speed up the percentile calculation by approximation - e.g. set it to 10."""
	if not 'raw' in ana['frames'][0]:
		raise ValueError("a list of frames was passed in without 'raw' data in them. noisereduce() requires data produced using 'storeraw=True'.")

	thedata = [frame['raw'] for frame in ana['frames']]
	if downsample:
		thedata = thedata[::downsample]

	# rather than running one single invocation of scoreatpercentile, it takes less memory if we handle each pixel individually.
	# it also works around scipy issue gh-3329
	shape = np.shape(thedata[0])
	rawmedian = np.zeros(shape)
	for xpos in range(shape[0]):
		for ypos in range(shape[1]):
			rawmedian[xpos,ypos] = scoreatpercentile([frame[xpos,ypos] for frame in thedata], percentile)

	ch = ana['ch']
	numtop = len(ana['frames'][0]['peaks'])
	hopsize = ana['hopsize']
	for frame in ana['frames']:
		# subtract the median from each frame, zeroing anything below the median
		frame['raw'] = np.fmax(frame['raw'] - rawmedian, 0)
		# and recalc the peaks
		frame['peaks'] = ch.frametopresults(frame['raw'], numtop, hopsize)
	return None    # in-place

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
	raise NotImplementedError
	audiodata = ch.resynth([frame['peaks'] for frame in frames], hopsize)
	sf = Sndfile(outpath, "w", Format(), 1, ch.sr)
	sf.write_frames(audiodata)
	sf.close()

def decompose(inpath, outdir='output', niters=3, framesize=1024, hopsize=0.5, writefiles=True, wintype='hann'):
	"""Given a path to an input file, runs a pursuit iteration to decompose the signal into atoms.
	Writes out quite a lot of files - for each iteration, partial resynth, total resynth, residual.
	Also returns the aggregated peaks and the residual."""
	raise NotImplementedError
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
	print(("chf.decompose: original signal energy %g" % sum(signal ** 2)))
	for whichiter in range(niters):
		print("----------------------------------------")
		print(("iteration %i" % whichiter))

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

		print(("resynth    signal energy %g" % sum(iterdata['resynth'] ** 2)))
		print(("resynthtot signal energy %g" % sum(resynthtot ** 2)))
		print(("residual   signal energy %g" % sum(residual   ** 2)))

	return {'ch':ch, 'peaks':aggpeaks, 'residual':residual}

def chirposcope(data, outpath, audioinpath=None, cmap=None):
	"""Given data as returned by analysefile(), this writes out every frame's chirpogram as an image then invokes ffmpeg to turn them into an animation.
	"audioinpath" arg is used only if you want the audiofile mixed back in to the animation.

python -c "import fileutils as chf; audiopath = '/home/dan/birdsong/phylloscopus-xenocanto/wav/part3/Phylloscopus_collybita/XC125867.mp3.wav'; data = chf.analysefile(audiopath, storeraw=True, maxdursecs=5); chf.chirposcope(data, 'chirposcope_chch.mp4', audiopath)"
python -c "import fileutils as chf; audiopath = '/home/dan/birdsong/skylarkbriefer/EL1.mp3.wav'; data = chf.analysefile(audiopath, storeraw=True, maxdursecs=5); chf.chirposcope(data, 'chirposcope_skyl.mp4', audiopath)"
python -c "import fileutils as chf; audiopath = '/home/dan/Hideo/rennes201303/20130317_b_aviary/rennes_aviary_birdsong_20130317_1double.wav'; data = chf.analysefile(audiopath, storeraw=True, maxdursecs=10); chf.chirposcope(data, 'chirposcope_aviary1.mp4', audiopath)"
	"""
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	import matplotlib.colors
	if cmap == None: cmap = cm.bone  # bone, gist_heat
	# create a tmp folder for all our pics
	tmpdir = tempfile.mkdtemp('_chirposcope')
	print("Temp folder: %s" % tmpdir)

	timestretchratio = 0.25

	# find max mag in each frame
	maxmags = [max([peak['mag'] for peak in frame['peaks']]) for frame in data['frames']]
	# find the overall maximum value, to be used as rangemax
	maxval = max(maxmags)
	print("Maximum value found is %g" % maxval)

	# decide the frame rate
	framerate = data['ch'].sr / (data['ch'].framesize * data['hopsize'])
	print("Frame rate found is %g" % framerate)
	# find out the max dur (so we know when to truncate the ffmpeg result)
	durationsecs = len(data['frames']) / framerate

	chirpoframestem = '%s/chirpoframe' % tmpdir

	# home-made colourmap (01pos, valfrombelow, valfromabove)
	midpos = 0.1
	toppos = 0.9
	cdict = {'red': ((0.0,    0.0, 0.0),
		         (midpos, 0.9, 0.9),
		         (toppos, 1.0, 1.0),
		         (1.0,    1.0, 1.0)),
		 'green': ((0.0,    0.0, 0.0),
		           (midpos, 1.0, 1.0),
		           (toppos, 1.0, 1.0),
		           (1.0,    1.0, 1.0)),
		 'blue': ((0.0,    0.0, 0.0),
		          (midpos, 0.5, 0.5),
		          (toppos, 1.0, 1.0),
		          (1.0,    1.0, 1.0))}
	cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

	normalisewindow = int(framerate * 0.5)   # we'll not use overall normalisation, but relative to a 1sec window

	# foreach frame
	for whichframe, frame in enumerate(data['frames']):
		if (whichframe % 100) == 0:
			print("Plotting frame %i of %i" % (whichframe, len(data['frames'])))
		windowstart = max(0, whichframe-normalisewindow)
		localmaxval = max(maxmags[windowstart:min(windowstart+normalisewindow+normalisewindow, len(data['frames']))])
		localmaxval = (localmaxval * 0.75) + (maxval * 0.25)
		fig = data['ch'].plotchirpogram(frame['raw'], title='', rangemax=localmaxval, cmap=cmap, vlinecol=(0.15, 0.15, 0.15), bare=True)
		# LATER: a stripped-down mode for the plot
		# write to a PNG file with the 000001 naming convention needed by ffmpeg
		plt.savefig('%s_%0.6i.png' % (chirpoframestem, whichframe), papertype='A4', format='png', bbox_inches="tight")
		# ensure the plots don't pile up in memory
		plt.clf()
		plt.close()

	# first we'll prepare a timestretch audio file, to make sure it's available later
	timestretchaudiopath = "%s/timestretched.wav" % tmpdir
	cmd = ["sox", audioinpath, timestretchaudiopath, "tempo", "-m", str(timestretchratio)]
	print(" ".join(cmd))
	subprocess.call(cmd)

	# invoke ffmpeg
	cmd = ["ffmpeg", "-y", "-v", "-10", "-r", str(framerate), "-b", "2048k", "-i", '%s_%%06d.png' % chirpoframestem]
	if audioinpath != None:
		cmd.extend(["-i", audioinpath])
	cmd.extend(['-t', str(durationsecs), '%s' % outpath])
	print(" ".join(cmd))
	subprocess.call(cmd)

	# and a timestretch version too
	timestretchoutpath = '-timestretch'.join(os.path.splitext(outpath))
	cmd = ["ffmpeg", "-y", "-v", "-10", "-r", str(framerate * timestretchratio), "-b", "2048k", "-i", '%s_%%06d.png' % chirpoframestem]
	if audioinpath != None:
		cmd.extend(["-i", timestretchaudiopath])
	cmd.extend(['-t', str(durationsecs / timestretchratio), '%s' % timestretchoutpath])
	print(" ".join(cmd))
	subprocess.call(cmd)


	print("mplayer -fs '%s'" % outpath)
	print("mplayer -fs '%s'" % timestretchoutpath)
	print("vlc --rate 1 -f '%s'" % outpath)
	print("vlc --rate 0.25 -f '%s'" % outpath)
	print("vlc -f '%s'" % timestretchoutpath)

	# tidy up after ourselves
	shutil.rmtree(tmpdir)

