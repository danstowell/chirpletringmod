#!/usr/bin/env python

# framed chirplet analysis facilitated by ring-modulation
# (c) Dan Stowell 2011--2013

from numpy import *
from operator import itemgetter

class Chirpletringmod:

	def __init__(self, samplerate = 44100.0, framesize = 1024, signalrange=(2000,8000), proberange=(6000,10000), wintype='modified'):
		self.sr = float(samplerate)
		self.framesize = framesize#256 #512 in original devt; 1024 seems good for birdsong, so far
		self.bintofreq = self.sr/self.framesize
		self.freqtobin = float(self.framesize)/self.sr
		framehalf = self.framesize / 2
		######################################################################################
		# Definitions of signal range. For example if probes are 6--10 kHz (all centred on 8) and the signal is expected in range 2--8 kHz, we detect in region 0--6
		signallobin = int( ceil(signalrange[0] * self.freqtobin)) # 2000
		signalhibin = int(floor(signalrange[1] * self.freqtobin)) # 8000
		probelobin  = int(floor( proberange[0] * self.freqtobin)) # 6000
		probehibin  = int( ceil( proberange[1] * self.freqtobin)) #10000
		probemiddlebin = (probehibin+probelobin)/2
		self.detectlobin = probemiddlebin - signalhibin
		self.detecthibin = probemiddlebin - signallobin
		print 'chipletringmod probetone range: bins %i to %i, freq range %g to %g' \
					% (probelobin, probehibin, probelobin * self.bintofreq, probehibin * self.bintofreq)
		print '   expecting signal in bin range %i to %i' % (signallobin, signalhibin)
		print '     thus detecting in bin range %i to %i' % (self.detectlobin, self.detecthibin)
		probebinrange = range(probelobin, probehibin)
		self.halfprobebinspan = (probehibin-probelobin)/2
		halfprobebinrange = range(self.halfprobebinspan)
		################################ choice of window:
		if wintype=='hann':
			window = hanning(self.framesize)
		elif wintype=='rect':
			window = ones(self.framesize)
		else: # 'modified' window is default
			window = hstack((hanning(self.framesize/2)[:self.framesize/4], ones(self.framesize/2), hanning(self.framesize/2)[self.framesize/4:]))
		self.window = window

		# Here we create our short-time chirp basis:
		self.breakpoints = [ (probelobin+delta, probehibin-delta) for delta in halfprobebinrange ] \
			    + [(probemiddlebin, probemiddlebin)] \
			    + [ (probehibin-delta, probelobin+delta) for delta in halfprobebinrange[-1::-1] ]
		print "num breakpoints is %i" % len(self.breakpoints)
		self.atoms = array([self.make_complex_chirp(bp[0], bp[1]) for bp in self.breakpoints])

	def make_complex_chirp(self, lobin, hibin, phaseoffset=0.):
		"Creates a single complex chirp"
		xs = arange(self.framesize, dtype=float)
		framehalf = self.framesize / 2
		# Here we're actually generating an array of phase-position-per-sample, which we will next convert to onde.
		phases = 2 * pi * (xs-framehalf) * (lobin  + ((hibin-lobin)*xs/self.framesize)) / self.framesize
		# enforce phase always starting at zero
		phases += phaseoffset - phases[0]
		# Make the complex-valued ('spiral') chirp
		atom = cos(phases) + sin(phases) * 1j
		#moved windowing into analyse()      atom = atom * self.window
		# and L2-normalise
		atom = atom / (sqrt(sum(abs(atom) ** 2)))
		return atom

	######################################################################################
	# Main calculations

	def analyseframe(self, frame):
		"frame must be of size 'self.framesize'. Do not window it, that's built in to the dictionary. returns [atom * fakefreq] result matrix of COMPLEX vals"
		return fft.fft((frame * self.window) * self.atoms)[:,self.detectlobin:self.detecthibin]

	def withinbandbinrange(self, chirpogram, colindex):
		"""The rectangle of FFT results contains some slopey ones that extend outside the specified freq range. 
		This function returns a [lo,hi) BINrange for a column, so that you can iterate only the within-band bins."""
		chshape = shape(chirpogram)
		pos = abs(colindex - ((chshape[0]-1)/2))
		return (pos, chshape[1] - pos)

	def revmap(self, atomindex, fakebin, hopsize=1.):
		"""reverse-map: if for atom Q we find a resonance at fakebin F, what's the corresponding start and end frequency?
		NB if the hopsize is e.g. 0.5 then we can pull the estimate inwards so the breakpoints match up with the previous frame."""
		# since the detectbin is always (probe-actualfreq), we can find the actualfreq as (probe-detectbin)
		return ((self.breakpoints[atomindex][0] - fakebin) * self.bintofreq, \
				(self.breakpoints[atomindex][1] - fakebin) * self.bintofreq)

	def frametopresults(self, chirpogram, n, hopsize=1., atomindexoffset=0):
		"""takes an [atom * fakefreq] result, and returns the top N chirplet peaks, each in the form {atom, fakebin, fromto:(from, to), mag}.
		May return fewer than N if there aren't that many peaks.

		OPTIMISING: Note that this function takes a large portion of the CPU time when analysing files.
		I have hand-optimised this to avoid list generation and to reduce the number of sort operations; still quite heavy."""
		results = [{'mag':0.0}] * n
		bigger_n = max(n, 10) * 10  # reduce the amount of times we sort
		lowest_mag_kept = -99999
		# Here we iterate, maintaining a sorted list of results and adding if we find a winning bin
		# Note: I reworked this to use the "iterate" method and it slooooowwwwed right down, so don't.
		for index in xrange(shape(chirpogram)[0]):
			for whichbin in xrange(*self.withinbandbinrange(chirpogram, index)):
				currmag = chirpogram[index,whichbin]
				if (currmag > lowest_mag_kept)  \
				     and (n==1 or self.isbinlocalpeak(chirpogram, index, whichbin)):
					# construct result item
					results.append((currmag, index, whichbin)) # NB tuple with mag as first item means that mag is used for sort
					if len(results) > bigger_n:
						results.sort(reverse=True)
						results = results[:n]
						lowest_mag_kept = results[-1][0]
		results.sort(reverse=True)
		results = results[:n]
		# Finally we construct the "fromto" information and the dict-style results (rather than wastefully doing it inside the loop and discarding many)
		results = [{'mag':res[0], 'atom':res[1], 'bin':res[2], \
			'fromto':self.revmap(res[1]+atomindexoffset, res[2], hopsize)} for res in results]
		return results

	def frametopresults_addphase(self, results, phasematrix):
		"After invoking frametopresults(), you can use this to add phase info to the list"
		for oneresult in results:
			oneresult['phase'] = phasematrix[oneresult['atom'], oneresult['bin']]

	def isbinlocalpeak(self, chirpogram, index, whichbin):
		"returns true if index & whichbin refer to a local peak"
		curval = chirpogram[index][whichbin]
		return not \
			(   ((index != 0                )       and chirpogram[index-1][whichbin] > curval)
			 or ((index != len(chirpogram)-1)       and chirpogram[index+1][whichbin] > curval)
			 or ((whichbin != 0)                    and chirpogram[index][whichbin-1] > curval)
			 or ((whichbin != len(chirpogram[0])-1) and chirpogram[index][whichbin+1] > curval))


	def iteratechirpogramdiamond(self, chirpogram, state, foreachslice, foreachbin):
		"iterates over the 'diamond' of bins that represents the signal band of interest (i.e. the slopier it is, the fewer bins are included)"
		for index, gramslice in enumerate(chirpogram):
			foreachslice(state, index)
			for whichbin in xrange(*self.withinbandbinrange(chirpogram, index)):
				foreachbin(state, index, whichbin, gramslice[whichbin])

	######################################################################################
	# derived features

	def slopeCentroid(self, chirpogram):
		"finds the slope-centroid of the frame, which yields a measure of how uppy or downy it is"
		state = {'total': 0., 'normaliser': 0., 'slopeval': 0}
		def foreachslice(state, index):
			"initialise the slope value"
			state['slopeval'] = index - self.halfprobebinspan
		def foreachbin(state, index, whichbin, binval):
			state['total'] += (binval * state['slopeval'])
			state['normaliser'] += binval
		self.iteratechirpogramdiamond(chirpogram, state, foreachslice, foreachbin)
		return state['total'] / state['normaliser']

	def specFlatness(self, chirpogram):
		state = {'numer': 0., 'denom': 0., 'n': 0}
		def noop(state, index):
			pass
		def foreachbin(state, index, whichbin, binval):
			state['numer'] += log(binval)
			state['denom'] += binval
			state['n']     += 1
		self.iteratechirpogramdiamond(chirpogram, state, noop, foreachbin)
		divider = 1. / state['n']
		return exp(state['numer'] * divider) / (state['denom'] * divider)

	def medianfilter(self, data, halfnum=10):
		return array([median(data[max(0,i-halfnum):i+halfnum+1]) for i in xrange(len(data))])

	def histogramBigram(self, frames, binsperdim=0, quantilecutoff=0.5):
		"""supply an array of frames analysed using analyseframeplusfeatures().
		this iterates through using BIGRAMS to create a normalised histogram (by summing peakbin amplitudes)"""
		if binsperdim > 0: # downsampling
			shapefacs = [float(binsperdim) / num for num in frames[0]['rawshape']]
			outshape = [binsperdim] * 4
		else: # raw
			shapefacs = [1] * 4
			outshape = frames[0]['rawshape'] * 2

		results = [] # our data will be stored in here, then sorted so we can use the top quantile
		for i in xrange(len(frames)-1):
			currpeaks = [frames[i]['peaks'][0], frames[i+1]['peaks'][0]]
			mag = (currpeaks[0]['mag'] + currpeaks[1]['mag']) * 0.5 # average the mags
			pos = (int(currpeaks[0]['atom'] * shapefacs[0]), int(currpeaks[0]['bin'] * shapefacs[1]), \
			       int(currpeaks[1]['atom'] * shapefacs[0]), int(currpeaks[1]['bin'] * shapefacs[1]))  # (atom, fakebin, atom, fakebin)
			results.append({'mag': mag, 'pos': pos})
		#print results
		results.sort(key=itemgetter('mag'))
		print "sorted magnitudes are %g, %g ... %g, %g" % (results[0]['mag'], results[1]['mag'], results[-2]['mag'], results[-1]['mag'])

		histo = zeros(outshape)
		magsum = 0.0
		# now with a sorted array, we can use the strongest values to build our histogram
		print "iterating range (%i, %i)" % (int(len(results) * quantilecutoff), len(results))
		for i in xrange(int(len(results) * quantilecutoff), len(results)):
			pos = results[i]['pos']
			mag = results[i]['mag']
			histo[pos] += mag
			magsum += mag
		# normalise
		histo /= magsum
		return histo

	def peaksTemporalCentroid(self, frames):
		"""supply an array of frames anlaysed using analyseframeplusfeatures().
		This calculates the amplitude temporal centroid, returning a number measured in frames"""
		weightedsum =0.0
		totalamp = 0.0
		for index, frame in enumerate(frames):
			for apeak in frame['peaks']:
				weightedsum += apeak['mag'] * float(index)
				totalamp += apeak['mag']
		return weightedsum / totalamp

	def peaksPad(self, frames, extrastart, extraend):
		"Pads a dataset with null frames to make it a certain length"
		numpeaks = len(frames[0]['peaks'])
		ournull = {'peaks': [(0, 0, (0,0), 0) for _ in xrange(numpeaks)]}
		# TODO for the null frequencies, we're using zero. this may be dodgy - perhaps we should use the first/last values instead?
		newdata = [ournull for _ in xrange(extrastart)] \
			+ frames \
			+ [ournull for _ in xrange(extraend)]
		return newdata

	######################################################################################
	# plots and accessories

	def plotchirpogram(self, ff, title='', rangemax=None, cmap=None):
		import matplotlib.pyplot as plt
		fig = plt.figure()
		ax = fig.add_subplot(111)
		normer = plt.Normalize(0, rangemax)
		ax.imshow(ff.T, aspect='auto', interpolation='nearest', norm=normer, cmap=cmap, \
					extent=(shape(ff)[0]*0.5, -shape(ff)[0]*0.5, shape(ff)[1], 0))
		#plt.axvline(shape(ff)[0] * 0.5 - 0.5, color='w')
		plt.axvline(0, color='w')
		plt.title(title)
		plt.ylabel('Detection bin')
		plt.xlabel('Slope (bins)')

	def analyseframeplusfeatures(self, data, hopsize, mode, numtop=1, storeraw=True, addphase=True):
		ana_complex = self.analyseframe(data)
		ana = abs(ana_complex) * 2   # magnitudes *2 because the expression for difference signal of (real * analytic) has a half in it
		phase = 0 - angle(ana_complex)   # negation here to convert phase-of-detection to phase-of-original
		if mode == 'fft':
			# To easily emulate fft mode, we simply restrict to the CENTRAL bin
			centreindex = (shape(ana)[0]-1) / 2
			ana = array([ana[centreindex, : ]])
			peaks = self.frametopresults(ana, numtop, hopsize, centreindex)
		else:
			peaks = self.frametopresults(ana, numtop, hopsize)
		if addphase:
			self.frametopresults_addphase(peaks, phase)
# DEACTIVATED the features - they're slow to calc and I don't use them in chiffchaff stuff or the eusipco expt
		if storeraw:
		#	return {'peaks': peaks, 'slopecent': self.slopeCentroid(ana), 'specflat': self.specFlatness(ana), 'rawshape': shape(ana), 'raw': ana}
			return {'peaks': peaks, 'rawshape': shape(ana), 'raw': ana, 'rawphase':phase}
		else:
		#	return {'peaks': peaks, 'slopecent': self.slopeCentroid(ana), 'specflat': self.specFlatness(ana), 'rawshape': shape(ana)}
			return {'peaks': peaks, 'rawshape': shape(ana)}

	def histoframetoppers(self, frames):
		"compiles histogram data for a sequence of frames given in the format provided by analyseframeplusfeatures()"
		results = zeros(frames[0]['rawshape'])
		for frame in frames:
			for peak in frame['peaks']:
				results[peak['atom'], peak['bin']] += peak['mag']
		return results

	######################################################################################
	# resynthesis from 'peak' objects
	def resynth(self, frames, hopsize=0.5):
		"""Accepts a list of lists of 'peak' dict items, and uses them to resynthesise a mono audio, which it returns as a numpy array.
		Note that phase info is required."""
		# FIXME: Not sure if the phase recovery is perfect -- need to check that. Also, there's a magnitude factor wrong somewhere.
		hopspls = int(self.framesize * hopsize)
		numspls = hopspls * (len(frames)-1) + self.framesize
		audioresult = zeros(numspls)
		curoffset = 0
		xs = arange(self.framesize, dtype=float)
		framehalf = self.framesize / 2
		for peaks in frames:
			for peak in peaks:
				# fromto is frequency... we actually like to convert back to bin
				bp = [datum * self.freqtobin for datum in peak['fromto']]

				chirpremade = self.make_complex_chirp(bp[0], bp[1], peak['phase'])
				chirpremade = real(chirpremade)
				chirpremade *= self.window
				chirpremade /= (2 * sum(chirpremade ** 2))    # Note manual factor of 2, to cancel out the factor of 2 used in analysis
				#chirpremade /= sqrt(sum(chirpremade ** 2))

				audioresult[curoffset:curoffset+self.framesize] += chirpremade * peak['mag']
			curoffset += hopspls
		return audioresult

	# this time using IFFT
	def resynth2(self, frames, hopsize=0.5):
		"""Accepts a list of lists of 'peak' dict items, and uses them to resynthesise a mono audio, which it returns as a numpy array.
		Note that phase info is required."""
		hopspls = int(self.framesize * hopsize)
		numspls = hopspls * (len(frames)-1) + self.framesize
		audioresult = zeros(numspls)
		curoffset = 0

		xs = arange(self.framesize, dtype=float)
		framehalf = self.framesize / 2
		for peaks in frames:
			for peak in peaks:
				# synthesise freq-domain data having the bin, phase and amplitude of this peak
				freqdomain = zeros(self.framesize, dtype=cfloat)
				bintouse = self.detectlobin + peak['bin']
				realbit = peak['mag'] * cos(peak['phase'])
				imagbit = peak['mag'] * sin(peak['phase'])
				freqdomain[bintouse] = realbit + imagbit * 1.j
				timedomain = fft.ifft(freqdomain)
				# complex-divide by the atom
				timedomain /= self.atoms[peak['atom']]
				timedomain *= self.window
				# take the real part, and sum it on
				audioresult[curoffset:curoffset+self.framesize] += real(timedomain)
			curoffset += hopspls
		return audioresult

	def decompose_oneiter(self, signal, hopsize=0.5):
		"""Given an input signal, decomposes it a bit like one round of matching-pursuit or suchlike, with the added constraint of
		one detection per frame. Returns the peaks found, the resynthesised version, and the residual."""
		hopspls = int(self.framesize * hopsize)
		nframes = int(ceil((len(signal) - (self.framesize - hopspls)) / float(hopspls)))
		framespeaks = []
		for whichframe in range(nframes):
			data = signal[hopspls * whichframe:hopspls * whichframe + self.framesize]
			if len(data) != self.framesize: # last frame may be short
				data = hstack((data, zeros(self.framesize - len(data))))
			ana_complex = self.analyseframe(data)
			ana = abs(ana_complex) * 2   # magnitudes *2 because the expression for difference signal of (real * analytic) has a half in it
			phase = 0 - angle(ana_complex)   # negation here to convert phase-of-detection to phase-of-original
			peaks = self.frametopresults(ana, 1, hopsize)
			self.frametopresults_addphase(peaks, phase)
			framespeaks.append(peaks)

		resynth = self.resynth(framespeaks, hopsize)
		# ensure duration matches up (might not be an exact multiple of frames)
		if len(resynth) > len(signal):
			resynth = resynth[:len(signal)]
		residual = signal - resynth
		return {'peaks':framespeaks, 'resynth':resynth, 'residual':residual}

