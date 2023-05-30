from numpy import *
import csv
import os.path
from scipy.stats import stats

########################################################################################
# These settings define which file(s) will be loaded:
#basepath = '/home/dan/birdsong/xenocanto/london'
basepath = '/Users/danstowell/birdsong/xenocanto-londonwav-20120109'
histobins = 2
#histobins = 3
#histobins = 4
#histobins = 5

# choice of difference/similarity measure (JSD is the most plausible; tf-idf seemed a nice idea but poor results):
#COMPARISON = "cosine"
COMPARISON = "jsd"
#COMPARISON = "tfidf"

SHOWFIGS = False

if SHOWFIGS:
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm

########################################################################################
def preprocessCsvNumbers(row):
	"Converts to numeric form and normalises"
	row = array([float(x) for x in row])
	if COMPARISON == "cosine":
		row = row / sqrt(sum(row ** 2))                # L2
	else:
		row = row / sum(abs(row))                      # L1
	return row

########################################################################################
# distances/similarities
def cosineSimilarity(a, b):
	"this requires L2-NORMALISED data"
	return dot(a, b)

def kullbackLeibler(a, b):
	"KL divergence. This requires L1-NORMALISED data"
	res = 0.0
	for index in range(len(a)):
		if a[index] > 0:
			res += a[index] * log(a[index] / b[index])
	return res

def jensenShannon(a, b):
	"JS divergence. This requires L1-NORMALISED data"
	m = 0.5 * (a + b)   # average distribution
	return 0.5 * (kullbackLeibler(a, m) + kullbackLeibler(b, m))

def idf(data):
	"precalc idf (for tf-idf). for every term (column), we want to know how common it is, i.e. count nonzeros. then do log(N/df)"
	detect = array([x['histo'].copy() for x in data])
	detect[detect != 0] = 1
	df = sum(detect, 0)
	df[df == 0] = 1e-6  # avoid div-by-zero for terms not present in dataset (they're unused anyway)
	idf = log(float(len(detect)) / df)
	print("IDF precalculation: max %g, min %g, mean %g, len %i" % (max(idf), min(idf), mean(idf), len(idf)))
	print(idf)
	return idf

def tfidf(a, b, idfs):
	"calculates the summed tf-idf for all 'query terms' in a and 'document' b, standardised by length. this comes out as a symmetric calculation in a and b, funnily."
	return sum(a * b * idfs)

########################################################################################

def loocvFile(csvpath, kvals):
	print("Analysing %s" % csvpath)
	reader = csv.reader(open(csvpath, "rb"))
	next(reader) # drop the header row
	# loaded as list of tuples ([numbers], classif)
	data = [{'histo': preprocessCsvNumbers(row[:-1]), 'class': row[-1]} for row in reader]

	# sort the data by class (makes dot-product plot more intelligible)
	data.sort(key=lambda x: x['class'])

	# precalc all pairwise similarities/distances
	if COMPARISON == "tfidf":
		idfs = idf(data)  # global precalculation for tf-idf
	comparisons = {
			"cosine": lambda: [[cosineSimilarity(a['histo'], b['histo']) for b in data] for a in data],   # cosine similarity
			"jsd":    lambda: [[jensenShannon(a['histo'], b['histo']) for b in data] for a in data],    # jensen-shannon divergence
			"tfidf":  lambda: [[tfidf(a['histo'], b['histo'], idfs) for b in data] for a in data]    # tf-idf relevance
		}[COMPARISON]()

	# colormap plot:
	if SHOWFIGS:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.imshow(comparisons, aspect='auto', cmap=cm.gray, interpolation='nearest')
		plt.title({
			"cosine": 'Cosine similarities (%s)',
			"jsd":    'Jensen-Shannon divergences (%s)',
			"tfidf":  'tf-idf similarities (%s)'
			}[COMPARISON] % os.path.basename(csvpath))

	# create a list of the classes used in the data (and prealloc confusion matrix):
	classes = {}
	for datum in data:
		classes[datum['class']] = 1
	classes = list(classes.keys())
	classes.sort()
	print("Classes in dataset:")
	print(classes)
	confusion = [zeros((len(classes), len(classes))) for _ in range(len(kvals))]
	aucs      = zeros((len(data))) # each datum in the LOO will calc an AUC; we will report mean and CI.

	# LOO-CV: foreach datum, "train" on all the others and test on the datum.
	# Since kNN is lazy there's no training, you just find the kNN.
	for index, datum in enumerate(data):
		#print "---------------------------"
		# take column 'index' from comparisons, store it with indices included
		rankedneighbours = array(\
			[(key,v,data[key]['class']) for key,v in enumerate(comparisons[index][:])], \
			dtype=[('index', int), ('comparisonval', float), ('class', 'S32')])
		# remove the self-reference
		rankedneighbours = hstack((rankedneighbours[:index], rankedneighbours[index+1:]))

		# now sort by similarity/distance value
		rankedneighbours.sort(order=['comparisonval'])
		# ONLY REVERSE IF using 'similarity' (cosine, tfidf) rather than 'distance' (jensenShannon)
		if COMPARISON != "jsd":
			rankedneighbours = rankedneighbours[::-1]
		#print "Top %i sorted neighbs for #%i" % (max(kvals), index)
		#print rankedneighbours[:max(kvals)]

		# find the majority vote from the top k.
		# we add a tiny weighting to the votes to split ties
		votes = [zeros(len(classes)) for _ in range(len(kvals))]
		for nindex in range(max(kvals)):
			neighb = rankedneighbours[nindex]
			myvote = classes.index(neighb['class'])
			myweight = 1.0 - (nindex * 0.001)
			for ik, k in enumerate(kvals):
				if nindex < k:
					votes[ik][myvote] += myweight
		for ik, k in enumerate(kvals):
			chosen = argmax(votes[ik])
			#print "k=%i chose to classify %i as %s (%i votes; true is %s)" % (k, index, classes[chosen], round(max(votes[ik])), datum['class'])
			confusion[ik][chosen, classes.index(datum['class'])] += 1

		#print "---------------------------"
		# Now let's calculate the AUC for this particular datum:
		numtp = 0
		numfp = 0
		prevtprate = 0
		prevfprate = 0
		number_of_matchers_in_database = [x['class'] for x in rankedneighbours].count(datum['class'])
		number_of_unmatchers_in_database = len(data) - number_of_matchers_in_database
		auc = 0.0
		for rankpos, rankedneighb in enumerate(rankedneighbours):
			# we take the top k hits, calc TP rate and FP rate. area under (TP, 1-FP) is what we want.
			numhitsconsidered = rankpos + 1
			if rankedneighb['class'] == datum['class']:
				numtp += 1
			else:
				numfp += 1
			tprate = numtp / float(number_of_matchers_in_database)
			fprate = numfp / float(number_of_unmatchers_in_database)

			# parallelogram calculation for AUC:
			# for each pair of results, take the AVERAGE of their TPs, and multiply it by the DIFFERENCE in their FPs.
			# must include the zero and the all (to get both endpoints).
			auc += (prevtprate + tprate) * 0.5 * abs(fprate - prevfprate)

			prevtprate = tprate
			prevfprate = fprate
		aucs[index] = auc * 100
		#print "AUC for item %i is %g %%" % (index, aucs[index])

	############################################################
	# iteration finished. show results.
	#print "---------------------------"
	if SHOWFIGS:
		for ik, k in enumerate(kvals):
			#print "Confusion matrix for k=%i:" % (k)
			#for key, cl in enumerate(classes):
			#	print("%s %s" % (str(confusion[ik][key]), cl))
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.imshow(confusion[ik], aspect='auto', cmap=cm.gray, interpolation='nearest')
			plt.title('Confusion matrix for k=%i (%s)' % (k, os.path.basename(csvpath)))
	print("---------------------------")
	for ik, k in enumerate(kvals):
		numcorrect = sum([confusion[ik][i][i] for i in range(len(classes))])
		print("Num correct for k=%i: %i/%i (%g %%)" % (k, numcorrect, len(data), 100 * float(numcorrect) / float(len(data))))
	print("---------------------------")
	print("AUC (%%): mean %g +- %g" % (mean(aucs), 1.96 * std(aucs) / sqrt(len(data))))
	return aucs

def loocvFileCompare(csvpath, kvals):
	"calls loocvFile() for a FFT and a CH then decides if the AUC difference is significant"
	aucs = [loocvFile(csvpath % ('fft'), kvals),
		loocvFile(csvpath % ('ch' ), kvals)]
	(t, prob) = stats.ttest_rel(aucs[0], aucs[1])
	print("---------------------------")
	print("Paired T-test: t=%g, p=%g" % (t, prob))

########################################################################################
# let us.

#loocvFile("%s/histob_%s_%ibins.csv" % (basepath, histomode, histobins), [1, 3, 5])
loocvFileCompare("%s/histob_%s_%ibins.csv" % (basepath, '%s', histobins), [1, 3, 5])
if SHOWFIGS:
	plt.show()
