chirpletringmod
===============

This is the Python implementation of chirplet sound analysis by heterodyning,
as described in the following publication:

D. Stowell and M. D. Plumbley, Framewise heterodyne chirp analysis of birdsong.
   In: Proceedings of the European Signal Processing Conference (EUSIPCO 2012),
   August 2012.

You can think of this as being like a spectrogram analysis, but augmented with
an extra dimension, that of *slope*.

Specific files of note:

 * chirpletringmod.py -- the main file, the implementation of the method.
    Does not depend on other files in the project, but does require numpy.

 * knn_birdclassif.py -- birdsong classification by simple kNN applied to
    chirplet histograms, as described in the above research paper.

