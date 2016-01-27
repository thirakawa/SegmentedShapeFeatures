# Segmented Shape Features (SSF)

This repository implements a texture feature which is called *Segmented Shape Features* (SSF) proposed by Häfner et al. [1].

This implementation has a small difference as follows.
Häfner et al. have used and simplified *Fast Level Lines Transform* (FLLT) [2] to construct upper and lower level line sets from an image.
While we construct the level line sets by *Tree of Shapes* (ToS) using an algorithm proposed by Géraud et al. [3].


## Execution environment
* OS: Mac OS X (10.9 or 10.11)
* Language: Python 2.7.\* (Anaconda 2.4.\*)
* Modules: Numpy, Matplotlib, OpenCV, NetworkX, Cython


## Usage
    python setup.py build_ext --inplace
    python SegmentedShapeFeatures.py [image file]


## Sample images
Some of the Brodatz dataset images are stored in *images* directory.  
[Brodatz dataset](http://multibandtexture.recherche.usherbrooke.ca/original_brodatz.html)


## References
1. M. Häfner, A. Uhl, and G. Wimmer, "A Novel Shape Feature Descriptor for the Classification of Polyps in HD Colonoscopy," Lecture Notes in Computer Science (LNCS), pp.205-213, 2014.
2. P. Monasse and F. Guichard, "Fast computation of a contrast-invariant image representation," iin Image Processing, IEEE Transactions on, vol.9, no.5, pp.860-872, 2000.
3. T. Géraud, E. Carlinet, S. Crozet, and L. Najman, "A quasi-linear algorithm to compute the tree of shapes of nD images," International Symposium on Mathematical Morphology (ISMM), pp.98-110, 2013.
