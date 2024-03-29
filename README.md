# ConformationalCoordPropagation
Propagation of Conformational Coordinates Across Angular Space.

We have used the Matlab version for this paper:
Maji et al. 2020."Propagation of Conformational Coordinates Across Angular Space in Mapping the Continuum of States from Cryo-EM Data by Manifold Embedding". DOI: 10.1021/acs.jcim.9b01115.
https://doi.org/10.1021/acs.jcim.9b01115

We are providing here with the python code which is part of the *ManifoldEM* python software, since it contains a cleaner and updated version of the original Matlab implementation.


## Instructions
Since the code is part of the *ManifoldEM* workflow, it takes intermediate output from *ManifoldEM* as input. 
Therefore to run the code independently we need the input files in a specific format. We will update the required file format and description soon.

The graph files are produced by 
FindCCGraph.py , FindCCGraphPruned.py (if required)

The main script is:
FindConformationalCoord.py


## Authors
Suvrajit Maji (Matlab and Python version)

Hstau Liao (Python version)

We are part of the "Propagation of Conformational Coordinates Across Angular Space ..." project and also part of the *ManifoldEM* python software team.

## Acknowledgments
Also thanks to other *ManifoldEM* python team members Evan Seitz and Sonya Hanson.




