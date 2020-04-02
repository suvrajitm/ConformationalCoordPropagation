# ConformationalCoordPropagation
Propagation of Conformational Coordinates Across Angular Space.

We have used the Matlab version for this paper:
Maji et al. 2020."Propagation of Conformational Coordinates Across Angular Space in Mapping the Continuum of States from Cryo-EM Data by Manifold Embedding". DOI: 10.1021/acs.jcim.9b01115.

We are providing here with the python code which is part of the ManifoldEM sofrware, since it contains a cleaner and updated version of the original Matlab implementation.

Myself (Suvrajit Maji) and Hstau Liao are part of the paper "Propagation of Conformational Coordinates Across Angular Space ...", but also thanks to the all other members of the ManifoldEM python software team incuding Evan Seitz and Sonya Hanson.

Since the code is part of the manifoldEM workflow, it takes intermediate output from manifoldEM as an input. 
Therefore to run the code independently we need the files in a specific format as input to this code. 
We will update the requirements soon.

The graph files are produced by 
FindCCGraph.py
FindCCGraphPruned.py (if required)

The main script is:
FindConformationalCoord.py





