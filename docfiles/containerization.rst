Singularity container for autoBOT
===============
One of the simplest ways to run autoBOT is via containerization. In this work, we adopt the Singularity containers, which are very suitable for ML-based tools. To use them, the following is in order:

1. ./containers/generate.sh generates a singularity container
2. You can run any autoBOT-based code as `singularity exec [imageName] python autoBOT [someSettings+DatasetsHere]`
