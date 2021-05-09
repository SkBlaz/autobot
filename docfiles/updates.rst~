Potentially interesting further work
===============
We finally discuss the potential implications of the current version of autoBOT, how and where it could be further used.

Extensions with contextual neural representations
---------
Current version of autoBOT does not include any Transformer-based e.g., sentence representations.
Adding this functionality is one API call away, and could notably boost the performance.

Improving the classification phase
---------
Given that the current implementation of autoBOT includes the SGD-based linear learners, a natural extension at this point
is the use of more involved classifiers. Current implementation (see `examples`) offers this functionality out-of-the-box.

Speeding up evolution
---------
Current implementation of evolution is one of the most basic ones. Should more involved, potentially multi-objective
scenarios be considered, it's possible this step can be drastically improved via e.g., inclusion of Pareto front-based traversals etc.

Meta-transfer learning
---------
Given that current implementation of autoBOT results in solution vectors that potentially represent the representation space suitable
for a given document, we believe that using such information as `priors` could speed up the evolution on novel data sets from the same domain.
