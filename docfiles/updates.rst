Releases
===============
Please see the tagged versions of the code to inspect different releases of the library. As we remain actively exploring what's possible, the codebase could change (and defaults with it). Note however, that the *vMLJ* release was the one used for benchmarking.

1. 8.5.21 -> Changed the defaults so that multilingual settings work fine.
2. 11.5.21 -> Added document graph-based features (default configuration accessible as representation_type = "neurosymbolic-default")
3. 20.5.21 -> Added a fully language-agnostic version accessible as representation_type = "neurosymbolic". This representation includes topic-level features and document graph-based information. Also added the "neurosymbolic-lite", a less memory-intensive version suitable for larger datasets. In the future, the selection of the mode of operation will be automated based on metadata.
