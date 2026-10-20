# Training-data overlap

This BBBC039 result is an **in-distribution/overlap-risk comparison**, not a validated out-of-distribution test. No fine-tuning or test-label parameter selection is performed in this project.

The [Cellpose-SAM paper, Methods: Datasets / Main retraining](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1.full) names Cellpose Nuclei among its training collections and states that about half of that collection originates from the 2018 Data Science Bowl. BBBC039 is not named separately in the paper's training-set list. The named collections are not an image-level provenance manifest.

The [Broad BBBC039 record](https://bbbc.broadinstitute.org/BBBC039) states that some BBBC039 fields overlap with BBBC038 (DSB2018). BBBC039 comes from BBBC022; BBBC038 is an overlapping collection, not its parent.

These facts establish a route for potential overlap. They do not establish which of the 50 official BBBC039 test fields were used to train the evaluated models. The official BBBC039 split controls this project's evaluation, not the original models' pretraining. Exact image-level leakage remains unresolved.

The [current Cellpose model documentation](https://cellpose.readthedocs.io/en/latest/models.html) lists `cpsam_v2`, a June 2026 update, and `cpsam`, the original model. The 2025 paper's training description alone does not certify the full provenance of `cpsam_v2`. Model SHA256 digests are saved in the run records so the evaluated weights can be identified.

Do not describe the resulting scores as independent generalisation or directly compare them as a model ranking against a paper using different test images, annotations, thresholds, or aggregation. Image-bootstrap intervals quantify field-to-field sampling variation here; they do not account for training overlap or experiment-level dependence.
