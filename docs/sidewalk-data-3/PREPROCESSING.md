# Image preprocessing

All source images are read directly from sidewalk-data-2, including recursively
nested new training collections. Each image is decoded fully, EXIF-transposed
into display orientation, converted using Pillow `.convert("RGB")`, resized
from the complete image to 224 x 224 with BICUBIC interpolation, and encoded as
lossless RGB PNG (compression level 6). Alpha is dropped in the same RGB conversion
used by the previous pipeline; source alpha minima are recorded for auditing.
The full oriented image is the crop rectangle. No center crop, padding, annotation
marker removal, color enhancement, or random augmentation is applied. Aspect ratios
are resized directly to a square, matching the previous naive-merged-nulls setup.
Screenshots already showing a selected sidewalk scene receive no further crop.
This standardizes format/shape, not the original field of view of heterogeneous
sources (GSV frames, existing crops, and screenshots).

RGB uint8 pixels are stored without model-specific normalization. Training loaders
should convert to tensors and apply the selected model's normalization consistently;
do not perform another center crop or change evaluation geometry unintentionally.
Original normalized_x/y annotations are retained in SOURCE-image coordinates.
Source dimensions, EXIF orientation, full-image crop bounds and output dimensions
are recorded, so coordinates can be transformed explicitly if necessary.

Every output is reopened and decoded, its shape/mode/format and hashes are checked,
and all targets are recomputed from preserved tag values. Images use a deterministic
20-hex-character SHA-256 prefix of their source-relative path plus the source stem;
original names and paths are retained verbatim in metadata. This avoids collisions
between collection folders or source formats.
