# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-07

### Added
- Initial release of the PWING VLF Data Processing and Filtering pipeline.
- `ANG_processing_v02.py`: Main parallel processing script that reads binary `.ANG` files, processes them, computes Stokes parameters, and outputs NASA CDF files.
- `PLHRfilter.py`: Filter to dynamically detect and remove Power Line Harmonic Radiation.
- `attenuateVLFstations.py`: Notch filter for known strong VLF transmitters (e.g., Russian Alpha system) to prevent sidelobe interference.
- `removepeakvalues.py`: Adaptive sferic filter for removing impulsive atmospheric noise.
- `requirements.txt`: Specified project dependencies (`cdflib`, `numpy`, `scipy`, `xarray`).
- `README.md`: Comprehensive documentation on project features, installation, and usage.
