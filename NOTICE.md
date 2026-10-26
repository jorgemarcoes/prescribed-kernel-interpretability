# License and attribution notices

This distribution is licensed under the GNU General Public License, version 3
(SPDX identifier: `GPL-3.0-only`). The complete license is in `LICENSE`.
It is distributed without any warranty, including the implied warranties of
merchantability or fitness for a particular purpose.

Original software and documentation contributions in this distribution:
Copyright (C) 2026 Jorge Marco Blanco.

## ROCKET adaptation

`src/rocket.py` is a modified adaptation of the reference ROCKET implementation
by Angus Dempster, François Petitjean and Geoffrey I. Webb:

- Repository: https://github.com/angus924/rocket
- Source: https://github.com/angus924/rocket/blob/master/code/rocket_functions.py
- Upstream license: https://github.com/angus924/rocket/blob/master/LICENSE
- Paper: Dempster, A., Petitjean, F. and Webb, G. I. (2020).
  ROCKET: exceptionally fast and accurate time series classification using
  random convolutional kernels. Data Mining and Knowledge Discovery, 34,
  1454–1495. https://doi.org/10.1007/s10618-020-00701-z

The original source attribution is retained in `src/rocket.py`. The adaptation
retains PPV features only, makes kernel lengths configurable, normalises centred
kernels to unit energy, adds input checks and provides a scikit-learn-style
wrapper. This modified distribution was prepared for publication on 26 October
2026. It is not the original ROCKET implementation.

## External dependencies and datasets

External Python dependencies retain their respective licenses; this repository
does not relicense them or bundle their source distributions. No downloaded UCR
time-series datasets are included. Datasets obtained through aeon retain their
respective terms and attribution requirements. `data/metadata.csv` contains the
supplied descriptive metadata used by the analyses; it is not the raw data.
