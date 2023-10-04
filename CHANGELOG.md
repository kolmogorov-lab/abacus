# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.0.4] - 2023-10-05

### Added

- Add GitHub Action for automated publishing to PyPI on release.
- Make available to plot bootstrap distribution of differences.
- Make bootstrap available for binary metrics.
- Report now returns dictionary with params, not just print report.
- Change format of report output and add warning about usage of results.
- Add ability to save experiment plot and select its kind.

### Changed

- Update lower bound of scipy to 1.10.0.
- target_flg -> target: unify property for setting target variable name.
- Remove unnecessary files.
- Change default treatment group name in splitter from 'target' to 'treatment'.

### Fixed

- Calculation of zero's p-value in bootstrap confidence interval. Now it takes into account directionality of hypothesis.
- Now injections are passed as MDEs, not multiplicators of a metric.


## [0.0.3] - 2023-09-27

### Added

- Add tests for test initialization and statistical tests execution.
- Add open-source community files (CHANGELOG, CODE_OF_CONDUCT, CONTRIBUTING, SECURITY).
- Introduce CI process for testing and linting.

### Changed

- Statistic value and p-value of statistical tests are now limited to 5 digits in decimal place.
- Refactor code with black.

### Fixed

- Pydantic is now limited to the 1.x.x version.
- Typos in links in docs and README.


## [0.0.2] - 2023-08-26
  
### Added

- **ABacus** now has logo.
- Technical documentation is now available on [readthedocs](https://kolmogorov-abacus.readthedocs.io/en/latest/).
- Add dynamic versioning instead of hard code.
 
### Changed

- Remove **hdbscan** from dependencies and move to **scikit-learn==1.3.0**.
- Relax dependencies versioning - now version of dependencies are not pinned to particular versions but constrained to a range of that.

 
## [0.0.1] - 2023-08-19
 
### Added

- Initial release of **ABacus**.
