# Changelog

## [0.1.2](https://github.com/queryboost/queryboost/compare/v0.1.1...v0.1.2)

- Improved resilience to transient gRPC connection losses (e.g., broken pipe, connection reset by peer) after long idle periods
- Fixed intermittent issues with progress bar rendering
- Enhanced prompt and data validation

## [0.1.1](https://github.com/queryboost/queryboost/compare/v0.1.0...v0.1.1)

- Improved cross-platform TLS support by using `certifi` for certificate verification
