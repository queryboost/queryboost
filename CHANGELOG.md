# Changelog

## [0.1.3](https://github.com/queryboost/queryboost/compare/v0.1.2...v0.1.3)

- Added `S3ParquetBatchHandler` for uploading results as Parquet files to S3
- Improved batch handler performance with internal buffering that accumulates record batches and flushes when size threshold is met (default 256 MB), reducing write overhead for network-based destinations like S3 and databases while creating optimally-sized Parquet files for better query performance

## [0.1.2](https://github.com/queryboost/queryboost/compare/v0.1.1...v0.1.2)

- Improved resilience to transient gRPC connection losses (e.g., broken pipe, connection reset by peer) after long idle periods
- Fixed intermittent issues with progress bar rendering
- Enhanced prompt and data validation

## [0.1.1](https://github.com/queryboost/queryboost/compare/v0.1.0...v0.1.1)

- Improved cross-platform TLS support by using `certifi` for certificate verification
