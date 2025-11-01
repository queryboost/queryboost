# Queryboost Python SDK

The official Python SDK for the Queryboost API, built on gRPC for high-performance AI data processing.

## Installation

```bash
pip install queryboost
```

## Features

Queryboost introduces a new architecture for AI-native data processing at scale:

- **Distributed Continuous Batching** — Ensures that GPUs are fully utilized by continuously batching and scheduling in-flight requests to a pool of reserved GPUs, maximizing throughput and lowering cost per inference.

- **Bidirectional Streaming** — Delivers results in real time as they're ready via gRPC streaming. No need to wait for the entire dataset to finish processing.

- **Column-Aware Prompting** — Run AI queries over your data with column context using native `{column_name}` syntax.

- **Schema-Aware AI Pipeline** — Automatically infers output schema from your data and then generates consistent, schema-compliant structured outputs.

- **Structured Columnar Outputs** — Returns data in Apache Arrow format with a consistent schema, ready to plug directly into analytics and BI workflows.

- **Confidence Scores** — Every output includes per-column confidence scores derived from token-level log probabilities, enabling quality filtering and uncertainty quantification.

- **Optimized Model** — Powered by `queryboost-4b`, a 4B-parameter model optimized for data processing tasks with structured outputs. Outperforms larger models on reading comprehension and natural language inference. [See benchmarks →](https://queryboost.com/#benchmarks)

The Python SDK provides a simple interface to the Queryboost API:

- **Flexible Data Input** — Supports Hugging Face Datasets, IterableDatasets, lists of dictionaries, or custom iterators for large data streams.

- **Pipeline Orchestration** — Starts the bidirectional streaming connection and tracks progress concurrently.

- **Extensible Output Handlers** — Save results locally with `LocalParquetBatchHandler` (default), upload to S3 with `S3ParquetBatchHandler`, or implement custom handlers for databases or any destination.

## Usage

```python
import os
from pathlib import Path
import pyarrow.parquet as pq
from datasets import load_dataset
from queryboost import Queryboost

qb = Queryboost(
    # This is the default and can be omitted
    api_key=os.getenv("QUERYBOOST_API_KEY")
)

# Load data (supports HF Dataset, IterableDataset, list of dicts, or iterator of dicts)
data = load_dataset("queryboost/OpenCustConvo", split="train")

# Select first 160 rows
data = data.select(range(160))

# Use {column_name} to insert column values into prompt
prompt = "Did the customer's issue get resolved in this {chat_transcript}? Explain briefly."

# Process data with bidirectional streaming
# By default, results are saved to ~/.cache/queryboost/ as parquet files
# Pass a custom BatchHandler to save elsewhere (e.g., S3, database)
qb.run(
    data,
    prompt,
    name="cust_convo_analysis", # Unique name for this run passed to the default LocalParquetBatchHandler
    num_gpus=5, # Number of GPUs to reserve for this run
)

# Read the results
table = pq.read_table(Path("~/.cache/queryboost/cust_convo_analysis").expanduser())
```

When you run this code, Queryboost executes end-to-end distributed streaming to process your data with AI:

- **Streams** batches of rows to the API via bidirectional gRPC
- **Distributes** the data stream across a pool of reserved GPUs for parallel processing
- **Dynamically and continuously batches** rows on the server side
- **Generates** structured outputs with consistent schema and associated confidence scores
- **Streams** batches of results back from the API in real time
- **Saves** results as Parquet files to `~/.cache/queryboost/cust_convo_analysis`

> **Note:** Custom `BatchHandler` implementations can save results to remote storage like S3, databases, etc. See [Custom Batch Handlers](#custom-batch-handlers) below.

### Example

Using the customer service transcript example from above:

**Input data**

| chat_id | chat_transcript                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1       | Customer: Hi, I'm having trouble logging into my account. The password reset link isn't working.<br>Agent: I understand that's frustrating. Let me help you with that. I've just sent a new password reset link to your email.<br>Customer: Got it! The new link worked. Thanks for your help!<br>Agent: You're welcome! Is there anything else I can assist you with today?                                                                                                                                                               |
| 2       | Customer: My order #12345 hasn't arrived yet. It's been 2 weeks.<br>Agent: I apologize for the delay. Let me check the status for you... I see there was a shipping delay due to weather conditions. We don't have an updated delivery estimate yet.<br>Customer: This is really frustrating. I need this order urgently.<br>Agent: I understand your frustration. I'll escalate this to our shipping department and have them contact you within 24 hours with an update.<br>Customer: Fine, but I'm very disappointed with this service. |

**Output**

| chat_id | is_resolved | explanation                                                                                                             | is_resolved_confidence | explanation_confidence |
| ------- | ----------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------- | ---------------------- |
| 1       | True        | The customer was able to successfully reset their password with the new link provided by the agent.                     | 0.9823                 | 0.9547                 |
| 2       | False       | The shipping issue remains unresolved, with the agent only promising to escalate and provide an update within 24 hours. | 0.9612                 | 0.9312                 |

> **Note:** Confidence scores are derived from token-level log probabilities.

## Custom Batch Handlers

Batch handlers control how Queryboost handles results that stream in from the API. They can write to local files, upload to cloud storage, insert into databases, or perform custom post-processing.

The `BatchHandler` base class implements buffering logic that accumulates record batches in memory until they exceed `target_write_bytes` (default: 256 MB), then flushes them to the destination. This reduces write overhead by combining multiple small batches into fewer, larger writes.

Queryboost provides built-in handlers:

- **`LocalParquetBatchHandler`** (default) — Saves results as parquet files to a local directory
- **`S3ParquetBatchHandler`** — Uploads results as parquet files to S3

### Using S3ParquetBatchHandler

```python
import pyarrow.parquet as pq
from datasets import load_dataset
from queryboost import Queryboost
from queryboost.handlers import S3ParquetBatchHandler

qb = Queryboost()

data = load_dataset("queryboost/OpenCustConvo", split="train")

data = data.select(range(160))

prompt = "Did the customer's issue get resolved in this {chat_transcript}?"

# Create S3 handler with name and bucket
# Name can include path separators: "prod/2025/customer-analysis"
batch_handler = S3ParquetBatchHandler(
    name="customer-analysis", # Unique name for this run
    bucket="my-data-bucket"
)

qb.run(
    data,
    prompt,
    batch_handler=batch_handler,
    num_gpus=5,
)

# Read the results
table = pq.read_table("s3://my-data-bucket/customer-analysis/")
```

> **Note:** When passing a custom `batch_handler`, do not specify the `name` parameter in `run()`. The handler is already configured with its own name. The `name` parameter is only used when relying on the default `LocalParquetBatchHandler`.

### Creating Custom Handlers

You can also create custom handlers for other destinations like databases (PostgreSQL, Snowflake, BigQuery) or other object stores (GCS, Azure Blob).

Custom handlers inherit from `BatchHandler` and implement a `_flush()` method that writes the accumulated batches to their destination. The `target_write_bytes` parameter can be configured to optimize for different backends. See `src/queryboost/handlers/local.py` or `src/queryboost/handlers/s3.py` for reference implementations.

## License

Copyright © 2025 Queryboost Inc.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
