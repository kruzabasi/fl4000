# docs/fl_framework.md

# Federated Learning Framework

This document describes the architecture, components, and usage of the FL framework implemented in Sprint 3.

## Directory Structure

- `src/federated/`: Core FL logic (aggregator, client, simulation, utils)
- `src/models/`: Model definitions
- `src/data_preprocessing.py`: Data preprocessing logic
- `src/feature_engineering.py`: Feature engineering logic
- `src/partition_data.py`: Data partitioning logic
- `src/baseline_trainer.py`: Baseline training logic
- `tests/`: Unit and integration tests

## Components

### Aggregator
Handles aggregation of model updates from clients.

### Client
Represents a data owner participating in FL.

### Simulation
Orchestrates the FL process.

### Utils
Helper functions for parameter management and other utilities.

### Models
Contains model definitions for use in FL.

## Usage

(Describe how to run the simulation, configure experiments, etc.)

---

*Expand this document as the implementation progresses.*
