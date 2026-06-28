# UAB-RICSE-TFM-Project

## Performance Analysis of Cloud-Based Distributed Machine Learning Workflows

This repository contains scripts, configurations, and results from a comprehensive thesis on performance analysis of cloud-based distributed machine learning workflows.

### 📋 Overview

This master’s thesis investigates the performance of hybrid cloud machine-learning workflows in which model training is performed on a private GPU machine while dataset storage is provided by Amazon S3. A ResNet-18 model was trained on the Tiny ImageNet dataset under three deployment scenarios. Each scenario was executed 30 times, and metrics such as training time, throughput, data-loading latency, GPU utilization, CPU and memory usage, S3 data transfer, cache behavior, and validation accuracy were collected and compared.

The results show that directly streaming training samples from S3 introduces a significant input bottleneck and leads to low GPU utilization. Applying local caching, cache warmup, prefetching, and increased DataLoader parallelism substantially improves steady-state training performance, although the initial cache-warmup stage increases the total workflow duration.

### 📁 Repository Structure

```
UAB-RICSE-TFM-Project/
├── Scenario_A/          # First experimental scenario
│   ├── scripts/         # Scenario A scripts and configurations
│   └── data/            # Scenario A Docker and other files
├── Scenario_B/          # Second experimental scenario
│   ├── scripts/         # Scenario B scripts and configurations
│   └── data/            # Scenario B Docker and other files
├── Scenario_C/          # Third experimental scenario
│   ├── scripts/         # Scenario C scripts and configurations
│   └── data/            # Scenario C Docker and other files
├── Results/             # Aggregated results and analysis
│   ├── metrics/         # Performance metrics and measurements
│   ├── plots/           # Tableau workbook for visualizations and charts
├── Dataset/             # Prepare validation data script
└── README.md            # This file
```

### 🔬 Scenarios

#### Scenario A
Scenario A represents the private baseline. The Tiny ImageNet dataset, training process, and output files are stored and processed on the private machine. Training data are read directly from local disk using four DataLoader workers, without relying on public-cloud storage during model execution.

#### Scenario B
Scenario B implements a basic hybrid workflow. The model is trained on the private GPU machine, while the dataset remains in Amazon S3. Images are downloaded on demand during training and validation, without maintaining a complete local copy. This scenario evaluates the performance overhead introduced by remote object-storage access.

#### Scenario C
Scenario C extends the hybrid workflow with data-pipeline optimizations. Before training, the dataset is downloaded from S3 into a local cache. During model execution, 16 DataLoader workers read and preprocess cached images in parallel, while prefetching prepares future batches in memory. This approach reduces repeated S3 access and improves data-loading latency, throughput, and GPU utilization.

### 📊 Results

All aggregated results, performance metrics, and visualizations are located in the `Results/` directory:
- **metrics/**: Raw performance data and measurements
- **plots/**: Graphs and visualizations

### 🛠️ Requirements

- Python 3.x
- Required packages: in requirements files for every scenario.
  

### 📚 Thesis Information

- **Title**: Performance Analysis of Cloud-Based Distributed Machine Learning Workflows
- **Institution**: Universitat Autònoma de Barcelona (UAB)
- **Program**: RICSE (Research and Innovation in Computer based Science and Engineering)
- **Date**: 2025/2026

### 📧 Contact

For questions or inquiries about this project, please contact the thesis author.


**Last Updated**: June 2026
