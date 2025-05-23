## Module: Validation

This module contains features that ensure high-quality deepfake generations and detect failure cases.

### Getting Started

#### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.
- Python 3.8 or higher.

#### Installation

1. reate a Conda environment and activate it:
```bash
conda create -n validation_env python=3.8 -y
conda activate validation_env
```

2. Install the required dependencies:  
```bash
pip install -r requirements.txt
```

3. Run the main validation script:  
```bash
python validation/main.py --df_generation_path <path_to_csv> --deepfake_dataset_path <path_to_dataset> --source_videos_path <path_to_source_videos> --df_mapper <path_to_mapper_csv> --report_csv <output_report_path>
```
