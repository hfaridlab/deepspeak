## Module: Identity Matching

This module provides tools for matching identities based on visual and vocal similarities. 

## Getting Started

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.
- Python 3.8 or higher.

### Installation

1. Create a Conda environment and activate it:  
```bash
conda create -n identity_matching_env python=3.8 -y
conda activate identity_matching_env
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main identity matching script for visual similarity: 
```bash
python video/main.py --config <path_to_config_file>
```

4. Run the main identity matching script for vocal similarity: 
```bash
python audio/audio_titanet_generate_cosine_pairs.py
```