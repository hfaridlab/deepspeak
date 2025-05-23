# DeepSpeak

Deepfakes represent a growing concern across domains such as imposter hiring, fraud, and disinformation. Despite significant efforts to develop robust detection classifiers to distinguish the real from the fake, commonly used training datasets remain inadequate: relying on low-quality and outdated deepfake generators, consisting of content scraped from online repositories without participant consent, lacking in multimodal coverage, and rarely employing identity-matching protocols to ensure realistic fakes. To overcome these limitations, we present the DeepSpeak dataset, a diverse and multimodal dataset comprising over 100 hours of authentic and deepfake audiovisual content. We contribute: i) more than 50 hours of real, self-recorded data collected from 500 diverse and consenting participants using a custom-built data collection tool, ii) more than 50 hours of state-of-the-art audio and visual deepfakes generated using 14 video synthesis engines and three voice cloning platforms, and iii) an embedding-based identity-matching approach to ensure the creation of convincing, high-quality identity swaps that realistically simulate adversarial deepfake attacks. We also perform large-scale evaluations of state-of-the-art deepfake detectors and show that these detectors fail to generalize to the DeepSpeak dataset. These evaluations highlight the importance of a large and diverse dataset containing deepfakes from the latest generative-AI tools.

## Features

- [data_collection](data_collection/) — web interface for collecting high-quality video data from participants' webcam
- [experiments](experiments/) — code for reproducing experiments for audio and video deepfake detection, as reported in the paper
- [identity_matching](identity_matching/) — tools for matching pariticipants based on their visual and vocal similatities
- [validation](validation/) — validation mechanisms ensuring high-quality deepfake generations and detecting failure cases

## Getting Started

Each module (folder) is a self-contained environment with its own set of environment preparation steps. Clone the repo `git clone https://github.com/hfaridlab/deepspeak.git` and get started with your exploration! See respective folders for  more detailed instructions.
