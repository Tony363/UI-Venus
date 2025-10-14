---
license: mit
task_categories:
- image-text-to-text
---

# UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction
<div style="display: flex; gap: 10px;">
  <a href="https://github.com/uivision/UI-Vision">
    <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="github" class="img-fluid" />
  </a>
  <a href="https://arxiv.org/abs/2503.15661">
    <img src="https://img.shields.io/badge/arXiv-paper-b31b1b.svg?style=for-the-badge" alt="paper" class="img-fluid" />
  </a>
  <a href="https://uivision.github.io/">
    <img src="https://img.shields.io/badge/website-%23b31b1b.svg?style=for-the-badge&logo=globe&logoColor=white" alt="website" class="img-fluid" />
  </a>
</div>

## Introduction

Autonomous agents that navigate Graphical User Interfaces (GUIs) to automate tasks like document editing and file management can greatly enhance computer workflows. While existing research focuses on online settings, desktop environments, critical for many professional and everyday tasks, remain underexplored due to data collection challenges and licensing issues. We introduce UI-Vision, the first comprehensive, license-permissive benchmark for offline, fine-grained evaluation of computer use agents in real-world desktop environments. Unlike online benchmarks, UI-Vision provides: (i) dense, high-quality annotations of human demonstrations, including bounding boxes, UI labels, and action trajectories (clicks, drags, and keyboard inputs) across 83 software applications, and (ii) three fine-to-coarse grained tasks-Element Grounding, Layout Grounding, and Action Prediction-with well-defined metrics to rigorously evaluate agents' performance in desktop environments. Our evaluation reveals critical limitations in state-of-the-art models like UI-TARS-72B, including issues with understanding professional software, spatial reasoning, and complex actions like drag-and-drop. These findings highlight the challenges in developing fully autonomous computer use agents. By releasing UI-Vision as open-source, we aim to advance the development of more capable agents for real-world desktop tasks.

<div align="center">
  <img src="assets/data_pipeline.png" alt="Dataset Overview" width="1000" height="450" class="img-fluid" />
</div>

## Data Structure

To get started with UI-Vision:

1. Clone the repository to get the images and annotations:
```bash
git clone https://huggingface.co/datasets/ServiceNow/ui-vision
```

2. The repository is organized as follows:

```
uivision/
├── annotations/                     # Dataset annotations
│   ├── element_grounding/
│   │   ├── element_grounding_basic.json
│   │   ├── element_grounding_functional.json
│   │   └── element_grounding_spatial.json
│   └── layout_grounding/
│       └── layout_grounding.json
├── images/                         # Dataset images
│   ├── element_grounding/
│   └── layout_grounding/
├── assets/                         # HuggingFace README assets
└── README.md
```

## Usage

To run the models:

1. Visit our [GitHub repository](https://github.com/uivision/UI-Vision) for the latest code
2. Make sure to specify the correct paths to:
   - Annotation files
   - Task image folders

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{nayak2025uivisiondesktopcentricguibenchmark,
  title={UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction}, 
  author={Shravan Nayak and Xiangru Jian and Kevin Qinghong Lin and Juan A. Rodriguez and 
          Montek Kalsi and Rabiul Awal and Nicolas Chapados and M. Tamer Özsu and 
          Aishwarya Agrawal and David Vazquez and Christopher Pal and Perouz Taslakian and 
          Spandana Gella and Sai Rajeswar},
  year={2025},
  eprint={2503.15661},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.15661}, 
}
```

## License

This project is licensed under the MIT License. 

## Intended Usage

This dataset is intended to be used by the community to evaluate and analyze their models. We are continuously striving to improve the dataset. If you have any suggestions or problems regarding the dataset, please contact the authors. We also welcome OPT-OUT requests if users want their data removed. To do so, they can either submit a PR or contact the authors directly.