---
license: cc-by-nc-4.0
task_categories:
- text-to-video
language:
- en
tags:
- human-feedback
- preference
- ai safety
- large vision model
- lvm
- large language model
- llm
- alignment
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
    - split: train
      path: config-train.json.gz
    - split: test
      path: config-test.json.gz
---

# Dataset Card for SafeSora

<span style="color: red;">Warning: this dataset contains data that may be offensive or harmful. The data are intended for research purposes, especially research that can make models less harmful. The views expressed in the data do not reflect the views of PKU-Alignment Team or any of its members. </span>

[[`🏠 Project Homepage`](https://sites.google.com/view/safe-sora)]
[[`🤗 SafeSora Datasets`](https://huggingface.co/datasets/PKU-Alignment/SafeSora)]
[[`🤗 SafeSora Label`](https://huggingface.co/datasets/PKU-Alignment/SafeSora-Label)]
[[`🤗 SafeSora Evaluation`](https://huggingface.co/datasets/PKU-Alignment/SafeSora-Eval)]
[[`⭐️ Github Repo`](https://github.com/PKU-Alignment/safe-sora)]

SafeSora is a human preference dataset designed to support safety alignment research in the text-to-video generation field, aiming to enhance the helpfulness and harmlessness of Large Vision Models (LVMs). It currently contains three types of data:

- A classification dataset (SafeSora-Label) of 57k+ Text-Video pairs, including multi-label classification of 12 harm labels for their text prompts and text-video pairs.
- A human preference dataset (SafeSora) of 51k+ instances in the text-to-video generation task, containing comparative relationships in terms of helpfulness and harmlessness, as well as four sub-dimensions of helpfulness.
- An evaluation dataset (SafeSora-Eval) containing 600 human-written prompts, with 300 being safety-neutral and another 300 constructed according to 12 harm categories as red-team prompts.

In the future, we will also open-source some baseline alignment algorithms that utilize these datasets.

# Multi-label Classification Dataset

The multi-label classification dataset contains 57k+ text-video pairs, each labeled with 12 harm tags.
We perform multi-label classification on individual prompts as well as the combination of prompts and the videos generated from those prompts. These 12 harm tags are defined as:

- S1: `Adult, Explicit Sexual Content`
- S2: `Animal Abuse`
- S3: `Child Abuse`
- S4: `Crime`
- S5: `Debated Sensitive Social Issue`
- S6: `Drug, Weapons, Substance Abuse`
- S7: `Insulting, Hateful, Aggressive Behavior`
- S8: `Violence, Injury, Gory Content`
- S9: `Racial Discrimination`
- S10: `Other Discrimination (Excluding Racial)`
- S11: `Terrorism, Organized Crime`
- S12: `Other Harmful Content`

<!-- For more details, please see here. -->

The distribution of these 14 categories is shown below:


![image/png](https://cdn-uploads.huggingface.co/production/uploads/6426a75336ab74ff2bca0542/ffb_0BWUh54t_bc-z-Xo0.png)

In our dataset, nearly half of the prompts are safety-critical, while the remaining half are safety-neutral. Our prompts partly come from real online users, while the remaining portion is supplemented by researchers for balancing purposes.

# Visualization example of data points

Multi-label Classification Dataset only contains classification information for T-V pairs

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6426a75336ab74ff2bca0542/9WXTN3wsPJvjsMsKPAQgT.png)
