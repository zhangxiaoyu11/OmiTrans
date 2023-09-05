# OmiTrans
***Please also have a look at our multi-omics multi-task DL freamwork 👀:***
[OmiEmbed](https://github.com/zhangxiaoyu11/OmiEmbed)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5728496.svg)](https://doi.org/10.5281/zenodo.5728496)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ff39650740bb4973b211a4fcfb6c1695)](https://www.codacy.com/gh/zhangxiaoyu11/OmiTrans/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=zhangxiaoyu11/OmiTrans&amp;utm_campaign=Badge_Grade)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/zhangxiaoyu11/OmiEmbed/blob/main/LICENSE)
![Safe](https://img.shields.io/badge/Stay-Safe-red?logo=data:image/svg%2bxml;base64,PHN2ZyBpZD0iTGF5ZXJfMSIgZW5hYmxlLWJhY2tncm91bmQ9Im5ldyAwIDAgNTEwIDUxMCIgaGVpZ2h0PSI1MTIiIHZpZXdCb3g9IjAgMCA1MTAgNTEwIiB3aWR0aD0iNTEyIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxnPjxnPjxwYXRoIGQ9Im0xNzQuNjEgMzAwYy0yMC41OCAwLTQwLjU2IDYuOTUtNTYuNjkgMTkuNzJsLTExMC4wOSA4NS43OTd2MTA0LjQ4M2g1My41MjlsNzYuNDcxLTY1aDEyNi44MnYtMTQ1eiIgZmlsbD0iI2ZmZGRjZSIvPjwvZz48cGF0aCBkPSJtNTAyLjE3IDI4NC43MmMwIDguOTUtMy42IDE3Ljg5LTEwLjc4IDI0LjQ2bC0xNDguNTYgMTM1LjgyaC03OC4xOHYtODVoNjguMThsMTE0LjM0LTEwMC4yMWMxMi44Mi0xMS4yMyAzMi4wNi0xMC45MiA0NC41LjczIDcgNi41NSAxMC41IDE1LjM4IDEwLjUgMjQuMnoiIGZpbGw9IiNmZmNjYmQiLz48cGF0aCBkPSJtMzMyLjgzIDM0OS42M3YxMC4zN2gtNjguMTh2LTYwaDE4LjU1YzI3LjQxIDAgNDkuNjMgMjIuMjIgNDkuNjMgNDkuNjN6IiBmaWxsPSIjZmZjY2JkIi8+PHBhdGggZD0ibTM5OS44IDc3LjN2OC4wMWMwIDIwLjY1LTguMDQgNDAuMDctMjIuNjQgNTQuNjdsLTExMi41MSAxMTIuNTF2LTIyNi42NmwzLjE4LTMuMTljMTQuNi0xNC42IDM0LjAyLTIyLjY0IDU0LjY3LTIyLjY0IDQyLjYyIDAgNzcuMyAzNC42OCA3Ny4zIDc3LjN6IiBmaWxsPSIjZDAwMDUwIi8+PHBhdGggZD0ibTI2NC42NSAyNS44M3YyMjYuNjZsLTExMi41MS0xMTIuNTFjLTE0LjYtMTQuNi0yMi42NC0zNC4wMi0yMi42NC01NC42N3YtOC4wMWMwLTQyLjYyIDM0LjY4LTc3LjMgNzcuMy03Ny4zIDIwLjY1IDAgNDAuMDYgOC4wNCA1NC42NiAyMi42NHoiIGZpbGw9IiNmZjRhNGEiLz48cGF0aCBkPSJtMjEyLjgzIDM2MC4xMnYzMGg1MS44MnYtMzB6IiBmaWxsPSIjZmZjY2JkIi8+PHBhdGggZD0ibTI2NC42NSAzNjAuMTJ2MzBoMzYuMTRsMzIuMDQtMzB6IiBmaWxsPSIjZmZiZGE5Ii8+PC9nPjwvc3ZnPg==)
[![GitHub Repo stars](https://img.shields.io/github/stars/zhangxiaoyu11/OmiTrans?style=social)](https://github.com/zhangxiaoyu11/OmiTrans/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/zhangxiaoyu11/OmiTrans?style=social)](https://github.com/zhangxiaoyu11/OmiTrans/network/members)

**The FIRST GANs-based omics-to-omics translation framework**

**Xiaoyu Zhang** (x.zhang18@imperial.ac.uk)

Data Science Institute, Imperial College London

## Introduction

OmiTrans is a generative adversarial networks (GANs) based omics-to-omics translation framework.

## Getting Started

### Prerequisites
-   CPU or NVIDIA GPU + CUDA CuDNN
-   [Python](https://www.python.org/downloads) 3.6+
-   Python Package Manager
    -   [Anaconda](https://docs.anaconda.com/anaconda/install) 3 (recommended)
    -   or [pip](https://pip.pypa.io/en/stable/installing/) 21.0+
-   Python Packages
    -   [PyTorch](https://pytorch.org/get-started/locally) 1.2+
    -   TensorBoard 1.10+
    -   Tables 3.6+
    -   prefetch-generator 1.0+
-   [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 2.7+

### Installation
-   Clone the repo
```bash
git clone https://github.com/zhangxiaoyu11/OmiTrans.git
cd OmiTrans
```
-   Install the dependencies
    -   For conda users  
    ```bash
    conda env create -f environment.yml
    conda activate omitrans
    ```
    -   For pip users
    ```bash
    pip install -r requirements.txt
    ```

### Try it out
-   Put the gene expression data (A.tsv) and DNA methylation data (B.tsv) in the default data path (./data)
-   Train and test using the default settings
```bash
python train_test.py
```
-   Check the output files
```bash
cd checkpoints/test/
```
-   Visualise the metrics and losses
```bash
tensorboard --logdir=tb_log --bind_all
```

## OmiEmbed
***Please also have a look at our multi-omics multi-task DL freamwork 👀:***
[OmiEmbed](https://github.com/zhangxiaoyu11/OmiEmbed)

## License
This source code is licensed under the [MIT](https://github.com/zhangxiaoyu11/OmiTrans/blob/master/LICENSE) license.
