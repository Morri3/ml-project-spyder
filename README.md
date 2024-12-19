# Joke Generator and Detector
This is the coursework of COMP4132 Advanced Topics in Machine Learning (2024-2025).

## Why do we have two GitHub repositories?
This repository aims to train and test the joke generator and detector using the GPT2 and BERT models in the form of Pytorch Neural Network architecture. 

Another repository seeks to train and test the joke generator and detector by using a Trainer provided by the Huggingface.

## How to use codes?
### 1. Spyder (Strongly recommended)
    
1) Open the Spyder (you can open it through Anaconda).
2) Open the code file (anyone you want to run and test).
3) Install required libraries.
4) Click the green play button to run the code.
5) Wait and see the results.

### 2. VS Code

1) Open the VS Code.
2) Open the code file (anyone you want to run and test).
3) Install required libraries.
4) Run the following command in the terminal.
   
```bash
  python [file_name].py
```

Here, the [file_name] should be set to the file's real name, including _BERT_, _GPT2_ and _GPT2_generator_detector_. 

For instance, if you want to run the 'GPT2.py', you should input the following command:

```bash
  python GPT2.py
```

5) Wait and see the results.

## What is the structure of this repository?
In this repository, each code is run independently. Here is its structure using `tree /f > tree.txt` with manual adjustments of styles.

> .
> │─ BERT.py # the initial version of trying using BERT as the joke generator, without the detector
> │─ GPT2.py # the version of trying to use GPT2 as the joke generator, without the detector
> │─ GPT2_generator_detector # the final version of joke generator (GPT2) and detector (BERT)
> │─ README.md
> │─ tree.txt # the tree structure of this repository
> │  
> └─dataset # dataset used in this repository
> &nbsp;&nbsp;&nbsp;&nbsp; │─ dev-middle.csv **
> &nbsp;&nbsp;&nbsp;&nbsp; │─ dev-small.csv **
> &nbsp;&nbsp;&nbsp;&nbsp; │─ dev.csv *
> &nbsp;&nbsp;&nbsp;&nbsp; └─ shortjokes.csv ***

Tip*: This dataset is cited from a paper [The rJokes dataset: a large scale humor collection](https://aclanthology.org/2020.lrec-1.753/)
```
@inproceedings{weller2020rjokes,
  title={The rJokes dataset: a large scale humor collection},
  author={Weller, Orion and Seppi, Kevin},
  booktitle={Proceedings of the Twelfth Language Resources and Evaluation Conference},
  pages={6136--6141},
  year={2020}
}
```
Tip**: These two datasets are preprocessed by the group member [Jiayu Zhang](https://github.com/zjy2414).
Tip***: It is provided by this module.

## What libraries does your environment need?
Here I list required libraries for each code file. You can install these libraries through conda or pip.

* transformers
* cudatoolkit *
* pytorch *
* numpy
* pandas
* collections **
* tqdm ***
* datasets ****
* matplotlib *****

Tip*: In order to use available GPU resources, according to [tutorial](https://blog.csdn.net/weixin_46446479/article/details/139004738), it is significant to install `cudatoolkit` library (we can consider it as the **conda version** of `cuda`) **at first**. **After that**, we should go to the offical website of the [Pytorch](https://pytorch.org/get-started/locally/) to install `pytorch` and related libraries.

To be specific, I installed `pytorch` by using the following command with specific versions:

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

Tip**: This library is Python's built-in standard library.
Tip***: It aims to show the process bar during training and evaluating models.
Tip****: Using this library to create the datasets.
Tip*****: It aims to show images of losses for the generator and detector in the form of pyplot-style.

## The inspiration for the project

