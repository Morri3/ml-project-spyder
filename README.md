# Joke Generator and Detector
This is the coursework of COMP4132 Advanced Topics in Machine Learning (2024-2025).

---

## Why do we have two GitHub repositories?
This repository aims to train and test the joke generator and detector using the GPT2 and BERT models (see the section [The inspiration for the project](#inspiration)). 

Another repository seeks to train and test the joke generator and detector using a Transformer-based model and pre-trained models using the Trainer provided by the Huggingface.

---

## How to use codes?
### Spyder (Strongly recommended)
    
1) Open the Spyder (you can open it through Anaconda).
2) Open the code file.
3) Install required libraries.
4) Click the green play button to run the code.
5) Wait and see the results.

---

## What is the structure of this repository?
In this repository, each code is run independently. Here is its structure using `tree /f > tree.txt` with manual adjustments of styles.

> .<br/>
> ├─ `GPT2_generator_BERT_detector.py` # the final version of joke generator (GPT2) and detector (BERT)<br/>
> ├─ `README.md`<br/>
> ├─ `tree.txt` # the tree structure of this repository<br/>
> │  <br/>
> ├─dataset # dataset used in this repository<br/>
> │&nbsp;&nbsp;&nbsp;&nbsp;├─ `dev-middle.csv` **<br/>
> │&nbsp;&nbsp;&nbsp;&nbsp;├─ `dev-small.csv` **<br/>
> │&nbsp;&nbsp;&nbsp;&nbsp;├─ `dev.csv` *<br/>
> │&nbsp;&nbsp;&nbsp;&nbsp;└─ `shortjokes.csv` ***<br/>
> │      <br/>
> └─tmp_train_process # used during the project, not used in the final version<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ `BERT_no_trainer.py` **** # the initial version of trying to use BERT as the joke generator, without the detector<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ `Generator_trainer.py` **** # trying to use Trainer to train the GPT2 (taken as the joke generator), without the detector<br/>

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
Tip**: These two datasets are preprocessed by the group member [Jiayu Zhang](https://github.com/zjy2414).<br/>
Tip***: It is provided by this module.<br/>
Tip****: **(It is important if you want to run the codes before the final version of models.)** 
Because these two codes were used before the final version and they need to use the BERT model downloaded to the local environment, I uploaded the compressed package to Baidu Netdisk, please download it through [this link](https://pan.baidu.com/s/1Wh1RvZ1POHLQ8gr9JKlUcw?pwd=1111), or I recommend downloading the `bert-base-cased` model from the [Hugging Face](https://huggingface.co/google-bert/bert-base-cased).

---

## What libraries does your environment need?
Here I list the required libraries for each code file. You can install these libraries through conda or pip.

* transformers==4.24.0
* cudatoolkit==11.3.1 *
* pytorch==1.12.1 *
* numpy==1.21.5
* pandas==1.3.5
* collections **
* tqd==4.64.1 ***
* datasets==2.6.1 ****
* matplotlib==3.5.3 *****

Tip*: In order to use available GPU resources, according to [tutorial](https://blog.csdn.net/weixin_46446479/article/details/139004738), it is significant to install `cudatoolkit` library (we can consider it as the **conda version** of `cuda`) **at first**. **After that**, we should go to the official website of [Pytorch](https://pytorch.org/get-started/locally/) to install `pytorch` and related libraries.

To be specific, I installed `pytorch` by using the following command with specific versions:

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

Tip**: It is Python's built-in standard library.<br/>
Tip***: It aims to show the process bar during training and evaluating models.<br/>
Tip****: Using this library to create the datasets.<br/>
Tip*****: Aim to show images of losses for the generator and detector in the form of pyplot-style.

* Besides, `Spyder`'s version is 5.3.3.

---

## The inspiration for the project   <a id="inspiration"></a>
### Code style
In this repository, I referred to the model-building framework of the Pytorch tutorial in the lab.

At the beginning of the project, I tried to use this code style because, during the whole module, this style gave me a clear understanding of the training and validation procedure, making the code more readable and structured.

However, it may be a good choice to use the `Trainer` provided by the [Hugging Face](https://huggingface.co/) to train the models, because it encapsulates the process of training the model, validating and evaluating the model and we should only input the parameters that it needs.

As for me, this coding style is pretty good for letting users understand how the model is trained and how to adjust hyperparameters to achieve better performance. As a result, I kept this code style.

### References
1. The coding style
> From this module's materials (labs and lectures)

2. https://github.com/google-research/bert
> Official GitHub repository of `BERT`

3. https://www.zhihu.com/tardis/bd/art/406240067?source_id=1001
> Interpretation of the `BERT` model (**TWO** pre-training tasks: **Masked Language Model** and **Next Sentence Prediction**)

4. https://zhuanlan.zhihu.com/p/524487313
> **How to download the pre-trained BERT model from the Hugging Face. Realizing the info (like parameters) of the `BertTokenizer` class.**

5. https://huggingface.co/docs/transformers/model_doc/bert#resources, https://huggingface.co/docs/transformers/model_doc/gpt2
> **Hugging face documentation, about `BERT` model and `GPT2` model.** There is no official Pytorch implementation version from Google. But the Hugging Face reimplemented it.

6. https://towardsdatascience.com/how-to-use-bert-from-the-hugging-face-transformer-library-d373a22b0209
> **How to use `BERT` from the Hugging Face Transformer library.**

7. https://arxiv.org/pdf/1609.08144
> `BERT` uses the WordPiece to tokenize.

8. https://blog.csdn.net/tainkai/article/details/130233338
> The function of each pre-trained model of the `BERT` models.

9. https://huggingface.co/docs/transformers/v4.47.1/en/generation_strategies
> **Text generation strategies from the Transformers library, including the parameters related to different strategies.**

10. https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb#scrollTo=LGJWXtNv3l_C
> **Fine-tuning the model using the `Trainer`.**

11. Kenton, J. D. M. W. C., & Toutanova, L. K. (2019, June). Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of naacL-HLT (Vol. 1, p. 2).
> **The paper of the `BERT` model**.

12. https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb#scrollTo=B9_DjWmfWx1q
> How to train the model for the Generator and Detector.

13. https://blog.csdn.net/yueguang8/article/details/136230203
> Randomly split the datasets.

14. https://pytorch.org/docs/main/generated/torch.optim.AdamW.html
> The optimizer used in this project.

15. https://blog.csdn.net/maweifeng1111/article/details/137630245?spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-7-137630245-blog-112730850.235%5Ev43%5Epc_blog_bottom_relevance_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-7-137630245-blog-112730850.235%5Ev43%5Epc_blog_bottom_relevance_base3&utm_relevant_index=14
> The difference between tokenizer() and tokenizer.encode()

16. https://blog.csdn.net/qq_16555103/article/details/136805147, https://huggingface.co/docs/transformers/v4.47.1/en/main_classes/text_generation#transformers.GenerationConfig
> About model.generate().

17. https://blog.csdn.net/weixin_48705841/article/details/144052409
> Use the process bar in Pytorch.

18. https://blog.csdn.net/weixin_44012667/article/details/143839028
> See the batch size and overall number of samples of DataLoader

19. https://huggingface.co/learn/nlp-course/chapter7/3?fw=pt
> **Perplexity.**

20. https://blog.51cto.com/u_16175520/9265189
> Convert `bool` to `int`.

21. https://matplotlib.org/stable/users/explain/quick_start.html#a-simple-example
> Draw loss graphs.

22. https://blog.csdn.net/qq_44858786/article/details/134698463
> Solve the problem: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu().

23. https://blog.csdn.net/qq_53298558/article/details/128951204
> Solve the problem: RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.

24. https://blog.csdn.net/MasterFT/article/details/1671672?spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-1671672-blog-127648136.235%5Ev43%5Epc_blog_bottom_relevance_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-1671672-blog-127648136.235%5Ev43%5Epc_blog_bottom_relevance_base3&utm_relevant_index=10
> Using `tree` to generate the tree structure of the GitHub repository.

25. https://blog.csdn.net/wuShiJingZuo/article/details/141160800
> Python comment specification.

26. https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf
> **The paper of the `GPT-2` model**.
