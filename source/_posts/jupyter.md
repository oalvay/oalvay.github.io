---
title: Jupyter
date: 2020-12-11 09:50:32
tags:
---
我曾经是信仰Jupyter notebook的，但是现在不是了。

<!-- more -->

写[TTDS CW2](http://www.inf.ed.ac.uk/teaching/courses/tts/CW2020/assignment2.html)的时候，我照常用了Jupyter notebook。把每一步的都写在一个cell里，分块执行。到要交作业的时候，再把每一个cell的代码块复制粘贴拼在一起。我觉得这样应该没问题。

结果出事了。。。Task 3里用[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)算的两个模型和原来分块执行的效果差得很大，没有找到原因。另外比较迷惑的是，SVC的`random_state`只在`probability=True`的时候起效，我没有设置这个参数，结果每次模型计算结果都不一样。需要`random_state`的地方除了这里就只有split data的时候用到了，那为什么每次模型计算结果都不一样呢？

总结经验：

+ 每次用Jupyter notebook工作完后，要用`Restart & Run All`执行一下看和原来结果有没有出入
+ 不能偷懒不用git，单靠jupyter的checkpoint没啥用。管理版本很重要，不然出错了没法对比找原因。