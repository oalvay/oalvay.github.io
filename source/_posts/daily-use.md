---
title: Hexo Cheatsheet
date: 2020-11-21 23:07:48
tags:
---

这个blog已经配置得很完美了，只是我老是忘了怎么用，所以就懒得更新。。。其实是有非常多东西可以写的。还是写个日常使用教程吧

<!-- more -->

Blog分为两个部分：

+ 源文件（github仓库下source分支）
+ Hexo生成的网站文件（master分支）

默认分支是source，所以本地commit push的时候会提交至source分支下。而Hexo部署会提交到master分支，因为这在`_config.yml`中设置过了。

在日常使用中，我应该：

行为|执行命令
---|---
创建新Blog|`hexo new post <name>`
部署|`hexo g && hexo d`
备份|本地commit后`git push (origin source)`

其实就这么简单。。以后不会再忘了吧