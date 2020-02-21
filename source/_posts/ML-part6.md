---
title: Coursera Machine Learning Week 9
date: 2019-07-25 12:01:41
tags:
- Notes
---

Contents: Anomaly detection and Recommender System

<!-- more -->

## Anomaly detection

In a nutshell, it's about using training set to obtain maximum likelyhood estimation(MLE) of parameters of **Normal distribution**, then use it to predict the probability of new examples being anomalies.

+ Use training sets to calculate **MLE** of parameters $$\mu_1,...,\mu_n, \sigma^2_1,..., \sigma2_n$$:

$$\mu_j = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}_j $$

$$\sigma_j^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)}_j - \mu_j)^2$$

+ compute p(x) for new example $$x \in \mathbb{R}^n $$:

$$p(x) = \prod_{i=1}^n p(x_j; \mu_j, \sigma_j^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}\sigma_j}\exp\left(-\frac{(x-\mu_j)^2}{2\sigma_j^2}\right)$$

+ preidct y based on p(x) with ϵ

$$y = \left\{\begin{array}{ll} 1 & \mbox{if } p(x) \leq \epsilon\\ 0 & \mbox{if } p(x) > \epsilon\end{array}\right.$$

Note that we assume all features are all **independent**, if any correlation exists amongest them, then either create new features manually (e.g. $$x_3 = \frac{x_1}{x_2}$$) to detect anomalies instead of the original ones ($$x_1, x_2$$), or use the **multivariate Normal distribution** (as the parameters of corvariace matrix ∑ generates potential relations automatically).

### Evaluation of performace

Classification accuracy is not a good way to measure the algorithm's performance, because of skewed classes (so an algorithm that always predicts y = 0 will have high accuracy). So people usually prefer F-Score, Precision & Recall or some other methods.

### Split between train & cv sets and Choosing ϵ

+ We are not able to use anomalies for training (get MLEs) so shall not include anu of them in the training sets. Instead use these anomalies in cv and test sets with a 50:50 split would be a nice choice. For instance, if we have 10000 normal and 20 abnormal examples, use 6000 of the normal ones for training, 2000 of normal ones and 10 of abnormal ones for cross validation and the rest for test.

+ The course did not introduce any method of choosing ϵ automatically, so we have to choose it manually (e.g. base on the performance on cross validation set).

### Anomaly detection vs. Supervised learning

The usage of anomaly detection is kind of like classification isn't it? You may wondering why not use supervised learning such as logistic regression instead, so the below explained how supervised learning would perform in two cases:

+ skewed data --- Hard for supervised algorithm to learn. As there exists potentially many different types of anormalies, so future anomalies may look nothing like any of the anomalous examples we’ve seen so far.
+ balanced data ---  Enough positive examples for algorithm to get a sense of what positive examples are like.
 
So use supervised learning algorithm when the number of positive and negative examples are both large, otherwise anomaly detection would be prefered.

The course gives some application scenario which can help you understand better:

 Anomaly detection|Supervised learning
---|---|---
Fraud detection|Email spam classification (easy to get large number of spams)
Manufacturing (e.g. aircraft engines)| Weather prediction (sunny/ rainy/etc).
Monitoring machines in a data center|Cancer classification

### Non-gaussian features

If the data of a feature not bell-shaped curve, using one-to-one transformation (e.g. natural log for positively skewed data) might be a good idea.

## Recommender System

### Content Based Recommendation

For this approachm, we assume that the features of contents are available to us. That means, for instance, we know how much romance or action content is in a movie, such that we are able to use these features to make predictions. The course gave the example below:

Movie | Alice (1)|Bob (2)	|Carol (3) |David (4) |(romance)| (action)
---|---|---|---|---|---|---
Love at last	|5	|5	|0	|0	|0.9	|0
Romance forever	|5	|?	|?	|0	|1.0	|0.01
Cute puppies of love|	?|	4|	0|	?	|0.99| 0
Nonstop car chases	|0	|0	|5	|4	|0.1|	1.0
Swords vs. karate|	0|	0|	5|	?|	0|	0.9

We now define some notations for later problem formulation:  

+ $$n_u$$ = no. users  
+ $$n_m$$ = no. movies  
+ $$r(i, j) = 1$$ if user j has rated movie i  
+ $$y^{(i, j)}$$ = rating given by user j to movie i (define only if $$r(i, j) = 1$$)  
+ $$x^{(i)}$$ = feature vector for movie i  
+ $$\theta^{(j)}$$: For each user j, learn a parameter $$\theta^{(j)} \in \mathbb{R}^3$$ to predict how this user would rate movie i with $$(\theta^{(j)})^T x^{(i)} $$ stars.  


For example, with the parameter $$\theta^{(1)} = \begin{bmatrix}0\\5\\0\end{bmatrix}$$, we predict Alice would rate *Cute puppies of love* a $$(\theta^{(1)})^T x^{(3)} = \begin{bmatrix}1 \ \ 0.99 \ \ 0\end{bmatrix} \begin{bmatrix}0\\5\\0\end{bmatrix} = 4.95$$ stars.


### Optimization objective

To learn a single parameter $$\theta^{(j)}$$ for user j:

$$\min_{\theta^{(j)}} \frac{1}{2} \sum_{i: r(i,j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i, j)})^2 + \frac{\lambda}{2} \sum^n_{k=1} (\theta^{(j)}_k)^2 $$

So to learn all parameters $$\theta^{(1)}, \theta^{(2)}, ..., \theta^{(n_u)}$$ simultaneously, we simply sum cost functions over all users :

$$\min_{\theta^{(1)}, \theta^{(2)}, ..., \theta^{(n_u)}} \frac{1}{2}\color{Blue}{
\sum^{n_u}_{j=1} \sum_{i: r(i,j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i, j)})^2 } + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum^n_{k=1} (\theta^{(j)}_k)^2 $$

#### Gradient Descend update

$$ \theta^{(j)}_k := 
\left\{\begin{array}{ll}
 \theta^{(j)}_k - \alpha \sum_{i: r(i,j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i,j)}) x^{(k)} \ \ & \mbox{for k = 0}\\
\theta^{(j)}_k - \alpha \left ( \sum_{i: r(i,j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i,j)}) x^{(k)} + \lambda \theta^{(j)}_k \right )\ \ & \mbox{for k ≠ 0} 
\end{array}\right.$$

for $$k = 1, ..., n$$ (number of features). 

### Collaborative Filtering 

Now the roles are swaped. Given the parameters θ's, the algorithm is able to predict x's as the following: 

$$\displaystyle \min_{x^{(1)},\dots,x^{(n_m)}} \frac{1}{2} \color{Blue}{
\sum_{i=1}^{n_m}\sum_{j:r(i,j)=1}\left((\theta^{(j)})^T x^{(i)}-y^{(i,j)}\right)^2 } + \frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^n(x_k^{(i)})^2$$

With Gradient Descend update:

$$ x^{(i)}_k := 
\left\{\begin{array}{ll}
 x^{(i)}_k - \alpha \sum_{j: r(i,j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i,j)}) \ \theta^{(j)} \ \ & \mbox{for k = 0}\\
x^{(i)}_k - \alpha \left ( \sum_{j: r(i,j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i,j)}) \ \theta^{(j)} + \lambda x^{(i)}_k \right )\ \ & \mbox{for k ≠ 0} 
\end{array}\right.$$

Therefore, given $$x^{(1)},\dots,x^{(n_m)}$$, we can estimate $$\theta^{(1)},..., \theta^{(n_u)}$$.  
And given $$\theta^{(1)},..., \theta^{(n_u)}$$, we can estimate $$x^{(1)},\dots,x^{(n_m)}$$.  

This algorithm that we are going to use is so-called **Collaborative Filtering**, where θ's and x's are estimated to estimate another.  
You might have noticed the blue-highlighted parts in the cost functions above are actually the same. In fact, rather than update one follow by another, we are able to come up with a more efficent and elegant way to train the algorithm:

$$\min_{x^{(1)},\dots,x^{(n_m)},  \theta^{(1)},..., \theta^{(n_u)}} J(x^{(1)},\dots,x^{(n_m)}, \theta^{(1)},..., \theta^{(n_u)}) $$


where $$J(x^{(1)},\dots,x^{(n_m)}, \theta^{(1)},..., \theta^{(n_u)}) = \\ \frac{1}{2} 
\color{Blue}{ \sum_{(i,j):r(i,j)=1}\left((\theta^{(j)})^T x^{(i)}-y^{(i,j)}\right)^2 } + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum^n_{k=1} (\theta^{(j)}_k)^2 + \frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^n(x_k^{(i)})^2$$

Note that the constant term $$x_0$$'s and $$\theta_0$$'s are no longer needed so are removed.

In the algorithm we described, it is necessary to initialize x's and θ's to small random values to serves as *symmetry breaking* (similar to the random initialization of a neural network’s parameters) and ensures the algorithm learns features x's that are different from each other.  

#### Summarised procedure

1. Initialize $$x^{(1)},\dots,x^{(n_m)},  \theta^{(1)},..., \theta^{(n_u)}$$ to small random values.
2. Minimize $$J(x^{(1)},\dots,x^{(n_m)}, \theta^{(1)},..., \theta^{(n_u)})$$ using gradient descent or advanced optimization algorithm. E.g. 
 $$ \forall \ j = 1,..., n_u, i = 1,...,n_m : \\ x^{(i)}_k := x^{(i)}_k - \alpha \left ( \sum_{j: r(i,j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i,j)}) \ \theta^{(j)} + \lambda x^{(i)}_k \right ) \\ \theta^{(j)}_k := \theta^{(j)}_k - \alpha \left ( \sum_{i: r(i,j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i,j)}) x^{(k)} + \lambda \theta^{(j)}_k \right )$$
3. For a user with parameters θ and a movie with (learned) features x, predict a star rating of $$\theta^T x$$.

### Vectorization: Low rank matrix factorization

It is intuitive to see that the whole process can be vectorised, we take the moving rating status as an example:

$$Y = \begin{bmatrix}
5 & 5 & 0 & 0 \\
5 & ? & ? & 0 \\
? & 4 & 0 & ? \\
0 & 0 & 5 & 4 \\
0 & 0 & 5 & 0
\end{bmatrix}$$ 

$$\displaystyle h(x) = \begin{bmatrix} (x^{(1)})^T(\theta^{(1)}) & \ldots & (x^{(1)})^T(\theta^{(n_u)})\\ \vdots & \ddots & \vdots \\ (x^{(n_m)})^T(\theta^{(1)}) & \ldots & (x^{(n_m)})^T(\theta^{(n_u)})\end{bmatrix}$$

By vectorizing $$X = \begin{bmatrix} - & (x^{(1)})^T & - \\ & \vdots & \\ - & (x^{(n_m)} & - \end{bmatrix},\ \Theta = \begin{bmatrix} - & (\theta^{(1)})^T & - \\ & \vdots & \\ - & (\theta^{(n_u)} & - \end{bmatrix}$$, we are able to write the hypothesis in a much more beatiful way:

$$h(X) = X\Theta^T$$

This matrix is also called a low rank matrix.

#### Finding related movies

For each movie i , we learn a feature vector $$x^{(i)} \in \mathbb{R}^n$$. E.g. $$x^{(i)}_1$$ for romance, $$x^{(i)}_2$$ for action, etc.

Therefore, in order to find whether movie j is similar to movie i, we can compare their distance:

$$\| x^{(i)} - x^{(j)} \|$$

such that small distance would reasonably suggest similarity. So to find 5 most similar movies to movie i, just find the 5 movies with the smallest distance.

#### Mean normalization

In the case where users have not rate any movie yet, the algorithm would predict these users to rate 0 for all movies as that minimise the cost function. Mean normalization is helpful to avoid this to happen by replacing $$X\Theta^T$$ with $$X\Theta^T + \mu$$ where μ is the means of movie ratings rated by some other users.

However, unlike some other applications of feature scaling, we did not scale the movie ratings by dividing by standard deviation. This is because all the movie ratings are already comparable (e.g. 0 to 5 stars), so they are already on similar scales.