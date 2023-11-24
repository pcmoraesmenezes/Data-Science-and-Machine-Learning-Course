# Statistics in Python

## Table of Contents

- [Statistics in Python](#statistics-in-python)
  - [Table of Contents](#table-of-contents)
  - [Probability vs Statistics](#probability-vs-statistics)
  - [Sets](#sets)
    - [Exercises about sets](#exercises-about-sets)
    - [Some operations with sets](#some-operations-with-sets)
    - [Subsets](#subsets)
    - [Complement](#complement)
    - [De Morgan's laws](#de-morgans-laws)
    - [Disjoint sets](#disjoint-sets)
    - [Joint sets](#joint-sets)
    - [Exercises:](#exercises)
  - [Experiment](#experiment)
    - [Random experiment / Random trial](#random-experiment--random-trial)
      - [Outcomes](#outcomes)
      - [Sample space](#sample-space)
      - [Events](#events)
      - [Exercices](#exercices)
      - [Events](#events-1)
      - [Exercices](#exercices-1)
  - [Probability Model](#probability-model)
    - [Probability Law](#probability-law)
    - [Axioms of Probability](#axioms-of-probability)
  - [Probability Model: Conditioning](#probability-model-conditioning)
    - [Conditional Probability in Machine Learning](#conditional-probability-in-machine-learning)
  - [Probability Model: Law of Total Probability](#probability-model-law-of-total-probability)
    - [Independence](#independence)
    - [Conditional Independence](#conditional-independence)
    - [Exercise:](#exercise)
  - [Bayes Rule](#bayes-rule)
  - [Towards Random Variables](#towards-random-variables)
    - [Discrete Random Variables](#discrete-random-variables)
    - [Continuous Random Variables](#continuous-random-variables)
    - [Exercises](#exercises)
    - [Bernoulli Random Variable](#bernoulli-random-variable)
    - [Geometric RV](#geometric-rv)
    - [Binomial RV](#binomial-rv)
    - [Continuous RV](#continuous-rv)
      - [Probability Density Function (PDF)](#probability-density-function-pdf)
      - [Exponential RV](#exponential-rv)
        - [Gaussian RV](#gaussian-rv)
        - [What is a Cumulative Distribution Function (CDF)?](#what-is-a-cumulative-distribution-function-cdf)
    - [Expectation](#expectation)
      - [Law of Large Numbers](#law-of-large-numbers)
    - [Transformations of Random Variables](#transformations-of-random-variables)
    - [Variance](#variance)
    - [Joint Probability Distribution](#joint-probability-distribution)
    - [Multivariate Gaussian Distribution](#multivariate-gaussian-distribution)
    - [Curse of Dimensionality](#curse-of-dimensionality)
    - [What is the expected value of a transformation of a random variable?](#what-is-the-expected-value-of-a-transformation-of-a-random-variable)
    - [Codes](#codes)
        - [Naive Bayes for iris dataset](#python-naive-bayes-implementation)
        - [Random Variables](#random-variables-1)
        - [Random variables in dataset](#random-variables-1)
        - [Expectations](#expectations)
        - [Sets](#sets-1)

---

## Probability vs Statistics

1. What's probability?

    - As a theory probability deals with to predict or calculate the likelihood of future events.

    - As a number probability is a number between 0 and 1 that measures the likelihood of an event occurring. The higher the probability, the more likely the event is to occur.


2. What's statistics?

    - Statistics is the science of collecting, organizing, analyzing, interpreting, and presenting data. It deals with all aspects of data including the planning of data collection in terms of the design of surveys and experiments.

    - Statistics is also about providing a measure of confidence in any conclusions.

    - Statistics is a set of tools that can be used to answer questions about data. In other words, statistics is the toolbox that helps us to understand data.

    - Envolves the analysis of past events.


Probability needs rule's to be applied and predict the future. The goal of statistics is to make those rules from the data.

The task of statistical is to look back at the data and try to find patterns that can be used to predict the future. Once we have a model that can predict the future, we can use probability to calculate the likelihood of future events.

The task of probability is to look forward and use the laws made by statistics to predict the future.

## Sets


Probability use sets to represent events. A set is a collection of objects. The objects in a set are called elements. Sets are usually denoted by capital letters. The elements of a set are usually denoted by small letters.

Is a unordered collection of distinct elements.

By unordered means that if we get an arrengement of the elements of a set, it's still a set. But if the order of the elements matters, then we have a sequence. The objects must have any position in the set.

By  distinct means that there are no duplicate elements in a set. If we have duplicate elements, we can remove them and the set will remain the same.

Objects in a set are called elements. The elements of a set are usually denoted by small letters. It can be any type of object like numbers, letters, words, etc.

### Exercises about sets

1. Create a set with the following elements: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89.

    ```python
    set1 = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}
    ```

2. Can a set have different types of objects?
    
        Yes, a set can have different types of objects. But is more common to have sets with the same type of objects. For example, a set of numbers, a set of letters, a set of words, etc.

3. What's multiset? 

    A multiset is a set that can have duplicate elements. The elements of a multiset are usually denoted by small letters with a subscript to indicate the number of times the element appears in the multiset.

    ```python
    multiset = {a1, a2, a3, ..., an}
    ```


### Some operations with sets

1. Union

    The union of two sets A and B is the set of elements which are in A, in B, or in both A and B. The union of A and B is denoted by A ∪ B.

    ```python
    A = {1, 2, 3, 4, 5}
    B = {4, 5, 6, 7, 8}
    A ∪ B = {1, 2, 3, 4, 5, 6, 7, 8}
    ```

2. Intersection

    The intersection of two sets A and B is the set of elements which are in both A and B. The intersection of A and B is denoted by A ∩ B.

    ```python
    A = {1, 2, 3, 4, 5}
    B = {4, 5, 6, 7, 8}
    A ∩ B = {4, 5}
    ```

3. Difference

    The difference of two sets A and B is the set of elements which are in A but not in B. The difference of A and B is denoted by A - B.

    ```python
    A = {1, 2, 3, 4, 5}
    B = {4, 5, 6, 7, 8}
    A - B = {1, 2, 3}
    ```

4. Belongs

    The belongs of an element x to a set A is true if x is in A and false otherwise. The belongs of x to A is denoted by x ∈ A.

    ```python
    A = {1, 2, 3, 4, 5}
    3 ∈ A = True
    6 ∈ A = False
    ```

    ```python
    C = {5,3}
    A = {2, -13, 'OI', C}
    
    C ∈ A = True
    5 ∈ A = False
    5 ∈ C = True

    ```


5. Not belongs

    The not belongs of an element x to a set A is true if x is not in A and false otherwise. The not belongs of x to A is denoted by x ∉ A.

    ```python
    A = {1, 2, 3, 4, 5}
    3 ∉ A = False
    6 ∉ A = True
    ```

### Subsets

A set A is a subset of a set B if every element of A is also an element of B. The subset of A is denoted by A ⊆ B.

```python

A = {1, 2, 3, 4, 5}
B = {1, 2, 3, 4, 5, 6, 7, 8}
A ⊆ B = True
B ⊆ A = False

{} ⊆ A = True
A ⊆ A = True
```

```python

A = {{1, 2}, {3, 4}}
B = {{1, 2}, {3, 4}, {5, 6}}
A ⊆ B = True
```

### Complement

The complement of a set A is the set of elements which are not in A. The complement of A is denoted by A'.

```python
universal_set = {1,2,3,4,5,8,10}
A = {1,2,3}
A' = {4,5,8,10}
```

### De Morgan's laws

1. The complement of the union of two sets is the intersection of their complements.

    ```python
    A = {1, 2, 3, 4, 5}
    B = {4, 5, 6, 7, 8}
    (A ∪ B)' = {1, 2, 3, 6, 7, 8}
    A' ∩ B' = {1, 2, 3, 6, 7, 8}
    ```
2. The complement of the intersection of two sets is the union of their complements.

    ```python

    A = {1, 2, 3, 4, 5}
    B = {4, 5, 6, 7, 8}
    (A ∩ B)' = {1, 2, 3, 6, 7, 8}
    A' ∪ B' = {1, 2, 3, 6, 7, 8}
    ```
### Disjoint sets

Two sets A and B are disjoint if they have no elements in common. In other words, A and B are disjoint if A ∩ B = ∅.

```python

A = {1, 2, 3}
B = {4, 5, 6, 7, 8}
A ∩ B = ∅ = True
```
### Joint sets

Two sets A and B are joint if they have elements in common. In other words, A and B are joint if A ∩ B ≠ ∅.

```python

A = {1, 2, 3}
B = {3, 4, 5, 6, 7, 8}
A ∩ B ≠ ∅ = True
```

### Exercises:

1. What is the union and intersection of a set with an empty set?

    ```python
    A = {1, 2, 3, 4, 5}
    B = {}
    A ∪ B = {1, 2, 3, 4, 5}
    A ∩ B = {}
    ```

    The union of a set with an empty set is the set itself. The intersection of a set with an empty set is an empty set.

2. What is a difference of a set from an empty set?

    ```python
    A = {1, 2, 3, 4, 5}
    B = {}
    A - B = {1, 2, 3, 4, 5}

    B - A = {}
    ```

    The difference of a set from an empty set is the set itself.

3. How many different 2-set partitions are there of a set having 10 elements?

    ```python
    A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    ```

    The number of different 2-set partitions of a set having n elements is given by the formula:

    ```python
    n! / (2! * (n - 2)!)
    ```

    ```python
    10! / (2! * (10 - 2)!) = 45
    ```

    There are 45 different 2-set partitions of a set having 10 elements.


4. What are collections that have ordered elements called?

    Sequences.

5. Write a python function that decides weather a given list of sets forms a partition of set or not.

    ```python
    def is_partition(sets, universal_set):
        union = set()
        for s in sets:
            union = union.union(s)
        return union == universal_set
    ```

6. Write a python code to verify the following identities about sets:

    (Ac)' = A'

    A - B = A ∩ B'

    ```python

    A = {1, 2, 3, 4, 5}
    B = {4, 5, 6, 7, 8}

    (A - B)' = {1, 2, 3}
    A' ∩ B' = {1, 2, 3}
## Experiment

An experiment is a procedure that yields one of a given set of possible outcomes. The set of all possible outcomes of an experiment is called the sample space of the experiment.

### Random experiment / Random trial

A random experiment is an experiment whose outcome cannot be predicted with certainty.

Examples:

- Tossing a coin.

- Rolling a die.

- Drawing a card from a deck of cards.

#### Outcomes

Outcomes basically is the result of an experiment.

#### Sample space

The sample space of an experiment is the set of all possible outcomes of the experiment. The sample space is usually denoted by S or Ω.

#### Exercices

What is the sample space for an experiment involving rolling a 4-sided dice 3 times along with tossing coing?

    There are 128 possible outcomes.
    The outcomes are: (111T, 111H, 112T, 112H, 113T, 113H, ..., 444T, 444H)


#### Events

An event is a subset of the sample space of an experiment. An event is said to occur if the outcome of the experiment is an element of the event. Including the empty set and the sample space itself.

#### Exercices

How many possible events are there for an experiment having a sample space of
size 16?

    There are 2^16 possible events.

What are disjoint events?

    Two events are disjoint if they have no outcomes in common. In other words, two events are disjoint if their intersection is an empty set.

Experiment: A four-sided die is rolled repeatedly until the first time (if ever)
that an even number is rolled.

What is the sample space of this experiment? Is the sample space finite or not?

    The sample space is: {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, ...}
    The sample space is infinite.

Write down the event that the sum of rolls of the expiriment/trials is even

    {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, ...}

Is the above event finite? 

    No, the event is infinite.


## Probability Model

A probability model is a mathematical description of an experiment consisting of the sample space S and a probability for each outcome in S.

### Probability Law

A probability law is a function that assigns a probability to each outcome in the sample space of an experiment.

### Axioms of Probability

1. The probability of an event is a non-negative real number.

    ```python
    P(A) >= 0
    ```

2. The probability of the sample space is 1.

    ```python 
    P(S) = 1
    ```

3. If A and B are disjoint events, then the probability of the union of A and B is the sum of the probabilities of A and B.

    ```python
    P(A ∪ B) = P(A) + P(B)
    ```

It's possible for an empty set to have a non-zero probability? 

    No, the probability of an empty set is always zero.

## Probability Model: Conditioning

Two dices each with 6 faces. Whats the probability of at least one of the two dice will show 6? 

    First imagine a sample space with all possible outcomes. The sample space is the set of all possible outcomes of an experiment. The sample space is usually denoted by S or Ω.

    | Die 1 \ Die 2 | 1  | 2  | 3  | 4  | 5  | 6  |
    |--------------|----|----|----|----|----|----|
    | 1            | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 |
    | 2            | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 |
    | 3            | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 |
    | 4            | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 |
    | 5            | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 |
    | 6            | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 |

    Knowing that, we can look at the rows and columns and see that the probability of at least one of the two dice will show 6 is 11/36.

Now, what's the probability of the sum will be 9 and at least one of the two dice will show 6?

    We can have 6 and 3, 5 and 4, 4 and 5, 3 and 6. So, the probability is 4/36. But we want only the cases where at least on of the two dice will show 6. So, the cases will be 6 and 3 and 3 and 6. So, the probability is 2/36.

### Conditional Probability in Machine Learning

The biggining of machine learning is the conditional probability.

This is extremely important in regression and classification problems.

## Probability Model: Law of Total Probability

The law of total probability is a fundamental rule relating marginal probabilities to conditional probabilities. It expresses the total probability of an outcome which can be realized via several distinct events.

### Independence 

Two events A and B are independent if the probability of A and B is the product of the probabilities of A and B.

```python

P(A ∩ B) = P(A) * P(B)
```

``` python
P(A|B) = P(A). Then A does not depend upon B
```

```python
If P(A|B) = A 
Means that P(A|B) = B?
```
For more then two events the independency holds if the probability of the intersection of all events is the product of the probabilities of all events.

```python

P(A ∩ B ∩ C) = P(A) * P(B)
             = P(A) * P(C)
             = P(B) * P(C)
```

### Conditional Independence

Two events A and B are conditionally independent given a third event C if the probability of A and B given C is the product of the probabilities of A and B given C.

```python
P(A ∩ B | C) = P(A | C) * P(B | C)
```

### Exercise:

Come up with an example where two dependent events become conditionally independent

A box contains two coins: a regular coin and one fake two-headed coin (P(H) = 1). I choose a coin at random and toss it twice. Define the following events.

    A = First coin toss results in an H.

    B = Second coin toss results in an H.

    C = Coin 1(regular) has been selected.

Find P(A|C), P(B|C), P(A ∩ B|C), P(A), P(B) and P(A ∩ B).

Note that A and B are NOT independent, but the are conditionally independent given C.

    P(A|C) = 1/2
    P(B|C) = 1/2
    P(A ∩ B|C) = 1/4
    P(A|C) * P(B|C) = 1/4
    P(A) = 3/4
    P(B) = 3/4
    P(A ∩ B) = 9/16

## Bayes Rule

Bayes rule is a fundamental rule in probability that allows one to calculate conditional probabilities. It is a direct consequence of the definition of conditional probability.

```python

P(A|B) = P(B|A) * P(A) / P(B)
```

Prove:

    A ∩ B = B ∩ A
    P(A|B) * P(B) = P(B|A) * P(A)
    P(A|B) = P(B|A) * P(A) / P(B)
All machine learning algorithms are based on Bayes rule.

## Towards Random Variables

A random variable is a variable whose value is subject to variations due to chance. A random variable can take on a set of possible different values (similarly to other mathematical variables), each with an associated probability, in contrast to other mathematical variables.

### Discrete Random Variables

A discrete random variable is a random variable that can take on a countable number of values.

### Continuous Random Variables

A continuous random variable is a random variable that can take on an uncountable number of values.

Those variables is essential to understand the probability distributions, and they are applied in machine learning.

- Lets say that you roll a 4 sided die and toss two coins simultaneously. Build a probability model where all outcomes are equally likelly.

- What is the probability that the roll will result even and the tosses will both result heads?

    ```python
    P(A ∩ B) = P(A) * P(B)
    P(A ∩ B) = 1/4 * 1/4
    P(A ∩ B) = 1/16
    ```

## Random Variables

Real - valued functions defined on the sample space of an experiment are called random variables.

### Exercise

If probability of a particular value of a random variable is zero, does it correspond to an empty/impossible event? 

    Yes, that corresponds to an empty/impossible event. If a variable is really discrete, then it goes to am empty event.
    P(X = a) = 0
    X must be a continuous variable.


### Bernoulli Random Variable

First we must understand the difference between Discrete RV and Continuous RV.

**Discrete RV** is a RV that can take on a countable number of values. Example: The number of heads in 10 coin tosses.
Discreate does not means that the variables are integers. It means that the variables can take on a countable number of values. So the variable can be a float, but it must be a countable number of values.

**Continuous RV** is a RV that can take on an uncountable number of values. Example: The height of a person.

A Bernoulli random variable is a random variable that can take on two values, 0 and 1, and the probability of success is p and the probability of failure is 1 - p.

### Geometric RV

Supose that you keep tossing a coin until the first head appears.

$X$ = number of tosses

$X$ = 1, 2, 3, ...

$P(X = k) = (1 - p)^{k - 1} * p$

### Binomial RV

Supose that you toss a coin n times and count the number of heads.

$X$ = number of heads

$X$ = 0, 1, 2, ..., n

$P(X = k) = {n \choose k} * p^k * (1 - p)^{n - k}$

### Continuous RV

The best way to describe a continous RV is beyond darts. Imagine that the center of the board is (0,0) and you want to know the distance from the center to the dart. The distance is a continous RV.

Let say a line is already divided into 6 disjoint intervals of possibly different, but known, lengths. You select an interval at random by rolling a die. Let X is a random variable that represents the mid point of the selected interval. Is X a continuous random variable?
    
        No, X is a discrete random variable. Because X can take on a countable number of values.

#### Probability Density Function (PDF)

The probability density function (PDF) of a continuous random variable is a function that can be integrated to obtain the probability that the random variable takes a value in a given interval.

The PDF common uses integrals to calculate the probability of a continous RV.

The formula of the PDF is:

$$\int_{-\infty}^{\infty} f(x) dx = 1$$

The PDF is a function that describes the relative likelihood for this random variable to take on a given value.

The proprierties of the PDF are:

1. $f(x) \geq 0$ for all $x$.

2. $\int_{-\infty}^{\infty} f(x) dx = 1$

3. $P(a \leq X \leq b) = \int_{a}^{b} f(x) dx$

4. $P(X = a) = 0$

5. $P(a \leq X \leq b) = P(a < X < b)$

6. $P(X \leq a) = P(X < a)$

    ...


### Exponential RV

It's always non-negative, can take elevated values and it's continuous .

$f(x) = \lambda e^{-\lambda x}$

$\lambda$ > 0

$X$ = 0, 1, 2, ...

What is the impact of the lambda in exponencial RV?
    
        The lambda is the rate parameter. The higher the lambda, the higher the rate of the exponential RV.


### Gaussian RV

Great contribution to the machine learning.

$f(x) = \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2} (\frac{x - \mu}{\sigma})^2}$

$\mu$ = mean

$\sigma$ = standard deviation

$X$ = -$\infty$, $\infty$

X at zero is the highest point of the curve.

The impact of the mean and standard deviation in the gaussian RV is the same as the impact of the mean and standard deviation in the normal distribution.


### What is a Cumulative Distribution Function (CDF)?

The cumulative distribution function (CDF) of a random variable X is the probability that X will take a value less than or equal to x.

$F(x) = P(X \leq x)$

The proprierties of the CDF are:

1. $F(x)$ is non-decreasing.

2. $\lim_{x \to -\infty} F(x) = 0$

3. $\lim_{x \to \infty} F(x) = 1$

4. $P(a \leq X \leq b) = F(b) - F(a)$

---

## Expectation

If X is a discrete random variable with probability mass function $f(x)$, then the expected value of X is defined by:

$E(X) = \sum_{x} x * f(x)$

If X is a continuous random variable with probability density function $f(x)$, then the expected value of X is defined by:

$E(X) = \int_{-\infty}^{\infty} x * f(x) dx$

The expected value of a random variable is a measure of the center of the distribution of the random variable.

The proprierties of the expected value are:

1. $E(aX + b) = aE(X) + b$

2. $E(X + Y) = E(X) + E(Y)$

Expectation is a linear operator. It means that the expected value of a sum of random variables is the sum of the expected values of the random variables.

### Law of Large Numbers

The law of large numbers states that as the number of trials of a random experiment increases, the empirical probability of an event will converge to the theoretical probability of the event.

I.I.D - Independent and Identically Distributed Random Variables

I.I.D Sample/Data, for example 2,4,7,9,2,0,1,3. We assume that the exists any random variable, for example X, with certain function.

$X$, $P_X(x)$

That random variable has a certain expected value, for example $E(X)$.

$E(X) = \sum_{x} x * P_X(x)$

The expectated value is called the population mean or true mean.

Sample mean is the mean of the sample/data. In the example above, the sample mean is 4 ( ( 2 + 4 + 7 + 9 + 2 + 0 + 1 + 3 ) / 8.)

All the random variable have the same distributions and the same expected value.

| $R.V$ | $E[X]$ | $x range$ | $Parmrange$|
|-------|--------|-----------|------------|
| $Bernoulli(P)$ | $P$ | $0,1$ | $0 \leq P \leq 1$ |
| $Binomial(n, P)$ | $n * P$ | $0,1,2,...,n$ | $0 \leq P \leq 1$ |
| $Geometric(P)$ | $1/P$ | $X >= 0 $ | $0 \leq P \leq 1$ |
| $Poisson(\lambda)$ | $\lambda$ | $X >= 0,1,...$ | $\lambda > 0$ |
| $Normal(\mu, \sigma^2)$ | $\mu$ | $X >= -\infty, \infty$ | $\mu \in \mathbb{R}, \sigma^2 > 0$ |
| $Exponential(\lambda)$ | $1/\lambda$ | $X >= 0$ | $\lambda > 0$ |

As the sample size increases the sample mean will actually aproach to the population mean.

### Transformations of Random Variables

If X is a random variable with probability density function $f(x)$, then the random variable Y = g(X) has probability density function $f_Y(y)$ given by:

$f_Y(y) = f_X(g^{-1}(y)) * | \frac{d}{dy} g^{-1}(y) |$

---
### Variance

It's the expected value of the squared deviation from the mean.

If X is a random variable with expected value $\mu$, then the variance of X is defined by:

$Var(X) = E((X - \mu)^2)$

The proprierties of the variance are:

1. $Var(X) = E(X^2) - E(X)^2$

2. $Var(aX + b) = a^2Var(X)$

---

#### What is the expected value of a transformation of a random variable?

$E[y]$ Where y is some function of x.

$E[y] = \sum_{y} y * P_{x}(y)$

### Joint Probability Distribution

The joint probability distribution of two discrete random variables X and Y is a function that assigns a probability to each pair of values (x, y) that X and Y can take on.

$P(X = x, Y = y)$

The proprierties of the joint probability distribution are:

1. $P(X = x, Y = y) \geq 0$ for all x and y.

2. $\sum_{x} \sum_{y} P(X = x, Y = y) = 1$

3. $P(X = x, Y = y) = P(X = x) * P(Y = y)$ if X and Y are independent.

### Multivariate Gaussian Distribution

The multivariate Gaussian distribution is a generalization of the univariate Gaussian distribution to two or more variables.

$x_1$, $x_2$, ..., $x_n$, x<sup>-></sup> = a vector of random variables.

$f_{x}$ = joint probability density function of $x_1$, $x_2$, ..., $x_n$.

$\mu$ = mean vector

$\Sigma$ = covariance matrix

$f_{x}(x) = \frac{1}{(2 \pi)^{n/2} |\Sigma|^{1/2}} e^{-\frac{1}{2} (x - \mu)^{T} \Sigma^{-1} (x - \mu)}$

The proprierties of the multivariate gaussian distribution are:

1. $f_{x}(x) \geq 0$ for all x.

2. $\int_{-\infty}^{\infty} ... \int_{-\infty}^{\infty} f_{x}(x) dx_1 ... dx_n = 1$

### Curse of Dimensionality

The curse of dimensionality refers to various phenomena that arise when analyzing and organizing data in high-dimensional spaces (often with hundreds or thousands of dimensions) that do not occur in low-dimensional settings such as the three-dimensional physical space of everyday experience.

The curse of dimensionality is the reason why we use PCA (Principal Component Analysis) to reduce the dimensionality of the data.

## Codes

### Python naive bayes implementation

The implementation of the naive bayes in python using iris dataset can be found [here](/statistics%20in%20python/naive_bayes.py)

### Expectations

The implementation of the expectations in python can be found [here](/statistics%20in%20python/expectations.ipynb)

### Random Variables

The implementation of the random variables in python can be found [here](/statistics%20in%20python/RV.ipynb)

And the RV in dataset can be found [here](/statistics%20in%20python/RV_in_datasets.ipynb)

### Sets

The implementation of the sets in python can be found [here](/statistics%20in%20python/set_practice.ipynb)

## Estimation

Estimation is the process of inferring the parameters of a distribution given some data.

$Normal(\mu, \sigma^2)$

$exponential(\lambda)$

$Geometric(p)$

$Binomial(n, p)$

This are the parameters of the distributions.

### Non parametric estimation

Non parametric estimation is the process of inferring the distribution of a random variable without assuming any parametric form for the distribution.

### Maximum Likelihood Estimation (MLE)

Maximum likelihood estimation is a method of estimating the parameters of a distribution by maximizing a likelihood function, so that under the assumed statistical model the observed data is most probable.

First of all assume that you have a lot of data, and the first data is i.i.d which means all the data is indendent and identically distributed.

$x_1, x_2, ..., x_n$

Assume that all sample points belong to a certain distribution with parameters $a, b, c$. The goal is to find the parameters $a, b, c$. 

Let's define an vector of parameters $\theta = (a, b, c)$.

MLE uses the following thing:

- All the samples are independent and identically distributed. Which means the 
$f(x_{1}, x_{2} ... )$ = $f_{x_{1}} * f_{x_{2}} * ... * f_{x_{n}}$ Our goal is to optime the product: $f_{x_{1}} * f_{x_{2}} * ... * f_{x_{n}}$ and what values of a, b and c maximizes this product.

#### Log Likelihood

The log likelihood is the log of the likelihood function. The log likelihood is used because it's easier to work with sums than products.

Assume for example that we have a lot of data, for example $x_1, x_2, ..., x_n$ and we want to find the parameters $\theta$ that maximizes the likelihood function.

As we know the density for a exponential random variable is given by $f(x) = \lambda e^{-\lambda x}$

$L(\lambda)$ = $\lambda e^{-\lambda x_{1}} * \lambda e^{-\lambda x_{2}} * ... * \lambda e^{-\lambda x_{n}}$

If we simplify this we have:

$L(\lambda)$ = $\lambda^{n} e^{-\lambda(x_{1}+x_{2}+...+x_{n})}$

We can have: $\lambda^{n}e^{-\lambda s}$ Where s is the sum of all the samples.

If we put a log in the likelihood function we have:

$log(L(\lambda))$ = $nlog(\lambda) - \lambda s$

The log likelihood is easier to work with because we can use sums instead of products.

Logistic regression is a classification algorithm that uses the log likelihood.

### Maximum A Posteriori Estimation (MAP)

Maximum a posteriori estimation is a method of estimating the parameters of a distribution by maximizing a posterior distribution, so that under the assumed statistical model the observed data is most probable.

In MLE we have a lot of data and we want to find the parameters that maximizes the likelihood function.

In map we think the parameter is lambda is a random variable and we want to find the distribution of lambda.

$\prod_{i=1}^{n} f_{x}(x_{i}) * f_{y}(\lambda)$

It's extremely important in Machine Learning. Whatever you apply regularization in machine learning models in reality you are applying MAP.

### Logistic Regression

Logistic regression is extremely powerful is compared with classic models in machine learning like linear regression. Deep learning is constructed based on logistic regression.

Logistic regression is a classification algorithm that uses the log likelihood.

i lov dick
