# Statistics in Python

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
