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


