# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# TO HIDE -- SETUP

from itertools import (
    compress,
    islice
)

import os.path
import sys

from random import random

from typing import (
    Any,
    Callable,
    Generator,
    Iterator,
    List
)

# %%
# TO HIDE -- SETUP

DIR_SCRIPT = os.path.realpath("dir_script")

# %%
# TO HIDE -- SETUP

sys.path.append(DIR_SCRIPT)

# %%
# TO HIDE -- SETUP

from src.generators.basic import (
    filtr,
    identity,
    repeater,
    thinner
)

# %% [markdown]
# A handful of `python` generator exercises are presented in this post. The aim is to accummulate an inventory of simple manipulations upon which useful utilities can be constructed. These both consume and produce a single stream of elements.
#
# ## Note
#
# The raw notebook is has been made accessible here. The scripts are deposited in this folder.
#
# ## Introduction
#
# ### Generators
#
# What `python` generators are can be discussed until the cows come home. For the purposes of these notes, the generator is a function that calves as an iterator which
# * yields an single element at a time
# * on demand
# * the iteration over which can be paused
# * and resumed
# * by maintaining a state
#
# ### Coding style
#
# The `itertools` standard library equips `python` with utilities by means of which it is possible to implement a large proportion the transformation patterns discussed herein more efficiently. In fact, we will be creating inefficient `C` code for the most part below.
#
# Docstrings shown here are constrained to show a short summary of the functions causing a considerable discomfort of the scribbler of these lines. This is in order to render the script entries a tad more terse to read.
#
# The iterable inputs of the functions are always `iterators`. These can created by the `iter` function applied to any iterable object. For the greater part, it is fairly inconsequential whether an `iterable` of any sorts, such as lists, strings or generators are the inputs to our functions. 
#
# ## Basic generators
#
# These generators perform simple tasks. Most of them are also available as either built-ins or in the `itertools` library. They reimplemented here as they provide a gradual introduction to more complex patterns.
#
# ### Identity
#
# The identity generator is added for the sake of completeness. The elements of the original iterator are yielded unchanged. Its factory is implemented in the `identity` function. Alternatively, a single line of `yield from generator` would suffice if the input is a `generator`.
#
# ```python
# def identity(iterator: Iterator) -> Any:
#     """
#     Creates a generator that yields
#     the elements of the consumed iterator
#     """
#
#     for element in iterator:
#         yield element
# ```
#
# The intricate behaviour of this utility is demonstrated below.

# %%
string = "abcdefghjikl"

print("DIY:")
iden_string = identity(iter(string))
print("\t", " ".join(next(iden_string) for i in range(6)))


print("\nbuiltin:")
iden_string = (char for char in string)  # this is a generator
print("\t", " ".join(next(iden_string) for i in range(6)))

# %% [markdown]
# ### Repeating
#
# An element is repeated multiple times in the generator returned by the `repeater` function.
#
# ```python
# def repeater(iterator: Iterator, n: int):
#     """
#     Creates a generator that repeats each element of the original
#     iterator at specified times.
#     """
#
#     for element in iterator:
#         for i in range(n):
#             yield element
# ```
#
# Like so:

# %%
print("DIY:")
repeat_string = repeater(iter(string), 3)
print("\t", " ".join(next(repeat_string) for i in range(6)))

print("\nbuiltin")
repeat_string = (char for char in iter(string) for i in range(3))
print("\t", " ".join(next(repeat_string) for i in range(6)))

# %% [markdown]
# ### Thinning
#
# Every  n-th element is returned by the generator that is defined in the `thinner` function. Please note, that it maintains a state in addition to that of the `iterator` upon which it acts. This state is the actual value of the variables enclosed in its scope. In our case, that is `i`, the call counter.
#
# ```python
# def thinner(iterator: Iterator, n: int) -> Any:
#     """
#     Creates a generator that selects every n-th element
#     of the original iterator.
#     """
#
#     i = 0
#
#     for el in iterator:
#         if i % n == 0:
#             yield el
#         i += 1
# ```
#
# In action:

# %%
print("DIY:")
thin_string = thinner(iter(string), 2)
print("\t", " ".join(next(thin_string) for i in range(3)))

print("\nitertools:")
thin_string = compress(iter(string), (i % 2 == 0 for i in range(5)))
print("\t", " ".join(next(thin_string) for i in range(3)))

# %% [markdown]
# The `make_thinner` function is the left inverse of the `make_repeater`:

# %%
repeat_string = repeater(iter(string), 2)
thin_string = thinner(repeat_string, 2)
print(" ".join(thin_string))

# %% [markdown]
# ### Filter
#
# The `filter` function selects elements based on a condition. A toy implementation, `filtr` (TM) is provided below only for the sake of completeness.
#
# ```python
# def filtr(iterator: Iterator, cond: Callable) -> Any:
#     """
#     Makes a filtering generator. Only those elements yielded
#     at which the condition evaluates to true.
#
#    Parameters:
#         iterator: Iterator : iterator!
#
#     Yields:
#         element: Any : a filtered element
#     """
#
#     for element in iterator:
#         if cond(element):
#             yield element
# ```
#
# It is instructed to select the first two vowels of the English alphabet in the example below.

# %%
print("DIY:")
filt_string = filtr(iter(string), lambda x: x in set("ae"))
print("\t", " ".join(filt_string))

print("\nbuiltin:")
filt_string = filter(lambda x: x in set("ae"), iter(string))
print("\t", " ".join(filt_string))
