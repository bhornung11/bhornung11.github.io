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

import os.path
import sys

from typing import Callable, Any, List, Dict, Tuple

# %%
# TO HIDE -- SETUP

DIR_SCRIPT = os.path.realpath("dir_script")

# %%
# TO HIDE -- SETUP

sys.path.append(DIR_SCRIPT)

# %%
# TO HIDE -- SETUP

from src.generators.multiplexer import (
    multiplexer,
    PotManager,
    TeeCup,
    TeePot
)

from src.coroutines.multiplexer_coro import (
    collector_coro,
    filter_coro,
    multiplex_coro
)

from src.util.printer import (
    print_source_with_trimmed_doc
)

# %% [markdown]
# This, third, part of the series on generators presents utilities that produce multiple generators from a single input iterable.
#
# ## Note
#
# The raw notebook can be found here. This folder contains the scripts ivoked therein.
#
# ## Introduction
#
# The previous posts followed the, rather tedious, implementation of generators which consumed a single iterable input. A handful of basic roles were discussed which were then combined to patterns. Batches were then discussed in no short paragraphs. The current notes further the complexity mostly in breadth. It will be shown that it is not possible multiplexing without storing the elements which were yielded by at least one but not all generators. A simple striplet of classes will then be implemented to store, handle and consume the elements of the original generator.

# %% [markdown]
# ## Multiplexer
#
# The multiplexer creates copies of a generator. Each copy can be manipulated without affecting the states of the others. It is thus evident that the elements which have been yielded by one of generators but not by all of them need to be stored.
#
# ### Design
#
# The principles by which our version of `itertools.tee` is implemented are briefly discussed below. The principle is to separate parts according to concerns.
#
# 1. multiplexed generator: its role is to
#     1. yield elements in the correct order, close itself
# 2. buffer: a shared resource which stores
#     1. the base iterator
#     2. the elements which have been yield by one but not all generators
#     3. and some lightweight bookkeeping information
# 3. buffer manager:
#     1. updates the buffer with the correct elements
#     2. removes the correct elements from the buffer
#
# ### Components
#
# These three constituents of the multiplexer detailed one by one.
#
# #### Buffer
#
# The mutable structure, `TeePot` collates the data shared by the generators which is the base iterator (`iterator`) and the buffer (`buffer`). It stores how many elements have been yielded from the iterator (`n_yielded`). The number of output generators (`n_gen`) is used to initialise a lookup table of many elements each generator has yielded (`generator_positions`).
#
# ```python
# @dataclasses.dataclass
# class TeePot:
#     """
#     Class to hold the shared resources and bookkeeping variables
#     of multiplexed iterators.
#     """
#
#     iterator: Iterator
#
#     n_gen: int
#
#     n_yielded: int = 0
#
#     buffer: Dict[int, Any] = dataclasses.field(
#         default_factory=dict
#     )
#
#     generator_positions: Dict[int, int] = dataclasses.field(
#         default_factory=dict
#     )
#
#     def __post_init__(self) -> None:
#         """
#         Initialises the generator positions.
#         """
#         self.generator_positions = {
#             i: - 1 for i in range(self.n_gen)
#         }
# ```

# %% [markdown] editable=true slideshow={"slide_type": ""}
# #### Buffer manager
#
# The `PotManager` takes care of providing the generators with the appropriate elements. It also keeps the buffer as slim as possible. 
#
# ##### Creating a new instance
#
# It is initialised by attaching the resource to it:
#
# ```python
# pot_manager = PotManager(teepot)
# ```
#
# ##### Retrieving elements
#
# A `generator` sends a request to the manager to yield the n-th element of the original `iterator`. There are two cases as to the status of the demanded element.
#
# i) If an element has not been yielded from the underlying iterator it is retreived by the `next(self.teepot.iterator)` call. The `idx` argument specifies which generator has called the manager. `pos` is the index of the element requested by this generator. If it is smaller than the number of elements yield from the underlying iterator (`n_yielded`) it means that a fresh element is taken. This is achieved by the calling the `next` function. `n_yielded` is incremented accordingly afterwards and the element is saved in the buffer before returning to the outside generator. `StopIteration` exceptions will propagate to the `__next__` method of the actual generators and will be handled there.
#
# ii) If the element has already been given to a generator it is retrieved from the buffer where it is stored under its index. If the element has been relayed to all generators it is removed from the buffer by `_trim_buffer`.
#
#
# ```python
# def yield_next(self, idx: int, pos: int) -> Any:
#         """
#         Produces the next element from the selected generator.
#         """
#
#         if pos > self.teepot.n_yielded:
#             raise IndexError(
#                 "Iteration ahead of iterator. This should not happen..."
#             )
#
#         # take an element from the underlying iterator (1st access)
#         if pos == self.teepot.n_yielded:
#
#             element = next(self.teepot.iterator)
#
#             self.teepot.buffer[pos] = element
#             self.teepot.n_yielded += 1
#             self.teepot.generator_positions[idx] = pos
#
#             return element
#
#         # take an element from the buffer (subsequent accesses)
#         if pos < self.teepot.n_yielded:
#
#             element = self.teepot.buffer[pos]
#
#             self.teepot.generator_positions[idx] = pos
#             self._trim_buffer(self.teepot)
#
#             return element
# ```
#
# ##### Trimming the buffer
#
# The simplistic `_trim_buffer` method removes the elements from the buffer which have already been yielded by all generators in three passes.
# 1. it finds the lowest generator position. 
# 2. Elements at these and at any smaller keys have surely been yielded by all generators. They are thus marked.
# 3. Finally these elements are deleted from the buffer.
#
# ```python
#     @staticmethod
#     def _trim_buffer(teepot: TeePot) -> None:
#         """
#         Removes the elements from the shared resources which have
#         already yielded by all generators.
#         """
#
#         pos_min = min(teepot.generator_positions.values())
#         positions_to_remove = [pos for pos in teepot.buffer if pos <= pos_min]
#
#         for pos in positions_to_remove:
#             del teepot.buffer[pos]
# ```
#
# This method is static because it does not require any information that is only accessible through the instance it is tied to. By the same token, the `yield_next` method could have been make static on the expense of registering the resource (`TeePot` instance) it acts upon. It is also worth noting that the cleanup can happen after any wisely crafted trigger. It is not demanded to be executed after each `yield` from the buffer. It can be called, for instance, if the number elements exceed a threshold. This is especially advantageous if this bookkeeping operation is expensive.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Generator
#
# A generator, `TeeCup` has an id to identify itself to the buffer manager and instance of which it is tied to it.
#
# ```python
# teecup = TeeCup(idx, pot_manager)
# ```
#
# Its `__next__` method requests and element from the manager and sets the number of elements yielded by itself.
#
# ```python
#     def __next__(self) -> Any:
#         """
#         Yields the subsequent element of a multiplexed generator.
#         """
#
#         element = self.pot_manager.yield_next(self.idx, self.pos)
#         self.pos = self.pos + 1
#
#         return element
# ```

# %% [markdown]
# ### Multiplexer utility
#
# The `multiplexer` function initialises the shared resource, adds it to the manager. The required number of generators are created each of which has the same manager linked.
#
# ```python
# def multiplexer(
#         iterator: Iterator,
#         n: int
#     ) -> Tuple[Generator]:
#     """
#     Creates indenpendent and identiacal generators from an iterator.
#     """
#
#     teepot = TeePot(iterator, n)
#
#     pot_manager = PotManager(teepot)
#
#     multiplexed = tuple(
#         TeeCup(i, pot_manager) for i in range(n)
#     )
#
#     return multiplexed
# ```
#
# ### Usage
#
# A toy example of the usage of the generator multiplexer is shown below:

# %%
gen1, gen2, gen3 = multiplexer(
    (chr(i) for i in range(97, 123)), 3
)

print("From generator 1.:", " ".join(next(gen1) for i in range(5)))
print("From generator 2.:", " ".join(next(gen2) for i in range(5)))
print("From generator 3.:", " ".join(next(gen3) for i in range(5)))
print("From generator 1.:", " ".join(next(gen1) for i in range(5)))
print("From generator 1.:", " ".join(gen1))
print("From generator 2.:", " ".join(gen2))
print("From generator 3.:", " ".join(gen3))

# %% [markdown]
# ## Compound generators
#
# Useful patterns emerge when the multiplexer function is combined with other generators. We, this time, only discuss one of them.
#
# ### Multiplexer + filter $\rightarrow$ separator
#
# The elements of each output generator can be filtered. By doing so, a stream of mixed element is separated to sources that are homogenous according to user defined criterion or multiple criteria.

# %%
n = 3
multiplexed = multiplexer(iter(range(20)), n)

separated = [
    filter(lambda x, y=i: x % n == y, multiplexed[i])
    for i in range(n)
]

for i in range(n):
    print(f"From generator {i}.:", list(separated[i]))

# %% [markdown]
# ## Coroutine multiplexer
#
# Generators pull data. This means that a filter generator can choke the entire pipeline if there is no element that satifies its condition, hence the need for buffers. A buffer can store all elements; importantly those that are rejected by a particular filter so that the others can advance.
#
# Is it possible to do away with the buffer if the data is pushed to the filters instead. This can be achieved by coroutines. A classic discussion of this class of constructions is available [here](https://www.dabeaz.com/coroutines/Coroutines.pdf). 
#
# ### Multiplexer
#
# The `multiplex_coro` function sends an element to multiple destinations, so-called targets. The element is sent to this function too and it is received by the `element = (yield)` instruction. The `start_coro` decorator initialises the coroutine.
#
# ```python
# @start_coro
# def multiplex_coro(targets: List[Callable]) -> Generator:
#     """
#     Sends an element to multiple coroutines.
#     """
#
#     while True:
#         element = (yield)
#         for target in targets:
#              target.send(element)
# ```

# %% [markdown]
# ### Filter
#
# The `filter_coro` sends an element to a target only if it satisfies a condition.
#
#
# ```python
# @start_coro
# def filter_coro(cond: Callable, target: Callable) -> Generator:
#     """
#     Sends elements to a target which satisfy the specified condition.
#     """
#
#     # this loop is needed to keep the coroutine alive
#     # otherwise it would exit after the first `send`
#     while True:
#         element = (yield)
#         if cond(element):
#             target.send(element)
# ```

# %% [markdown]
# ### Collector
#
# Unlike building chains of generators where the source needs to be specified at first, coroutines are to be linked starting from the final target which consumes the elements for good. `collector_coro` stashes elements in a list buffer.
#
# ```python
# @start_coro
# def collector_coro(buffer: List[Any]) -> Generator:
#     """
#     Collects sent elements in a list.
#     """
#
#     while True:
#         element = (yield)
#         buffer.append(element)
# ```

# %% [markdown]
# ### Example coroutine filter
#
# The same functionality as in the generator example a few paragraphs above is implemented by the way of coroutines here.

# %%
n = 3

# final targets -- the buffers
separated = [[] for i in range(n)]

# direct output of the filters to the buffers
filters = [
    filter_coro(
        lambda x, y=i: x % 3 == y, collector_coro(separated[i])
    ) for i in range(n)
]

# add targets to the multiplexer
separator = multiplex_coro(filters)

# push values through the pipeline
for i in range(20):
    separator.send(i)

for i in range(3):
    print(f"From coro {i}.:", list(separated[i]))
