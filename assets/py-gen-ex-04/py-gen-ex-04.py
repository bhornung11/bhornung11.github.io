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
    chain,
    compress,
    islice,
    repeat
)
import os.path
import sys

# %%
# TO HIDE -- SETUP

DIR_SCRIPT = os.path.realpath("dir_script")

# %%
# TO HIDE -- SETUP

sys.path.append(DIR_SCRIPT)

# %%
# TO HIDE -- SETUP

from src.generators.batches import (
    serialiser,
    make_batcher,
    loop_terminate_batch_function
)

from src.generators.multi_input import (
    compressor,
    gater,
    merger,
    switcher,
    zipper
)

from src.util.printer import (
    print_source_with_trimmed_doc
)

# %% [markdown]
# The third post on generators briefly discusses generators which consume multiple streams of inputs.
#
# ## Note
#
# As per usual, the raw notebook has been made accessilbe [here], whereas the scripts are version in [this repository]().
#
# ## Introduction
#
# Multiple inputs can have multiple roles. They can be values which carry the same meaning and face a fate alike when they are consumed. It might happen that some of them are auxilliary elements that govern how the others are processed. This distiction defines two broad groups of generators that combine multiple inputs.
#
# 1. `mergers` that coerce the inputs streams to single output sequence
# 2. `governers` where some of the input streams determine when and how the elements of the sibling series are consumed

# %% [markdown]
# ## Mergers
#
# ### Zipper
#
# The Ur-itertool. The `zipper` function advances multiple iterators and yields. their current elements as a single output. Its slightly more famous twin is the builtin `zip`. Please note, `zipper` yield generators over the collated objects as opposed to a tuple which is served by `zip. The `loop_terminate_batch_function` produces a sequence of the zipped elements which terminated once the feed iterator of the fewest elements is exhausted.
#
# ```python
# def zipper(*iterators) -> Generator:
#     """
#     Collates elements from multiple iterators and yields the
#     bundle as a single element.
#     """
#
#     def inner():
#         for iterator in iterators:
#             try:
#                 yield next(iterator)
#             except StopIteration:
#                 return
#
#    return return loop_terminate_batch_function(inner)
# ```
#
# To sequences are paired until the shortest one is exhausted in the example below.

# %%
zipped = zipper(iter("abcdef"), iter(range(12)))

print("From DIY:")
print("\t", " ".join(str(tuple(pair)) for pair in zipped))

zipped = zipper(iter("abcdef"), iter(range(12)))

print("Builtin:")
print("\t", " ".join(str(tuple(pair)) for pair in zipped))

# %% [markdown]
# ### Merger
#
# The elements from multiple sources are yielded sequentially one-by-one not grouped as in `zip`. `merger` represents one possible way of creating such a utility.
#
# ```python
# def merger(*iterators) -> Generator:
#     """
#     Merges (interlaces) iterators.
#     """
#
#     while True:
#         for iterator in iterators:
#             try:
#                 yield next(iterator)
#             except StopIteration:
#                 return
# ```
#
# Two sequences are conveniently interlaced:

# %%
merged = merger(iter("abcdef"), iter(range(12)))

print("DIY:")
print("\t", " ".join(str(el) for el in merged))

merged = chain.from_iterable(zip(iter("abcdef"), iter(range(12))))

print("itertools:")
print("\t", " ".join(str(el) for el in merged))

# %% [markdown]
# ## Governers
#
# ### Gater
#
# The next element of an iterator is yielded if a condition is satisfied. No elements produced of discarded is the condition is false. This functionality is implemented by `gater`.
#
# ```python
# def gater(
#         iterator: Iterator,
#         selector: Iterator
#     ) -> Generator:
#     """
#     Gate generator. The next element of an iterator is yielded
#     when the selector is true.
#     """
#
#     while True:
#         try:
#             # only advance `iterator` if the condition is met
#             # no elements are discarded
#             if next(selector):
#                 yield next(iterator)
#
#         except StopIteration:
#             return
# ```
#
# The yielding of the elements is governed by a binary pattern:

# %%
gated = gater(iter("abcdef"), iter([1, 0, 0, 1, 1, 1]))

print("\t", " ".join(gated))

# %% [markdown]
# ### Compressor
#
# This function is our `DIY` implementation of the `itertools.compress`. Only those elements are yielded where the aligned condition is True. All other elements are discarded.
#
# ```python
# def compressor(
#         iterator: Iterator,
#         selector: Iterator
#     ) -> Generator:
#     """
#     Compressor generator. The an element of an iterator is yielded
#     when the selector is true.
#     """
#
#     while True:
#         try:
#             element = next(iterator)
#             if next(selector):
#                 yield element
#
#         except StopIteration:
#             return
# ```
#
# The same binary pattern results in omission of elements:

# %%
compressed = compressor(iter("abcdef"), iter([1, 0, 0, 1, 1, 1]))

print("DIY:")
print("\t", " ".join(compressed))

compressed = compress(iter("abcdef"), iter([1, 0, 0, 1, 1, 1]))

print("itertools:")
print("\t", " ".join(compressed))

# %% [markdown]
# ### Switcher
#
# This generator is explicitly instructed as to from which input iterable the element must be taken at each yield.
#
# ```python
# def switcher(
#         iterators: Tuple[Iterator],
#         switch: Iterator
#     ) -> Generator:
#     """
#     Selects elements from iterators based on the iterators' indices.
#     """
#
#     for which in switch:
#         try:
#             yield next(iterators[which])
#
#         except StopIteration:
#             return
# ```
# Let us jump between three input iterators:

# %%
switched = switcher(
    [iter("abcdef"), iter("ABCDEF"), iter("ZYXWVU")],
    iter([1, 0, 1, 0, 2, 0, 2, 1,1,1,1,1,1,1,])
)

print("\t", " ".join(switched))
