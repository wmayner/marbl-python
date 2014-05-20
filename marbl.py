#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  __  __            _     _        _____       _   _
# |  \/  |          | |   | |      |  __ \     | | | |
# | \  / | __ _ _ __| |__ | | ____ | |__) |   _| |_| |__   ___  _ __
# | |\/| |/ _` | '__| '_ \| ||____||  ___/ | | | __| '_ \ / _ \| '_ \
# | |  | | (_| | |  | |_) | |      | |   | |_| | |_| | | | (_) | | | |
# |_|  |_|\__,_|_|  |_.__/|_|      |_|    \__, |\__|_| |_|\___/|_| |_|
#                                          __/ |
#                                         |___/
"""
============
Marbl-Python
============

Marbl-Python is an implementation of the `Marbl specification
<https://github.com/wmayner/marbl>`_ for normalized representations of Markov
blankets in Bayesian networks.

It provides objects and methods for serializing and hashing **Marbls** (Markov
blankets normalized per the Marbl spec), and unordered collections of them.


Usage
~~~~~

Transition probability matrices are represented as ``p``-dimensional nested
lists of floats, where ``p`` is the number of the node's parents (this makes
lexicographic sorting trivial, since lists are natively sorted
lexicographically, though at the cost of overhead in the conversion between
NumPy array and Python list form).

Normalization
``````````````
A :class:`Marbl` represents the normal form of a Markov blanket. Its TPMs are
normalized upon initialization.

A :class:`MarblSet` is similarly used to represent the normal form of an
unordered collection of Marbls.

If you need to get the normal form of a single TPM, rather than an entire
Markov blanket, use :func:`normalize_tpm`.

Hashing
```````
Just use the native Python ``hash`` function on Marbls and MarblSets.

Serialization
`````````````
Both the Marbls and MarblSets can be serialized with :func:`pack`. Each object
also has its own ``pack()`` method.

To deserialize, use :func:`unpack` for Marbls, and :func:`unpack_set` for
MarblSets.


API
~~~
"""

from itertools import permutations
import collections.abc
import functools
import numpy as np
import hashlib
import msgpack


@functools.total_ordering
class Marbl():

    """A Markov blanket in normal form.

    Provides methods for serialization and hashing.

    Attributes:
        node_tpm (list): The normal form of the covered node's
            ``p``-dimensional transition probability matrix (where ``p`` is the
            number of the node's parents).
        augmented_child_tpms (list): The normalized augmented child tpms. A
            normalized augmented child TPM contains the index of the dimension
            corresponding to the covered node in the child's normalized TPM,
            and the normalized TPM itself.
    """

    def __init__(self, node_tpm, augmented_child_tpms,
                 already_normalized=False):
        """Marbls are rendered into normal form upon initialization.

        Args:
            node_tpm (list): The un-normalized node's TPM.
            augmented_child_tpms (Iterable): Each augmented child TPM contains
                the index of the dimension corresponding to the covered node in
                the child's TPM, and the TPM itself.

        Keyword Args:
            already_normalized (bool): Flag to indicate TPMs have already been
                normalized. Defaults to ``False``.

        Warning:
            Incorrect use of the ``already_normalized`` flag can cause hashes
            to differ when they shouldn't. Make sure everything really is
            already normalized.

        Examples:
            >>> tpm = [[[0.3, 0.4],
            ...         [0.1, 0.3]],
            ...        [[0.4, 0.5],
            ...         [0.3, 0.1]]]
            >>> tpm2 = [[[0.3, 0.1],
            ...          [0.4, 0.3]],
            ...         [[0.4, 0.3],
            ...          [0.5, 0.1]]]
            >>> augmented_child_tpms = [[0, tpm], [0, tpm]]
            >>> marbl1 = Marbl(tpm, augmented_child_tpms)
            >>> marbl2 = Marbl(tpm2, augmented_child_tpms)
            >>> marbl1 == marbl2
            True
            >>> augmented_child_tpms2 = [[0, tpm], [1, tpm]]
            >>> marbl2 = Marbl(tpm, augmented_child_tpms2)
            >>> marbl1 == marbl2
            False
        """
        # Cast Iterable to list
        augmented_child_tpms = list(augmented_child_tpms)
        # The underlying representation is the covered node's normalized TPM
        # followed by the normalized TPMs of its children, per the Marbl spec.
        if already_normalized:
            self._list = [node_tpm, augmented_child_tpms]
        else:
            self._list = [
                normalize_tpm(node_tpm),
                [normalize_tpm(tpm, track_parent_index=covered_node_index) for
                 covered_node_index, tpm in augmented_child_tpms]
            ]

    @property
    def node_tpm(self):
        return self._list[0]

    @property
    def augmented_child_tpms(self):
        return self._list[1]

    def __eq__(self, other):
        return self._list == other._list

    def __lt__(self, other):
        # Marbls are ordered lexicographically by their underlying lists.
        return self._list < self._list

    def __hash__(self):
        """Return the canonical hash of the Marbl.

        If two Marbls have the same hash, they are equivalent up to rearranging
        the labels of the covered node's parents and the covered node's
        children's parents.

        Example:
            >>> tpm = [[[0.3, 0.4],
            ...         [0.1, 0.3]],
            ...        [[0.4, 0.5],
            ...         [0.3, 0.1]]]
            >>> augmented_child_tpms = [[0, tpm], [1, tpm]]
            >>> marbl = Marbl(tpm, augmented_child_tpms)
            >>> hash(marbl)
            482032824703719516
        """
        return int(hashlib.sha1(self.pack()).hexdigest(), 16)

    def pack(self):
        """Serialize the Marbl.

        Example:
            >>> tpm = [[[0.3, 0.4],
            ...         [0.1, 0.3]],
            ...        [[0.4, 0.5],
            ...         [0.3, 0.1]]]
            >>> augmented_child_tpms = [[0, tpm]]
            >>> marbl = Marbl(tpm, augmented_child_tpms)
            >>> marbl.pack()
            b'\\x92\\x92\\x92\\x92\\xcb?\\xd3333333\\xcb?\\xb9\\x99\\x99\\x99\\x99\\x99\\x9a\\x92\\xcb?\\xd9\\x99\\x99\\x99\\x99\\x99\\x9a\\xcb?\\xd3333333\\x92\\x92\\xcb?\\xd9\\x99\\x99\\x99\\x99\\x99\\x9a\\xcb?\\xd3333333\\x92\\xcb?\\xe0\\x00\\x00\\x00\\x00\\x00\\x00\\xcb?\\xb9\\x99\\x99\\x99\\x99\\x99\\x9a\\x91\\x92\\x00\\x92\\x92\\x92\\xcb?\\xd3333333\\xcb?\\xb9\\x99\\x99\\x99\\x99\\x99\\x9a\\x92\\xcb?\\xd9\\x99\\x99\\x99\\x99\\x99\\x9a\\xcb?\\xd3333333\\x92\\x92\\xcb?\\xd9\\x99\\x99\\x99\\x99\\x99\\x9a\\xcb?\\xd3333333\\x92\\xcb?\\xe0\\x00\\x00\\x00\\x00\\x00\\x00\\xcb?\\xb9\\x99\\x99\\x99\\x99\\x99\\x9a'
        """
        # Pack it up, pack it in
        return msgpack.packb(self._list)

    def __repr__(self):
        return ''.join(('Marbl(', str(self.node_tpm), ', \n',
                        str(self.augmented_child_tpms), ')'))

    def __str__(self):
        return repr(self)


def unpack(packed_marbl):
    """Deserialize a Marbl.

    Example:
        >>> tpm = [[[0.3, 0.4],
        ...         [0.1, 0.3]],
        ...        [[0.4, 0.5],
        ...         [0.3, 0.1]]]
        >>> augmented_child_tpms = [[0, tpm], [1, tpm]]
        >>> marbl = Marbl(tpm, augmented_child_tpms)
        >>> marbl == unpack(pack(marbl))
        True
    """
    unpacked = msgpack.unpackb(packed_marbl)
    # Don't normalize, since if it was packed it must already be normalized
    return Marbl(unpacked[0], unpacked[1], already_normalized=True)


class MarblSet(collections.abc.Set):

    """
    An immutable, normalized, unordered collection of **not necessarily
    unique** Markov blankets.

    Provides methods for serialization and hashing.
    """

    def __init__(self, marbls, already_normalized=False):
        """
        Args:
            marbls (Iterable): The Marbls to include in the set.

        Keyword Args:
            already_normalized (bool): Flag to indicate TPMs have already been
                normalized. Defaults to ``False``.

        Warning:
            Incorrect use of the ``already_normalized`` flag can cause hashes
            to differ when they shouldn't. Make sure everything really is
            already normalized.
        """
        # The underlying representation is a list of Marbl TPMS ordered
        # lexicographically, per the Marbl spec.
        self.marbls = list(marbls)
        self._list = [b._list for b in sorted(self.marbls)]

    def __contains__(self, x):
        return x in self.marbls

    def __iter__(self):
        return iter(self.marbls)

    def __len__(self):
        return len(self.marbls)

    def __eq__(self, other):
        return self.marbls == other.marbls

    def __hash__(self):
        """Return the canonical hash of the multiset of Marbls.

        Example:
            >>> tpm = [[[0.3, 0.4],
            ...         [0.1, 0.3]],
            ...        [[0.4, 0.5],
            ...         [0.3, 0.1]]]
            >>> augmented_child_tpms = [[0, tpm], [1, tpm]]
            >>> marbl = Marbl(tpm, augmented_child_tpms)
            >>> marbls = MarblSet([marbl]*3)
            >>> hash(marbls)
            170586149808347817
        """
        return int(hashlib.sha1(self.pack()).hexdigest(), 16)

    def pack(self):
        """Serialize the multiset of Marbls.

        Example:
            >>> tpm = [[0.3, 0.4],
            ...        [0.1, 0.3]]
            >>> augmented_child_tpms = [[0, tpm]]
            >>> marbl = Marbl(tpm, augmented_child_tpms)
            >>> marbls = MarblSet([marbl]*2)
            >>> marbls.pack()
            b'\\x92\\x92\\x92\\x92\\xcb?\\xd3333333\\xcb?\\xb9\\x99\\x99\\x99\\x99\\x99\\x9a\\x92\\xcb?\\xd9\\x99\\x99\\x99\\x99\\x99\\x9a\\xcb?\\xd3333333\\x91\\x92\\x01\\x92\\x92\\xcb?\\xd3333333\\xcb?\\xb9\\x99\\x99\\x99\\x99\\x99\\x9a\\x92\\xcb?\\xd9\\x99\\x99\\x99\\x99\\x99\\x9a\\xcb?\\xd3333333\\x92\\x92\\x92\\xcb?\\xd3333333\\xcb?\\xb9\\x99\\x99\\x99\\x99\\x99\\x9a\\x92\\xcb?\\xd9\\x99\\x99\\x99\\x99\\x99\\x9a\\xcb?\\xd3333333\\x91\\x92\\x01\\x92\\x92\\xcb?\\xd3333333\\xcb?\\xb9\\x99\\x99\\x99\\x99\\x99\\x9a\\x92\\xcb?\\xd9\\x99\\x99\\x99\\x99\\x99\\x9a\\xcb?\\xd3333333'
        """
        return msgpack.packb(self._list)

    def __repr__(self):
        return 'MarblSet([' + ', '.join(str(m) for m in self.marbls) + '])'

    def __str__(self):
        return repr(self)


def unpack_set(packed_marbls):
    """Deserialize a multiset of Marbls.

    Example:
        >>> tpm = [[[0.3, 0.4],
        ...         [0.1, 0.3]],
        ...        [[0.4, 0.5],
        ...         [0.3, 0.1]]]
        >>> augmented_child_tpms = [[0, tpm], [1, tpm]]
        >>> marbl = Marbl(tpm, augmented_child_tpms)
        >>> marbls = MarblSet([marbl]*3)
        >>> marbls == unpack_set(pack(marbls))
        True
    """
    return MarblSet([Marbl(m[0], m[1], already_normalized=True) for m in
                     msgpack.unpackb(packed_marbls)], already_normalized=True)


def pack(obj):
    """Alias for :func:`Marbl.pack()` and :func:`MarblSet.pack()`."""
    return obj.pack()


def normalize_tpm(tpm, track_parent_index=None):
    """Return the normal form of a TPM. Optionally, also return the new
    dimension index of a particular parent in the normalized TPM.

    The TPM should be ``p``-dimensional, where ``p`` is the number of parents.
    For example, with three parents, ``TPM[0][1][0]`` should give the
    transition probability if the state of the parents is ``(0,1,0)``.

    Args:
        tpm (list): The child TPM to be normalized.

    Keyword Args:
        track_parent_index (int): The zero-based index of the dimension
            corresponding to the covered node in the un-normalized child TPM.
            If this is not ``None``, an normalized augmented child TPM will be
            returned instead of just a normalized TPM.

    Examples:
        >>> tpm = [[[0.3, 0.4],
        ...         [0.1, 0.3]],
        ...        [[0.4, 0.5],
        ...         [0.3, 0.1]]]
        >>> equivalent = [[[0.3, 0.1],
        ...                [0.4, 0.3]],
        ...               [[0.4, 0.3],
        ...                [0.5, 0.1]]]
        >>> normalize_tpm(tpm) == normalize_tpm(equivalent)
        True
        >>> answer = [2, [[[0.3, 0.1],
        ...                [0.4, 0.3]],
        ...               [[0.4, 0.3],
        ...                [0.5, 0.1]]]]
        >>> normalize_tpm(tpm, track_parent_index=1) == answer
        True
    """
    # Convert to a NumPy float array
    tpm = np.array(tpm).astype(float)
    # Get the all permuations of the parents (the number of parents is given by
    # the number of dimensions of the TPM).
    p_permutations = tuple(permutations(range(tpm.ndim)))
    # Get a list containing permuted TPMs, back in list form.
    tpm_permutations = [np.transpose(tpm, p).tolist() for p in p_permutations]
    # Lexicographic sort of the TPM permutations.
    sorted_permutations = sorted(tpm_permutations)
    # Immediately return the lexicographically least TPM permutation if we're
    # not keeping track of a parent node, otherwise find the parent node's new
    # index and return it with the minimal permutation.
    normal_tpm = sorted_permutations[0]
    if track_parent_index is None:
        return normal_tpm
    else:
        # Find the indices of the permuations that could yeild the normal form.
        valid_permutations = [perm for i, perm in enumerate(p_permutations) if
                              normal_tpm == tpm_permutations[i]]
        # The canonical new parent index is the image under the
        # lexicographically least the valid permutations
        minimal_valid_permutation = sorted(valid_permutations)[0]
        new_parent_index = minimal_valid_permutation[track_parent_index]
        # Return the new parent index with the normalized TPM
        return [new_parent_index, normal_tpm]


__title__ = 'marbl'
__version__ = '1.0.0'
__description__ = ('An implementation of the Marbl specification for '
                   'normalized representations of Markov blankets in Bayesian '
                   'networks.')
__author__ = 'Will Mayner'
__author_email__ = 'wmayner@gmail.com'
__author_website__ = 'http://willmayner.com'
__copyright__ = 'Copyright 2014 Will Mayner'
