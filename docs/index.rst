.. VirtLinAlg documentation master file, created by
   sphinx-quickstart on Sat Jun 28 20:47:55 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VirtLinAlg documentation
========================

A library for virtual linear algebra
------------------------------------


This library enables users to write code that uses matrix operations while
abstracting several unrelated implementation details of said operations.

1. The special mathematical structure of some matrices:

   * low-rank matrices and low-rank perturbations of matrices,
   * more trivially, diagonal matrices and scalar multiple of identity matrices.

Matrices having such a structure can exploit linear algebra tricks, such as
the `Woodbury matrix identity <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`__,
to perform some operations, mostly inverses and determinants. The library
implements these tricks for you.

3. Collection of related matrices, such as several matrices differing by a
low-rank perturbation, and block matrices where blocks are any of the above.
Here again the structure of the block matrix or stack of matrics allows
linear algebra tricks that the library offers. Such collections are often
obtained by performing "leave-one-out" operation, which the library offers.
Even with unrelated matrices, it offers a comfortable syntax to vectorize
matrix functions.

4. The specific library used to store the matrix and operates on them: NumPy,
Torch, Tensorflow, JAX. You can write functions once, and then use them
immediately with any of these frameworks. The library handles the
translation of operations from one framework to another. This should
hopefully become increasingly unimportant as standardization efforts
progress: for instance Tensorflow has a tf.experimental.numpy API, and the
Array API exists. This is not really the main focus of the library, rather
it is the solution chosen so that VLA is not locked in to a specific backend
or API.

Put together, the library allows you for instance to write a function using
linear algebra tricks to efficiently handle a stack of matrices differing from
one another by low-rank updates, and have that function work out of the box with
NumPy, or with Torch and JAX arrays to get automatic differentiation on it.

Getting started
---------------

VirtLinAlg is under active development and has not yet been released. To try out
the current development version, install it directly from the Github
repository::

        pip install "virtlinalg @ git+https://github.com/emorice/virtlinalg"

API documentation
-----------------

.. automodule:: virtlinalg
   :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

