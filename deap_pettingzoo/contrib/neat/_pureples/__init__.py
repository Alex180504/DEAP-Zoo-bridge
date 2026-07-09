"""Vendored subset of `pureples <https://github.com/ukuleleplayer/pureples>`_
(MIT license).

Only the core algorithmic modules needed for HyperNEAT and ES-HyperNEAT
phenotype construction are included.  OpenAI Gym runners, visualisation
helpers, and experiment scripts are omitted — our integration layer
provides its own evaluation and logging.

Modules
-------
* ``substrate`` — :class:`Substrate` (input / output / hidden coordinate layout).
* ``hyperneat`` — :func:`create_phenotype_network` (CPPN -> substrate-query -> ANN).
* ``es_hyperneat`` — :class:`ESNetwork` (evolvable-substrate HyperNEAT via quadtree).
* ``create_cppn`` — :func:`create_cppn` (genome -> :class:`~neat.nn.FeedForwardNetwork`).

Patches applied for neat-python ≥ 2.0
--------------------------------------
* ``neat.config.Config`` -> ``neat.Config`` (top-level import).
* ``neat.population.Population`` -> ``neat.Population``.
* ``neat.genome.DefaultGenome`` / ``neat.reproduction.DefaultReproduction`` /
  ``neat.species.DefaultSpeciesSet`` / ``neat.stagnation.DefaultStagnation``
  — these are still importable from their sub-packages in neat-python 2.x;
  no changes required.

Original copyright notice
--------------------------
::

    MIT License

    Copyright (c) 2017-2018 Adrian Westh

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

__all__ = [
    "Substrate",
    "create_cppn",
    "create_phenotype_network",
    "query_cppn",
    "ESNetwork",
    "QuadPoint",
    "Connection",
    "find_pattern",
]

# All imports are lazy at the submodule level so users can import
# individual modules without pulling in unused dependencies.
