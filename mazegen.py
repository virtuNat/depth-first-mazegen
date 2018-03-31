#!/usr/bin/env python

from collections import deque
import numpy as np
from numpy.random import choice, randint
import matplotlib.pyplot as plt

def maze(cells=(25, 25), start=(0, 0), exit=(-1, -1)):
    """Generates a binary numpy array that represents a maze.
    Values of 1 are colored and are representing walls, while
    values of 0 are not colored and represent path.
    """
    if cells[0] <= 0 or cells[1] <= 0:
        # Just quit if invalid parameters are passed.
        raise ValueError('Invalid maze dimensions')

    # --- Lookup Table Data ---
    # Table of direction vectors mapped to wallflags.
    vdirtable = {
        1: np.array([0, 1]),
        2: np.array([1, 0]),
        4: np.array([0, -1]),
        8: np.array([-1, 0]),
        }
    # Table of opposing directions mapped to each other.
    odirtable = {1: 4, 2: 8, 4: 1, 8: 2}
    # Table of direction tuples mapped to wallflag sums.
    walltable = {i: tuple(j for j in (8, 4, 2, 1) if i & j) for i in range(16)}
    # --- Maze grid initialization ---
    # Build maze using an odd shape value to make walls their own cells.
    mgrid = np.ones((2*cells[0] + 1, 2*cells[1] + 1, 2), dtype=int)
    # Create the seed used to generate the maze from.
    mseed = np.fromiter((2*randint(0, size-1) + 1 for size in cells), int)
    # Initialize wallflag matrix.
    if cells == (1, 1):
        mgrid[1, 1, 1] = 0
    elif cells[0] == 1:
        mgrid[1, 1, 1] = 1
        mgrid[1, 3:-2:2, 1] = 5
        mgrid[1, -2, 1] = 4
    elif cells[1] == 1:
        mgrid[1, 1, 1] = 2
        mgrid[3:-2:2, 1, 1] = 10
        mgrid[-2, 1, 1] = 8
    else:
        # Corners
        mgrid[1, 1, 1] = 3
        mgrid[-2, 1, 1] = 9
        mgrid[-2, -2, 1] = 12
        mgrid[1, -2, 1] = 6
        # Edges
        mgrid[1, 3:-2:2, 1] = 7
        mgrid[3:-2:2, 1, 1] = 11
        mgrid[-2, 3:-2:2, 1] = 13
        mgrid[3:-2:2, -2, 1] = 14
        # Centre
        mgrid[3:-2:2, 3:-2:2, 1] = 15
    # The number of maze nodes specified.
    nsize = cells[0] * cells[1]
    
    def open_wall(npos):
        """Opens the wall of a maze to allow entry and exit."""
        gpos = np.fromiter(map(lambda x: 2*x + int(x >= 0), npos), int)
        try:
            wallflag = walltable[mgrid[gpos[0], gpos[1], 1]]
        except IndexError:
            raise IndexError('Entrypoint cell outside of maze')
        for d in (i for i in (4, 1, 8, 2) if i not in wallflag):
            wpos = gpos + vdirtable[d]
            if mgrid[wpos[0], wpos[1], 0]:
                mgrid[wpos[0], wpos[1], 0] = False
                break
        else:
            raise ValueError('Entrypoint cell not on maze border')
    
    # Open the designated entrance and exit.
    open_wall(start)
    open_wall(exit)
    del open_wall
    # Current path stack.
    mpath = deque([mseed])
    # Open the starting cell.
    mgrid[mseed[0], mseed[1], 0] = False
    # Debug counter.
    itercnt = 0
    # Counts the number of nodecells opened in the maze.
    # When the number of open cells equals the original passed mazesize,
    # The algorithm has filled up the maze and may stop.
    cellcnt = 1
    # If all of the original nodes have been visited, the maze is done.
    while cellcnt != nsize:
        itercnt += 1
        # Obtain current cell.
        thispos = mpath[-1]
        thiscel = mgrid[thispos[0], thispos[1]]
        # print(mgrid[:,:,0])
        while thiscel[1]:
            cdir = choice(walltable[thiscel[1]])
            vdir = vdirtable[cdir]
            # Remove chosen direction from both templist and matrix.
            thiscel[1] -= cdir
            # Get the position of the next cell.
            nextpos = thispos + 2 * vdir
            nextcel = mgrid[nextpos[0], nextpos[1]]
            odir = odirtable[cdir]
            if nextcel[1] & odir:
                nextcel[1] -= odir
            if nextcel[0]:
                cellcnt += 1
                # Get the position of the wall to open.
                wallpos = thispos + vdir
                # Add the next cell to the current path.
                if nextcel[1]:
                    mpath.append(nextpos)
                # Open the path to the next cell.
                mgrid[nextpos[0], nextpos[1], 0] = False
                mgrid[wallpos[0], wallpos[1], 0] = False
                break
        else:
            # There are no more unexplored directions, backtrack.
            mpath.pop()
    print("Performed", itercnt, "iterations for maze of size", nsize)
    # Return only the binary array of walls and paths.
    return mgrid[:,:,0]

if __name__ == '__main__':
    newmaze = maze((100, 100), (50, 0), (50, -1))
    plt.imsave('maze.png', newmaze, cmap=plt.cm.binary)
    plt.figure(figsize=(9, 8))
    plt.imshow(newmaze, cmap=plt.cm.binary, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.show()
