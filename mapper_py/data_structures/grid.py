"""Cell, Point, and Grid classes for 16-362: Mobile Robot Algorithms Laboratory
"""

import numpy as np
from copy import copy


class Cell:
    """A single cell in the occupancy grid map.

    Attributes:
        row: Row number of the cell. Corresponds to Y-axis in 2D plane.
        col: Col number of the cell. Corresponds to X-axis in 2D plane.
    """

    def __init__(self, row=0, col=0):
        """Initializes the row and col for this cell to be 0."""
        self.row = row
        self.col = col

    def __str__(self):
        return f'Cell(row: {self.row}, col: {self.col})'

    def to_numpy(self):
        """Return a numpy array with the cell row and col."""
        return np.array([self.row, self.col], dtype=int)


class Point:
    """A point in the 2D space.

    Attributes:
        x: A floating point value for the x coordinate of the 2D point.
        y: A floating point value for the y coordinate of the 2D point.
    """

    def __init__(self, x=0.0, y=0.0):
        """Initializes the x and y for this point to be 0.0"""
        self.x = x
        self.y = y

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __str__(self):
        return f'Point(x: {self.x}, y: {self.y})'

    def __eq__(self, second):
        if isinstance(second, Point):
            return ((self.x == second.x) and (self.y == second.y))
        else:
            raise TypeError('Argument type must be Point.')

    def __ne__(self, second):
        if isinstance(second, Point):
            return ((self.x != second.x) or (self.y != second.y))
        else:
            raise TypeError('Argument type must be Point.')

    def __add__(self, second):
        if isinstance(second, Point):
            return Point(self.x + second.x, self.y + second.y)
        elif isinstance(second, float):
            return Point(self.x + second, self.y + second)
        else:
            raise TypeError('Argument type must be either float or Point.')

    def __sub__(self, second):
        if isinstance(second, Point):
            return Point(self.x - second.x, self.y - second.y)
        elif isinstance(second, float):
            # when subtracting with a float, always post-subtract
            # YES: Point(1.2, 3.2) - 5.0
            # NO: 5.0 - Point(1.2, 3.2)
            return Point(self.x - second, self.y - second)
        else:
            raise TypeError('Argument type must be either float or Point.')

    def __mul__(self, second):
        if isinstance(second, Point):
            return (self.x * second.x + self.y * second.y)
        elif isinstance(second, float):
            # when multiplying with a float, always post-multiply
            # YES: Point(1.2, 3.2) * 5.0
            # NO: 5.0 * Point(1.2, 3.2)
            return Point(self.x * second, self.y * second)
        else:
            raise TypeError('Argument type must be either float or Point.')

    def __truediv__(self, second):
        if isinstance(second, float):
            # when dividing by a float, always post-divide
            # YES: Point(1.2, 3.2) / 5.0
            # NO: 5.0 / Point(1.2, 3.2)
            if np.abs(second - 0.0) < 1e-12:
                raise ValueError(
                    'Divide by zero error. Second argument is too close to zero.')
            else:
                return Point(self.x / second, self.y / second)
        else:
            raise TypeError('Argument type must be float.')

    def to_numpy(self):
        """Return a numpy array with the x and y coordinates."""
        return np.array([self.x, self.y], dtype=float)


class Grid2D:
    """Occupancy grid data structure.

    Attributes:
        resolution: (float) The size of each cell in meters.
        width: (int) Maximum number of columns in the grid.
        height: (int) Maximum number of rows in the grid.
        min_clamp: (float) Logodds corresponding to minimum possible probability
        (to ensure numerical stability).
        max_clamp: (float) Logodds corresponding to maximum possible probability
        (to ensure numerical stability).
        free_thres: (float) Logodds below which a cell is considered free
        occ_thres: (float) Logodds above which a cell is considered occupied
        N: (int) Total number of cells in the grid
        data: Linear array of containing the logodds of this occupancy grid
    """

    def __init__(self, res, W, H, min_clamp, max_clamp, free_thres=0.13, occ_thres=0.7):
        """Initialize the grid data structure.

        Note that min_clamp, max_clamp, free_thres, and occ_thres inputs to this constructor
        are probabilities. You have to convert them to logodds internally for numerical stability.
        """
        self.resolution = res

        self.width = int(W)
        self.height = int(H)

        self.min_clamp = self.logodds(min_clamp)
        self.max_clamp = self.logodds(max_clamp)
        self.free_thres = self.logodds(free_thres)
        self.occ_thres = self.logodds(occ_thres)

        self.N = self.width * self.height

        # Initially all the logodds values are zero.
        # A logodds value of zero corresponds to an occupancy probability of 0.5.
        self.data = [0.0] * self.N

    def to_numpy(self):
        """Export the grid in the form of a 2D numpy matrix.

        Each entry in this matrix is the probability of occupancy for the cell.
        """
        g = np.zeros((self.height, self.width))
        for row in range(self.height):
            for col in range(self.width):
                v = self.get_row_col(row, col)
                g[row][col] = self.probability(v)

        return g

    def to_index(self, cell: Cell):
        """Return the index into the data array (self.data) for the input cell.

        Args:
            cell: (Cell) The input cell for which the index in data array is requested.

        Returns:
            idx: (int) Index in the data array for the cell
        """
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)

        #will go row-major order
        return cell.row * self.width + cell.col

    def from_index(self, idx):
        """Return the cell in grid for the input index.

        Args:
            idx: (int) Index in the data array for which the cell is requested.

        Returns:
            cell: (Cell) Cell corresponding to the index.
        """
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)

        return Cell(int(idx // self.width), int(idx % self.width))

    def get(self, idx):
        """Return the cell value corresponding to the input index.

        Args:
            idx: (int) Index in the data array for which the data is requested.

        Returns:
            val: (float) Value in the data array for idx
        """
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)

        return self.data[idx]

    def get_cell(self, cell):
        """Return the cell value corresponding to the input index."""
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        # Hint: Use the `to_index` and `get` methods.

        return self.get(self.to_index(cell))

    def get_row_col(self, row, col):
        """Return the cell value corresponding to the row and col."""
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        # Hint: Use the `get_cell` method and the `Cell` constructor.

        return self.get_cell(Cell(int(row), int(col)))

    def set(self, idx, value):
        """Set the cell to value corresponding to the input idx."""
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)

        self.data[idx] = value

    def set_cell(self, cell, value):
        """Set the cell to value corresponding to the input cell."""
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        # Hint: Use `to_index` and `set` methods.

        self.set(self.to_index(cell), value)

    def set_row_col(self, row, col, value):
        """Set the cell to value corresponding to the input row and col."""
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        # Hint: Use the `set_cell` method and the `Cell` constructor.

        self.set_cell(Cell(int(row), int(col)), value)

    def probability(self, logodds):
        """Convert input logodds to probability.

        Args:
            logodds: (float) Logodds representation of occupancy.

        Returns:
            prob: (float) Probability representation of occupancy.
        """
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)

        #logodds = log(p / (1-p))

        l = (np.e ** logodds)
        return l / (1 + l)

    def logodds(self, probability):
        """Convert input probability to logodds.

        Args:
            logodds: (float) Logodds representation of occupancy.

        Returns:
            prob: (float) Probability representation of occupancy.
        """

        # TODO: Assignment 2, Problem 1.1 (test_data_structure)

        #logodds = log(p / (1-p))
        not_p = 1 - probability
        res = np.log(probability / not_p)

        # res = min(self.max_clamp, res)
        # res = max(self.min_clamp, res)

        return res

    def cell_to_point(self, cell: Cell):
        """Get the cell's lower-left corner in 2D point space.

        Args:
            cell: (Cell) Input cell.

        Returns:
            point: (Point) Lower-left corner in 2D space.
        """
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)

        x = cell.col * self.resolution #left
        y = cell.row * self.resolution #lower
        return Point(x, y)

    def cell_to_point_row_col(self, row, col):
        """Get the point for the lower-left corner of the cell represented by input row and col."""
        # TODO: Assignment 2, Problem 1.1 (test_data_structure)
        # Hint: Use the `cell_to_point` function and the `Cell` constructor.

        return self.cell_to_point(Cell(int(row), int(col)))

    def point_to_cell(self, point: Point):
        """Get the cell position (i.e., bottom left hand corner) given the point.

        Args:
            point: (Point) Query point

        Returns:
            cell: (Cell) Cell in the grid corresponding to the query point.
        """
        # TODO: Assignment 2, Problem 1.1 (test_traversal)

        row = np.floor(point.y / self.resolution)
        col = np.floor(point.x / self.resolution)
        return Cell(int(row), int(col))

    def inQ(self, cell: Cell):
        """Is the cell inside this grid? Return True if yes, False otherwise."""
        # TODO: Assignment 2, Problem 1.1 (test_traversal)

        return (0 <= cell.row and cell.row < self.height and
               0 <= cell.col and cell.col < self.width)

    def traverse(self, start: Point, end: Point):
        """Figure out the cells that the ray from start to end traverses.

        Corner cases that must be accounted for:
        - If start and end points coincide, return (True, [start cell]).
        - Check that the start point is inside the grid. Return (False, None) otherwise.
        - End point can be outside the grid. The ray tracing must stop at the edges of the grid.
        - Perfectly horizontal and vertical rays.
        - Ray starts and ends in the same cell.

        Args:
            start: (Point) Start point of the ray
            end: (Point) End point of the ray

        Returns:
            success, raycells: (bool, list of Cell) If the traversal was successful, success is True
                                and raycells is the list of traversed cells (including the starting
                                cell). Otherwise, success is False and raycells is None.
        """
        # TODO: Assignment 2, Problem 1.1 (test_traversal)

        # the idea behind ray casting is we are trying to determine which time step we are going to cross horizontally or vertically
        # We cross the most recently cross then find the next time step
        # Ie. we split up horizontal and vertical crossings and calc each seperately but we know which one we cross first

        start_cell = self.point_to_cell(start)
        end_cell = self.point_to_cell(end)

        cb = self.cell_to_point(start_cell) #lower-left corner of start

        #corner cases
        if (not self.inQ(start_cell)):
            return False, None
        
        if (start == end):
            return True, [copy(start_cell)]

        #init
        #direction and velocity of the ray
        #NOTE: mistake is I changed dir_magnitude by doing dir_x first then using the result for dir_y magnitude calculation
        dir_x = end.x - start.x
        dir_y = end.y - start.y
        dir_magnitude = np.sqrt(dir_x**2 + dir_y**2)
        dir_x /= dir_magnitude
        dir_y /= dir_magnitude

        #note: cannot be 0
        step_col = 0 if dir_x == 0 else int(dir_x // abs(dir_x))
        step_row = 0 if dir_y == 0 else int(dir_y // abs(dir_y))

        #time it takes to go across a cell in x and y directions
        tDeltaX = float('inf') if dir_x == 0 else self.resolution * float(step_col) * (1.0 / dir_x)
        tDeltaY = float('inf') if dir_y == 0 else self.resolution * float(step_row) * (1.0 / dir_y)

        #time inits (time to first cross x or y boundaries)
        tMaxX = (cb.x + self.resolution - start.x) * (1.0 / dir_x) if dir_x > 0 else (cb.x - start.x) * (1.0 / dir_x) if dir_x < 0 else float('inf')
        tMaxY = (cb.y + self.resolution - start.y) * (1.0 / dir_y) if dir_y > 0 else (cb.y - start.y) * (1.0 / dir_y) if dir_y < 0 else float('inf')

        #update
        curr_cell = copy(start_cell)
        raycells = []

        while (not (curr_cell.row == end_cell.row and curr_cell.col == end_cell.col)):

            # print(curr_cell)
            if not self.inQ(curr_cell):
                return True, raycells

            raycells.append(copy(curr_cell))

            #update curr cell
            if tMaxX < tMaxY: #go horizontal first
                curr_cell.col += step_col
                tMaxX += tDeltaX
            else: #go vertical first
                curr_cell.row += step_row
                tMaxY += tDeltaY

        if self.inQ(curr_cell):
            #for the case where cell is directly at the edge, out of bounds
            raycells.append(copy(curr_cell)) #add last, equal cell

        return True, raycells


    def freeQ(self, cell: Cell):
        """Is the cell free? Return True if yes, False otherwise."""
        # TODO: Assignment 2, Problem 1.3
        # Hint: Use `get_cell` and `free_thres`

        if self.get_cell(cell) < self.free_thres:
            return True
        
        return False

    def occupiedQ(self, cell):
        """Is the cell occupied? Return True if yes, False otherwise."""
        # TODO: Assignment 2, Problem 1.3
        # Hint: Use `get_cell` and `occ_thres`

        if self.get_cell(cell) > self.occ_thres:
            return True
        
        return False

    def unknownQ(self, cell):
        """Is the cell unknown? Return True if yes, False otherwise."""
        # TODO: Assignment 2, Problem 1.3
        # Hint: Use `get_cell`, `occ_thres`, and `free_thres`
        
        if self.free_thres <= self.get_cell(cell) and self.get_cell(cell) <= self.occ_thres:
            return True
        
        return False
