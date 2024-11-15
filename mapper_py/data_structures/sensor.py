"""Ray and Sensor classes for 16-362: Mobile Robot Algorithms Laboratory

Author(s): Kshitij Goel, Wennie Tabib
"""

import numpy as np

from .grid import Point


class Ray:
    """A ray in 2D space.

    Attributes:
        o: (Point) Origin of the ray
        d: (Point) Direction vector for the ray
    """

    def __init__(self, origin=Point(0.0, 0.0), direction=Point(0.0, 0.0)):
        """Initially the origin and direction are zero."""
        self.o = origin
        self.d = direction

    def point_at_dist(self, t):
        """Get the point at the input distance along this ray.

        Args:
            t: (float) Distance along the ray.

        Returns:
            p: (Point) Point along this ray at distance `t`.
        """
        # TODO: Assignment 1, Problem 1.2

        pt_x = self.o.x + t * self.d.x
        pt_y = self.o.y + t * self.d.y

        return Point(pt_x, pt_y)

    def to_numpy(self):
        return np.hstack((self.o.to_numpy(), self.d.to_numpy()))


class Sensor:
    """A range sensor in 2D space.

    Attributes:
        num_rays: (int) Total number of rays.
        max_range: (float) Maximum reliable range for the sensor.
    """

    def __init__(self, max_range=1.0, num_rays=20):
        """Default max range is 1.0 meters and the number of rays are 20."""
        self.num_rays = num_rays
        self.max_range = max_range

    def rays(self, pos):
        """Generate rays at the input position.

        Rays around the given position `pos` at equal angular intervals within
        `[0, 2.0 * np.pi]` (i.e., 0 to 360 degrees).  Do not double count `0.0`
        and `2.0 * np.pi`

        Args:
            pos: (Point) Position of the sensor in 2D point space.

        Returns:
            rays: (list of Ray) List of `num_ray` amount of ray objects.
        """
        # TODO: Assignment 2, Problem 1.2
        # Hint 1: Utilize the `endpoint` option in the `np.linspace` function.
        # Hint 2: Should not require using `self.max_range`.

        rays_list = []
        for curr_ray_direction in np.linspace(0.0, 2.0 * np.pi, self.num_rays, False):
            direction_vector = Point(np.cos(curr_ray_direction), np.sin(curr_ray_direction))
            rays_list.append(Ray(pos, direction_vector))

        return rays_list
    

        # direction_gap = (2.0 * np.pi) / self.num_rays
        # #will not double count 0.0
        # for curr_ray_num in range(self.num_rays):
        #     rays_list.append(Ray(pos, curr_ray_num * direction_gap))

        # return rays_list
