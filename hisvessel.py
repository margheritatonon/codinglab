from abc import abstractmethod
from math import cos, sin

import matplotlib.pyplot as plt
from vectorfield import VectorField, circular


class Vessel:
    def __init__(
        self,
        vectorfield: VectorField,
        x: float = 0,
        y: float = 0,
        thrust: float = 0,
        theta: float = 0,
        color: str = "red",
    ):
        self.vectorfield = vectorfield
        self.x = x
        self.y = y
        self.thrust = thrust
        self.theta = theta
        self.color = color
        # Hidden attributes to store the plot of the vessel
        self._plot_point = None

    @property
    def thrust_x(self) -> float:
        """
        This property returns the x-component of the thrust vector.
        A property does not need to be called as a method, but as an attribute.
        """
        return self.thrust * cos(self.theta)

    @property
    def thrust_y(self) -> float:
        """
        This property returns the y-component of the thrust vector.
        A property does not need to be called as a method, but as an attribute.
        """
        return self.thrust * sin(self.theta)

    def move(self, dt: float = 0.05):
        """This method moves the vessel in the direction of its heading."""
        u, v = self.vectorfield(self.x, self.y)
        dx = u + self.thrust_x
        dy = v + self.thrust_y
        self.x += dx * dt
        self.y += dy * dt

    @abstractmethod
    def head_to(self, x: float, y: float, dt: float = 0.05):
        """This method moves the vessel to a given position."""
        raise NotImplementedError

    def change_heading(self, dt: float = 0.05):
        """This method computes the drift of the vessel over a given time interval."""
        # Get the velocity components (u, v) at the current position (x, y)
        u, v = self.vectorfield(self.x, self.y)

        # Current heading
        theta = self.theta

        # Compute the total velocity from thrust and vector field
        vx = u + self.thrust * cos(theta)
        vy = v + self.thrust * sin(theta)

        # The rate of change of heading due to vector field's influence
        dtheta_dt = (v * cos(theta) - u * sin(theta)) / (self.thrust + 1e-10)

        # Update heading using finite difference (Euler's method)
        self.theta += dtheta_dt * dt

        # Update position based on the total velocity (vx, vy)
        self.x += vx * dt
        self.y += vy * dt

    def plot(self):
        # Remove the previous plot if it exists
        if self._plot_point:
            self._plot_point.remove()
            self._plot_arrow.remove()
        # Plot the vessel at its current heading
        self._plot_point = plt.scatter(self.x, self.y, color=self.color)
        self._plot_arrow = plt.arrow(
            self.x,
            self.y,
            self.thrust_x / 2,
            self.thrust_y / 2,
            color=self.color,
            head_width=self.thrust / 10,
        )


def test_move(vessel: Vessel, dt: float = 0.05):
    vessel.vectorfield.plot([-5, 5, -5, 5])
    # Start an animation to show the drift of the vessel
    # User controls when it stops
    while True:
        vessel.move(dt)
        vessel.plot()
        plt.pause(dt)
        if not plt.get_fignums():
            break


def test_move_multiple(ls_vessels: list[Vessel], dt: float = 0.05):
    ls_vessels[0].vectorfield.plot([-5, 5, -5, 5])
    # Start an animation to show the movement of the vessel
    # User controls when it stops
    while True:
        for vessel in ls_vessels:
            vessel.move(dt)
            vessel.plot()
        plt.pause(dt)
        if not plt.get_fignums():
            break


def test_change_heading(vessel: Vessel, dt: float = 0.05):
    vessel.vectorfield.plot([-5, 5, -5, 5])
    # Start an animation to show the movement of the vessel
    # User controls when it stops
    while True:
        vessel.change_heading(dt)
        vessel.plot()
        plt.pause(dt)
        if not plt.get_fignums():
            break


def main():
    vectorfield = VectorField(circular)

    # Vessel with no thrust
    vessel_nothrust = Vessel(vectorfield, x=2, y=2)
    test_move(vessel_nothrust)

    # Several vessels
    ls_vessels = []
    for x in range(4):
        for y in range(4):
            ls_vessels.append(Vessel(vectorfield, x=x, y=y))
    test_move_multiple(ls_vessels)

    # Vessel with thrust
    vessel_thrust = Vessel(vectorfield, x=2, y=2, thrust=2)
    test_move(vessel_thrust)

    # Vessel changing its heading
    vessel_thrust = Vessel(vectorfield, x=2, y=2, thrust=2)
    test_change_heading(vessel_thrust)


if __name__ == "__main__":
    main()