from math import atan2

import matplotlib.pyplot as plt
from vectorfield import VectorField, circular, bam
from vessel import Vessel


class VesselChild(Vessel):
    def head_to(self, x: float, y: float, dt: float = 0.05):
        """This method moves the vessel to a given position."""
        dx = x - self.x
        dy = y - self.y
        self.theta = atan2(dy, dx)
        self.move(dt)


def test_head_to(vessel: Vessel, x: float = -2, y: float = -2, dt: float = 0.05):
    vessel.vectorfield.plot([-5, 5, -5, 5])
    plt.scatter(x, y, color="blue")
    # Start an animation to show the movement of the vessel
    # User controls when it stops
    while True:
        vessel.head_to(x, y, dt)
        vessel.plot()
        plt.pause(dt)
        if not plt.get_fignums():
            break


def main():
    vectorfield = VectorField(circular)

    # Vessel adapting its heading
    vessel_thrust = VesselChild(vectorfield, x=2, y=2, thrust=2)
    test_head_to(vessel_thrust)


if __name__ == "__main__":
    main()