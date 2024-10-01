from math import atan2

import matplotlib.pyplot as plt
from vectorfield import VectorField, circular, bam
from vessel import Vessel
from vesselchild import VesselChild



class VesselClever(Vessel):

    def __init__(
        self,
        vectorfield: VectorField,
        x: float = 0,
        y: float = 0,
        thrust: float = 0,
        theta: float = 0,
        damping: float = 1,
        color: str = "red",
    ):
        super().__init__(vectorfield, x, y, thrust, theta, color)
        self.damping = damping

    def head_to(self, x: float, y: float, dt: float = 0.05):
        # We want to move in the following direction
        dx = x - self.x
        dy = y - self.y
        theta_goal = atan2(dy, dx)

        # However the vectorfield pushes us
        u, v = self.vectorfield(self.x, self.y)
        theta_field = atan2(v, u)
        diff_theta = theta_goal - theta_field

        # Aim our angle to compesate for the vectorfield
        self.theta += diff_theta / self.damping
        self.move(dt)


def compute_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def competition(
    ls_vessels: list[Vessel], x: float = -2, y: float = -2, dt: float = 0.05
):
    """This function simulates a competition between vessels to reach a given point.

    Parameters
    ----------
    ls_vessels : list[Vessel]
        A list of vessels that will compete in the same vector field
    x : float, optional
        Goal position in the x-direction, by default -2
    y : float, optional
        Goal position in the y-direction, by default -2
    dt : float, optional
        Time interval for the simulation, by default 0.05
    """
    # Ensure all vessels share the same parameters
    for vessel in ls_vessels:
        vessel.vectorfield = ls_vessels[0].vectorfield
        vessel.x = ls_vessels[0].x
        vessel.y = ls_vessels[0].y
        vessel.thrust = ls_vessels[0].thrust

    ls_dist = [compute_distance(vessel.x, vessel.y, x, y) for vessel in ls_vessels]

    ls_vessels[0].vectorfield.plot([-5, 5, -5, 5])
    plt.scatter(x, y, color="blue")
    # Start an animation to show the movement of the vessel
    # User controls when it stops
    while True:
        for vessel in ls_vessels:
            vessel.head_to(x, y, dt)
            vessel.plot()
        plt.pause(dt)
        if not plt.get_fignums():
            break
        ls_dist = [compute_distance(vessel.x, vessel.y, x, y) for vessel in ls_vessels]
        if any(dist < 0.1 for dist in ls_dist):
            winner: Vessel = ls_vessels[ls_dist.index(min(ls_dist))]
            # Display message on plot
            msg = f"Vessel {winner.color} wins!"
            plt.text(0, 0, msg, fontsize=12, color="black")
            plt.show()
            break


def main():
    vectorfield = VectorField(circular)
    x0, y0 = 0, 0
    thrust = 2
    x, y = -2, -2

    # Compete against the basic vessel
    ls_vessels = [
        VesselChild(vectorfield, x=x0, y=y0, thrust=thrust),
        VesselClever(vectorfield, x=x0, y=y0, thrust=thrust, color="blue"),
    ]
    competition(ls_vessels, x=x, y=y)

    # Compete against themselves
    ls_vessels = [
        VesselClever(
            vectorfield, x=x0, y=y0, thrust=thrust, damping=0.5, color="orange"
        ),
        VesselClever(vectorfield, x=x0, y=y0, thrust=thrust, damping=1, color="blue"),
        VesselClever(vectorfield, x=x0, y=y0, thrust=thrust, damping=2, color="cyan"),
    ]
    competition(ls_vessels, x=x, y=y)


if __name__ == "__main__":
    main()