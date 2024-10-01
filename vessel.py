from vectorfield import VectorField
from vectorfield import bam
from vectorfield import circular
import matplotlib.pyplot as plt
import numpy as np
from math import atan2
from math import cos
from math import sin
from vectorfield import vf1, vf2, vf3, vf4, vf5, vf6, vf7

class Vessel:
    def __init__(self, vectorfield, x=0, y=0, thrust = 1, theta=0, color="red"):
        #theta is the heading of the vessel
        self.vectorfield = vectorfield
        self.x= x
        self.y = y
        self.thrust = thrust
        self.theta = theta
        self.color = color

    def plot(self):
        plt.plot(self.x, self.y, "ro", color=self.color)
        #plt.show()
        plt.quiver(self.x, self.y, np.cos(self.theta), np.sin(self.theta), color = self.color, scale=5, scale_units="inches")

    def move(self, dt=0.05): #the time step we are going to move our vessel
        u, v = self.vectorfield(self.x, self.y)
        dx = u + self.thrust * np.cos(self.theta)
        dy = v + self.thrust * np.sin(self.theta)
        self.x += dx * dt
        self.y += dy * dt

    def head_to(self, x, y, dt=0.05): #point vessel wants to arrive to
        dx = x - self.x
        dy = y - self.y
        self.theta = atan2(dy, dx) #updates theta angle of the vessel
        self.move(dt)

class VesselRandom(Vessel): #defining the inheritance
    def __init__(self, vectorfield, x=0, y=0, thrust = 1, theta=0, color="red"):
        super().__init__(vectorfield, x, y, thrust, theta, color)

    def head_to_random(self, x, y, dt=0.05):
        self.theta = (np.random.randint(1, 360))  # Convert to radians
        u, v = self.vectorfield(self.x, self.y)
        dx = u + self.thrust * np.cos(self.theta)
        dy = v + self.thrust * np.sin(self.theta)
        self.x += dx * dt
        self.y += dy * dt

#I fed the previous code to AI as well as the task instructions and this is the code it generated:
class VesselAggressive(Vessel):
    def __init__(self, vectorfield, x=0, y=0, thrust=1, theta=0, color="red"):
        super().__init__(vectorfield, x, y, thrust, theta, color)

    def head_to_aggressive(self, x, y, dt=0.05):
        dx = x - self.x #updating the same as the head_to method, distance b/w current position and target x and y
        dy = y - self.y
        distance = np.sqrt(dx**2 + dy**2) #calculates how far away the vessel is from the target positions
        if distance > 1:
            self.theta = atan2(dy, dx) #when the vessel is far enough away, the theta is updated in the same was as in the head_to method.
        else: #when the distance is less than 1, so relatively close to the target point
            self.theta = np.arctan2(dy, dx) + np.pi / 2 #the theta will be perpendicular to the target direction. takes a detour, avoids the vessel to get closer and closer to the target but never actually reaching it, takes a detour 
        self.move(dt)

#we need to define a class vectorfield that can help us compute a numerical approximation of the gradient. we hve already defined this in the vectorfield.py script,so we will add a method to that class.
#we want to use the gradient of a vector field. the gradient tells us the direction of steepest increase. so we can follow that path to hopefully reach the goal point faster.
#we want to compute the direction that will bring us to (xgoal,ygoal) the fastest. this is the gradient.

class VesselGradient(Vessel):
    def __init__(self, vectorfield, x=0, y=0, thrust=1, theta=0, color="red"):
        super().__init__(vectorfield, x, y, thrust, theta, color)

    def head_to_gradient(self, x, y, dt=0.05):
        du_dx, dv_dx, du_dy, dv_dy = self.vectorfield.gradient(self.x, self.y)  # Compute gradient at vessel's current position
        dx_goal = x - self.x
        dy_goal = y - self.y
        theta_goal = np.arctan2(dy_goal, dx_goal)  # Compute direction from current position to goal point
        theta_gradient = np.arctan2(du_dy, du_dx)  # Compute direction of gradient at vessel's current position
        self.theta = theta_goal + 0.5*theta_gradient  # Update theta to point in direction of gradient relative to goal point, with some weighting towards the goal direction
        self.move(dt)
    

#the concept for this was trying to move in the direction of the negative gradient.
class VesselNegativeGradient(Vessel):
    def __init__(self, vectorfield, x=0, y=0, thrust=1, theta=0, color="red"):
        super().__init__(vectorfield, x, y, thrust, theta, color)

    def head_to_neg_gradient(self, x, y, dt=0.05):
        du_dx, dv_dx, du_dy, dv_dy = self.vectorfield.gradient(x, y)
        u, v = self.vectorfield(self.x, self.y)
        theta_field = np.arctan2(v, u)
        dx_goal = x - self.x
        dy_goal = y - self.y
        theta_goal = np.arctan2(dy_goal, dx_goal)
        theta_gradient = np.arctan2(-du_dy, -du_dx)
        self.theta = theta_field + np.pi + theta_gradient + theta_goal
        self.move(dt)
    

class VesselPerpendicular(Vessel):
    def __init__(self, vectorfield, x=0, y=0, thrust=1, theta=0, color="red"):
        super().__init__(vectorfield, x, y, thrust, theta, color)

    def head_to_perpendicular(self, x, y, dt=0.05):
        dx = x - self.x
        dy = y - self.y
        theta_goal = np.arctan2(dy, dx)
        u, v = self.vectorfield(self.x, self.y)
        theta_field = np.arctan2(v, u)
        self.theta = theta_field + np.pi / 2  # Move perpendicular to the vector field
        self.move(dt)

#however, all of the above classes have vessels that move exactly in the direction of the gradient.
#we want to move against the gradient!

#we want one that goes out into the wider part immediately, and then goes inside slower, bc otherwise we risk it converging.
#with vf2, vf3, vf4, vf5 all of the approaches diverge outside.
#maybe we can use the vector field properties to determine what to code: based on the curl and the divergence?

class VesselProperties(Vessel):
    def __init__(self, vectorfield, x=0, y=0, thrust=1, theta=0, color="red"):
        super().__init__(vectorfield, x, y, thrust, theta, color)

    #we can compare the direction we want to move to with the gradient that we have.
    def head_to_properties(self, x, y, dt=0.05):
        #the direction we want to move to:
        dx = x - self.x 
        dy = y - self.y
        theta_towards = np.arctan2(dx, dy)

        #the direction of the gradient:
        grad_x, grad_y = self.vectorfield(self.x, self.y)
        theta_gradient = np.arctan2(grad_x, grad_y)

        #we know that the gradient pushes the vessel a lot. but we do not want to move in the direction of the gradient.
        #we can look at the difference between the thetas - one is the one we want to move towards and the other one is the one we are "forced" to move towards.
        theta = theta_towards - theta_gradient

        #we do not want to move in the direction of the vector field. so we add the "theta" to the self.theta so that we minimize our movement with the gradient.
        if abs(theta) > np.pi/2: # If the angle difference is greater than pi/2, move in the opposite direction of the vector field - this is the tip that AI gave me when I fed them the above code in this method.
            self.theta = theta_gradient + np.pi
        else:
            self.theta = theta_towards
        self.move(dt)

class VesselOut(Vessel):
    def __init__(self, vectorfield, x=0, y=0, thrust=1, theta=0, color="red"):
        super().__init__(vectorfield, x, y, thrust, theta, color)

    def head_to_out(self, x, y, dt=0.05):
        dx = x - self.x 
        dy = y - self.y
        theta_towards = np.arctan2(dx, dy)
        grad_x, grad_y = self.vectorfield(self.x, self.y)
        theta_gradient = np.arctan2(grad_x, grad_y)
        self.theta = 0.5*theta_towards - 2*theta_gradient
        self.move(dt)


def main():
    vf = VectorField(vf6)

    vessel = Vessel(vf, x=1, y=1)
    vessel2 = VesselRandom(vf, x=1, y=1, color="blue")
    vessel3 = VesselAggressive(vf, x=1, y=1, color = "purple")
    vessel4 = VesselGradient(vf, x=1, y=1, color="orange")
    vessel5 = VesselNegativeGradient(vf, x=1, y=1, color="yellow")
    vessel6 = VesselPerpendicular(vf, x=1, y=1, color="pink")
    vessel7 = VesselProperties(vf, x=1, y=1, color="brown")
    vessel8 = VesselOut(vf, x=1, y=1, color="teal")

    xgoal, ygoal = -2, -2

    #animate the vessel movement until the user closes the plot window
    plt.ion()
    while True:
        vessel.head_to(xgoal, ygoal)
        vessel2.head_to_random(xgoal, ygoal)
        vessel3.head_to_aggressive(xgoal, ygoal)
        vessel4.head_to_gradient(xgoal, ygoal)
        vessel5.head_to_neg_gradient(xgoal, ygoal)
        vessel6.head_to_perpendicular(xgoal, ygoal)
        vessel7.head_to_properties(xgoal, ygoal)
        vessel8.head_to_out(xgoal, ygoal)
        plt.scatter(xgoal, ygoal, color = "green")
        vf.plot(limits=(-5, 5, -5, 5))
        vessel.plot() #ends up going to the middle, gets 'stuck' at (0,0) w circular
        vessel2.plot() #this one ends up converging to the middle w circular
        vessel3.plot() #also ends up converging w circular
        vessel4.plot()
        vessel5.plot() #this one diverges / moves away
        vessel6.plot() #in circular, moves to middle.
        vessel7.plot()
        vessel8.plot()

        #checking the distances to see if any vessels have reached the target point
        distances = [
            np.sqrt((vessel.x - xgoal)**2 + (vessel.y - ygoal)**2),
            #np.sqrt((vessel2.x - xgoal)**2 + (vessel2.y - ygoal)**2),
            np.sqrt((vessel3.x - xgoal)**2 + (vessel3.y - ygoal)**2)
        ]
        if any(distance < 0.001 for distance in distances):
            break


        plt.pause(0.05)
        plt.clf()

        #if window is closed, the execution will stop
        fig = plt.gcf()
        fig.canvas.mpl_connect('close_event', lambda event: exit())

    plt.show()

if __name__ == "__main__":
    main()