# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 20:58:51 2019

@author: Jonathan Matthews (jxm1027)

Requirements:
    - matplotlib (and dependencies)
    - numpy (and dependencies)

Naming conventions:
    -Classes use UpperCamelCase
    -functions and variables use snake_case

Script can be run as main to produce all the data present in the associated report, or can be imported
to access functions directly or to create custom scenarios. In the latter case, refer to the docstrings
of each function.

"""

from math import acos as arccos, atan2 as arctan2, sqrt, sin, cos, pi, log
from copy import deepcopy
from matplotlib import pyplot as plt
from numpy import std, mean


#################
### Functions ###
#################

def sign(x):
    """Returns the sign (-1 or 1) of a number 'x'."""
    if x < 0:
        return -1
    else:
        return 1


def quad(a, b, c, addsub):
    """Solve a quadratic equation in the form ax^2 + bx + c = 0. Takes coefficients a, b and c as
       arguments, as well as 'addsub' to determine whether sqrt(b*b - 4ac) is added or subtracted. 'addsub'
       should be either +1 or -1."""
    
    if abs(addsub) != 1:
        raise ValueError("Argument 'addsub' should either be 1 or -1.")
       
    try:
        return (-b + addsub * sqrt(b*b - 4*a*c)) / (2*a)
    except ValueError: # Incase float imprecision causes complex root.
        return -b/(2*a)


def get_balls(table, initial, n=2):
    """For a given boundary 'table', and given ball 'initial' (required to be within table's boundaries), and
       given number of collisions 'n', will return a list of Balls pertaining to the reflected initial ball."""
    
    balls = [initial]
    
    for _ in range(n+1):
        balls.append(table.collide(balls[-1]))
    
    return balls

###############
### Classes ###
###############

class Ball:
    """Defines a billiard ball with initial position and velocity."""
    
    def __init__(self, pos, vel, end=None, rebound=None):
        """
        Constructs the billiard ball from initial position 'pos' and initial velocity 'vel', with
        optional argument to set ball and point (though this can be set later by accessing Ball.end).
        Will calculate gradient and y-axis intersection of the line formed by the motion of the ball,
        stored in member Ball.eqn .
        
        pos : [x, y]
        vel : [vx, vy]
        end : [x, y]
        """
            
        self.pos = pos # Initial ball position.
        self.vel = [round(v, 7) for v in vel]
        # Round velocity to 7 decimal places, as any more than this produce strange errors inside Ball.y() .
        
        if self.vel == [0, 0]:
            raise ValueError("Ball has invalid velocity (zero or too small).")
        
        try:
            gradient = self.vel[1]/self.vel[0]
            self.eqn = [gradient, pos[1] - gradient*pos[0]] # [m, c], as in: y = mx + c
            
        except ZeroDivisionError: # If ball moves vertically.
            self.eqn = [None, None]
        
        self.end = end # Ball end position (intersection with table boundary).
        self.rebound = rebound # Angle of rebound, to be set when reflection is calculated.
    
    def __repr__(self):
        """Defines the method to reproduce an instance of Ball."""
        return f"Ball({self.pos}, {self.vel}, {self.end}, {self.rebound})"
    
    def y(self, x):
        """Evaluates the y-coordinate of the ball at x. Returns None if motion is vertical."""
        if self.eqn[0] is not None:
            return self.eqn[0]*x + self.eqn[1]
    
    def x(self, y):
        """Evaluates the x-coordinate of the ball at y. Returns None if motion is horizontal."""
        if self.eqn[0]:
            return (y - self.eqn[1]) / self.eqn[0]
        if self.eqn[0] is None:
            return self.pos[0]
    
    def reflect(self, gradient):
        """Return a reflected ball for a given gradient upon which the ball is incident. Only valid for balls
           whose endpoint (self.end) has been set, in the form [x, y]. 'gradient' should be set to None for an
           infinite gradient (that is, a vertical line).
           
           Assumes ball is moving."""
           
        if self.end is None:
            raise Exception("Ball endpoint has not been set. Set with Ball.end.")

        if gradient is None:
            incident = arccos(self.vel[1]/sqrt(self.vel[0]**2 + self.vel[1]**2))
            self.rebound = incident if incident < pi/2 else pi - incident
            return Ball(self.end, [-self.vel[0], self.vel[1]])
        
        if gradient == 0:
            incident = arccos(self.vel[0]/sqrt(self.vel[0]**2 + self.vel[1]**2))
            self.rebound = incident if incident < pi/2 else pi - incident
            return Ball(self.end, [self.vel[0], -self.vel[1]])
        

        # cos(theta) = a.b / |a||b|
        # Vector of boundary = [1, gradient]
        dot = self.vel[0] + self.vel[1]*gradient
        magnitudes = sqrt((self.vel[0]**2 + self.vel[1]**2) * (1 + gradient**2))
        
        incident = arccos(dot / magnitudes)
        self.rebound = incident if incident < pi/2 else pi - incident
        # Rotate by -2*theta if coming from below, else 2*theta.
        
            
        if self.vel[0] > 0:
            if gradient < self.eqn[0]: # Ball is coming from below.
                rotation = -2*incident
            else: # Ball is coming from above.
                rotation = 2*incident
        
        elif self.vel[0] < 0:
            if gradient < self.eqn[0]: # Ball is coming from above.
                rotation = 2*incident
            else: # Ball is coming from below.
                rotation = -2*incident
        
        else: # vx == 0, ball moves vertically.
            if self.vel[1] > 0: # Ball is coming from below.
                rotation = -2*incident
            else: # Ball is coming from above.
                rotation = 2*incident

        
        # Rotation matrix * velocity = new velocity.
        newVel = [self.vel[0]*cos(rotation) - self.vel[1]*sin(rotation),
                  self.vel[0]*sin(rotation) + self.vel[1]*cos(rotation)]
            
        return Ball(self.end, newVel)
        
            
        
        

class Rectangle:
    """Defines a rectangular boundary."""
    boundary_type = "Rectangular"
    
    def __init__(self, width, height):
        """Construct a rectangular billiards table of width "width" and height "height"."""
        
        self.x = abs(width)/2 # Positive x boundary.
        self.y = abs(height)/2 # Positive y boundary.

        
    def collide(self, ball):
        """Calculate the next collision a ball will make with the table, for a given argument "ball", whose type
           is Ball. Assumes that the ball is moving and is in a valid location (inside the table). A ball on a boundary,
           moving directly along it can produce strange results.
           
           Sets the endpoint (ball.end) for the ball to be the point at which it contacts the table boundary, and
           returns a new ball whose starting position is that same point, and whose velocity is obtained by reflecting
           at the boundary."""
        
        
        y = ball.y(self.x)
        if abs(y) == self.y and ball.vel[0] > 0: # Handle case: ball collides with right corner.
            ball.end = [self.x, y]
            return Ball(ball.end, [-i for i in ball.vel])
        
        y = ball.y(-self.x)
        if abs(y) == self.y and ball.vel[0] < 0: # Handle case: ball collides with left corner.
            ball.end = [-self.x, y]
            return Ball(ball.end, [-i for i in ball.vel])
        
        if ball.y(self.x) is not None: # Handle special case: ball moves vertically.
        
            # Collision with left boundary.                
            y = ball.y(-self.x)
            if abs(y) < self.y and ball.vel[0] < 0:
                ball.end = [-self.x, y]
                return ball.reflect(None)
            
            # Collision with right boundary.                
            y = ball.y(self.x)
            if abs(y) < self.y and ball.vel[0] > 0:
                ball.end = [self.x, y]
                return ball.reflect(None)
        
        # Collision with upper boundary.                
        x = ball.x(self.y)
        if abs(x) < self.x and ball.vel[1] > 0:
            ball.end = [x, self.y]
            return ball.reflect(0)
                
        # Collision with lower boundary.                
        x = ball.x(-self.y)
        if abs(x) < self.x and ball.vel[1] < 0:
            ball.end = [x, -self.y]
            return ball.reflect(0)
                

        raise Exception("Collision not found.")
    
    
    def plot(self, balls):
        """Plot the table boundaries and the motion of a list of given Balls 'balls', up to but excluding the
           final ball in that list (whose endpoint is not expected to be known)."""
        
        plt.plot([-self.x, self.x, self.x, -self.x, -self.x], [self.y, self.y, -self.y, -self.y, self.y], label="Boundary")
        plt.plot([balls[b].pos[0] for b in range(len(balls) - 1)], [balls[b].pos[1] for b in range(len(balls) - 1)], label="Ball path")
        plt.title("Motion of ball reflected at Rectangular table boundaries")
        plt.xlabel(r"$x$ (m)")
        plt.ylabel(r"$y$ (m)")
        #plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.show()
        
        

class Ellipse:
    """Defines an elliptical boundary."""
    boundary_type = "Elliptical"
    
    def __init__(self, width, height):
        """ Construct an elliptical billiards table of width 'width' and height 'height'."""
        
        # (x/self.x)**2 + (y/self.y)**2 = 1
        self.y = abs(height)/2
        self.x = abs(width)/2
        self.fx = lambda x: self.y*sqrt(1 - (x/self.x)**2) # |y(x)| for ellipse.
    
    
    def grad(self, x, y):
        """Returns the gradient of the Ellipse at a point (x, y), or None if gradient
           is infinite. Accepts arguments x and y."""
        try:
            return sign(y) * -self.y*x / (self.x*sqrt(self.x**2 - x**2))
        except (ValueError, ZeroDivisionError):
            return None
        

    def collide(self, ball):
        """Calculate the next collision a ball will make with the table, for a given argument "ball", whose type
           is Ball. Assumes that the ball is moving and is in a valid location (inside the table).
           
           Sets the endpoint (ball.end) for the ball to be the point at which it contacts the table boundary, and
           returns a new ball whose starting position is that same point, and whose velocity is obtained by reflecting
           at the boundary."""
        
        if ball.vel[0] == 0: # If ball is moving vertically.
            ball.end = [ball.pos[0], self.fx(ball.pos[0]) * sign(ball.vel[1])]
            return ball.reflect(self.grad(*ball.end))
        
        
        else: # If ball is not moving vertically.
            a = (ball.eqn[0] / self.y)**2 + 1 / self.x**2
            b = 2 * ball.eqn[0] * ball.eqn[1] / self.y**2
            c = (ball.eqn[1] / self.y)**2 - 1
            # 1 = (y/h)^2 + (x/w)^2 for ellipse, y = mx + c for ball.
            # Therefore: 0 = ((m/h)^2 + 1/w^2)x^2 + (2mc/h^2)x + (c/h)^2 - 1.
            
            solution = quad(a, b, c, sign(ball.vel[0])) # x-velocity determines which solution is correct.
            ball.end = [solution, ball.y(solution)]
            
            return ball.reflect(self.grad(*ball.end))

    
    
    def plot(self, balls, n=100):
        """Plot the table boundaries and the motion of a list of given Balls 'balls', up to but excluding the
           final ball in that list (whose endpoint is not expected to be known). Optionally takes argument 'n',
           the number of points to plot each quadrant of the ellipse with."""

        x = [x*self.x/n for x in range(-n, n+1)]
        y1 = [self.fx(i) for i in x]
        y2 = [-self.fx(i) for i in x[::-1]]
        y = y1 + y2
        x += x[::-1]
        
        plt.plot(x, y)
        plt.plot([balls[b].pos[0] for b in range(len(balls) - 1)], [balls[b].pos[1] for b in range(len(balls) - 1)])
        plt.title("Motion of ball reflected at Elliptical table boundaries")
        plt.xlabel(r"$x$ (m)")
        plt.ylabel(r"$y$ (m)")
        plt.show()



class Stadium:
    """Defines the boundary of a stadium billiard table."""
    boundary_type = "Stadium"
    
    def __init__(self, width, height):
        """Construct a stadium billiards table of width 'width' and height 'height'. Table's height must be no larger
           than its width."""
           
        if height > width:
            raise Exception("Height of table must be no larger than width.")
        
        self.x = abs(width)/2
        self.y = abs(height)/2 # Also equal to radius. 
        
        
    
    def fx(self, x):
        """Returns the positive y-coordinate (the magnitude) of the point on the boundary for a given x-coordinate 'x'."""
        
        x0 = self.x - self.y # X-coordinate of right circle's centre. 
        
        if abs(x) < x0: # Pertains to rectangular region.
            return self.y
        
        else: # Pertains to a semi-circular region.
            return sqrt(self.y**2 - (abs(x) - x0)**2) # Symmetrical table, so just take positive x.
        
    
    def grad(self, x, y):
        """Returns the gradient of the split semi-circle at a point (x, y). Only valid for points on the semi-circle,
           and not in the rectangular region. Returns None if gradient is infinite."""
        
        x0 = sign(x)*(self.x - self.y) # Centre of circle.
        
        try:
            return (x0 - x)/y
        except ZeroDivisionError:
            return None
    
    
    def solve(self, ball, leftright, addsub):
        """Solve (for x) the quadratic equation produced by a Ball's (argument 'ball') line equation and one of
           the two semi-circles. To specify the left semi-circle, 'leftright' should be set to -1, and should be
           set to 1 for the right semi-circle.
           
           Given that each semi-circle (essentially treated as a circle for calculation) will produce 2
           solutions, 'addsub' can be used to specify whether it will be the left or right
           solution: +1 specifies right and -1 specifies left. This essentially pertains to whether
           the determinant is added or subtracted in the solution for x."""
       
        if abs(leftright) != 1 or abs(addsub) != 1:
            raise ValueError("'leftright' and 'addsub' should be either +1 or -1.")
        
        a = ball.eqn[0]**2 + 1
        b = 2*(ball.eqn[0]*ball.eqn[1] - leftright*(self.x - self.y))
        c = ball.eqn[1]**2 + (self.x - self.y)**2 - self.y**2
        
        return quad(a, b, c, addsub)
    
    
    def collide(self, ball):
        """Calculate the next collision a ball will make with the table, for a given argument "ball", whose type
           is Ball. Assumes that the ball is moving and is in a valid location (inside the table). A ball on a boundary,
           moving directly along it can produce strange results.
           
           Sets the endpoint (ball.end) for the ball to be the point at which it contacts the table boundary, and
           returns a new ball whose starting position is that same point, and whose velocity is obtained by reflecting
           at the boundary."""
        
        # Handle special case: ball moves directly up/down.
        if ball.eqn[0] is None:
            ball.end = [ball.pos[0], self.fx(ball.pos[0])*sign(ball.vel[1])]
            return ball.reflect( 0 if abs(ball.end[0]) < self.x - self.y else self.grad(*ball.end) )
        
        # Check if collide with top or bottom.
        if ball.vel[1] > 0: # Check if collide with top.
            x = ball.x(self.y)
            if abs(x) < self.x - self.y:
                ball.end = [x, self.y]
                return ball.reflect(0)
        
        if ball.vel[1] < 0: # Check if collide with bottom
            x = ball.x(-self.y)
            if abs(x) < self.x - self.y:
                ball.end = [x, -self.y]
                return ball.reflect(0)
        
        
        # Now check for collisions with semi-circles.
        
        if abs(ball.y(-sign(ball.pos[0])*(self.x - self.y))) > self.y:
            # ^- Check if the ball line intersects the opposite rectangle end. If it doesn't, then it couldn't 
            #    possibly collide with the opposite semi-circle.
            
            # Two possible collisions with a single semi-circle, on same side as ball.
            # Determine which semi-circle is correct by sign(x), and which solution is correct by sign(v_x).
            
            x = self.solve(ball, sign(ball.pos[0]), sign(ball.vel[0]))
            ball.end = [x, ball.y(x)]
            return ball.reflect(self.grad(*ball.end))
        
        else:
            # A single collision possible with either semi-circle.
            # Determine both which semi-circle and which solution is correct based on sign(v_x).
            x = self.solve(ball, sign(ball.vel[0]), sign(ball.vel[0]))
            ball.end = [x, ball.y(x)]
            return ball.reflect(self.grad(*ball.end))
        
    
    def plot(self, balls, n=100):
        """Plot the table boundaries and the motion of a list of given Balls 'balls', up to but excluding the
           final ball in that list (whose endpoint is not expected to be known). Optionally takes argument 'n',
           the number of points to plot each quadrant with."""
        
        x = [i*self.x/n for i in range(-n, n+1)]
        y_pos = [self.fx(i) for i in x]
        y_neg = [-i for i in y_pos]
        x, y = x + x[::-1], y_pos + y_neg[::-1]
        
        plt.plot(x, y)
        #plt.plot([-i for i in x], y, 'b')
        
        plt.plot([balls[b].pos[0] for b in range(len(balls) - 1)], [balls[b].pos[1] for b in range(len(balls) - 1)])
        plt.title("Motion of ball reflected at Stadium table boundaries")
        plt.xlabel(r"$x$ (m)")
        plt.ylabel(r"$y$ (m)")
        plt.show()



#########################
### Testing functions ###
#########################


def ellipse_eccentricity_rebound_angle(eccentricities, x_ball=0, angle_range=[0, pi], n=1000):
    """Plot angle of rebound against angle of ball motion about origin. Takes list of eccentricities 'eccentricities',
       for elliptical table, the horizontal starting position of the ball 'x_ball', the range of angles to launch the ball
       from 'angle_range', and the number of angles within that range to test 'n'. 
       
       Any given eccentricity must be such that: 1 > eccentricity >= 0.
       Table width will always be 1, so don't set x_ball to something outside of this."""
    
    
    
    balls = []
    angles = [] # Angle of ball velocity about origin.
    
    diff = angle_range[1] - angle_range[0]
    
    for i in range(n):
        angles.append(angle_range[0] + i*diff/(n-1))
        balls.append(Ball([x_ball, 0], [cos(angles[-1]), sin(angles[-1])]))
    
    for ecc in eccentricities:
        table = Ellipse(1, sqrt(1-ecc*ecc))
        
        for i in range(n):
            table.collide(balls[i]) # Set rebound angles for ball in balls.
        
        plt.plot(angles, [ball.rebound for ball in balls], label=f"e = {ecc}")

    plt.ylabel("Angle of rebound (radians)")
    plt.xlabel("Ball trajectory (radians)")
    plt.title(f"Rebound angle against ball trajectory for elliptical table,\n" +\
              f"varying eccentricities, ball start = {(x_ball, 0)}")
    plt.legend()
    plt.show()



def rate_of_separation(table, initial, n_repeat, n_max=1000):
    """This simulation is not perfect, and so for periodic trajectories, balls will likely begin to deviate
       from the expected path as more and more rebounds are made. This function will plot a graph of separation
       from the expected point for an increasing number of rebounds. 
       
       Takes as arguments the table object being tested on, 'table', the starting Ball, 'ball', the number
       of rebounds expected before the ball overlaps its original position, 'n', and the maximum number of
       reflections to be calculated, 'n_max'."""
    
    balls = get_balls(table, initial, n_max)[::n_repeat]
    sep = lambda b1, b2: sqrt((b1.pos[0] - b2.pos[0])**2 + (b1.pos[1] - b2.pos[1])**2) # Radial separation function.
    separations = [sep(balls[0], b) for b in balls]
    num_reflections = [n_repeat*i for i in range(len(separations))] # The horizontal axis, number of reflections.
    
    plt.plot(num_reflections, separations)
    plt.title(f"Radial separation of collision from expected position,\n{table.boundary_type} table")
    plt.xlabel("Number of reflections")
    plt.ylabel("Separation (m)")
    plt.ylim(0)
    plt.xlim(0, num_reflections[-1])
    plt.show()
    

def phase_line(balls, table_type=""):
    """Function to plot the motion of a list of Balls 'balls' in (x, vx)-space. Can be given a string
       'table_type' specifying the type of table used, which will appear in the title of the plot."""
    
    for b in balls[:-1]:
        plt.plot([b.pos[0], b.end[0]], [b.vel[0], b.vel[0]])
    plt.title(f"{table_type} table phase space plot of billiard ball in x dimension")
    plt.xlabel(r"$x$ (m)")
    plt.ylabel(r"$v_x$ (m/s)")
    plt.show()



def lyapunov(func, table, initial_1, initial_2, n=500, show=True, sep_type="", unit=""):
    """Numerically compute the Lyapunov exponent for a given 'table' and initial balls 'initial_1' and 'initial_2'.
       Separation of balls between balls is calculated by function 'func', and must be of the form func(ball1, ball2),
       and return a scalar value pertaining to their separation in some way (be it angular, radial separation, etc).
       Lyapunov exponent will be calculated between successive positions and then averaged over 'n' collisions.
       
       If 'show' is True, then the separation will be plotted as a function of number of reflections. If it is False,
       it will not be. 'sep_type' will be the type of separation, eg. radial, angular..., and should be a string."""
    
    ball_1 = deepcopy(initial_1)
    ball_2 = deepcopy(initial_2)
    sep = [func(initial_1, initial_2)]
    
    for _ in range(n):
        ball_1 = table.collide(ball_1)
        ball_2 = table.collide(ball_2)
        sep.append(func(ball_1, ball_2))
    
     # Don't start at First position, as separation may be zero.
     # This is exceedingly unlikely to occur at any other point for chaotic tables.
     
    lyapunov_i = lambda i: log(sep[i+1]/sep[i]) # Function to get ith Lyapunov exponent.
    lyapunov_avg = sum((lyapunov_i(i+1) for i in range(n-1))) / (n-1)
    
    if show: # Display results.
        print(table.boundary_type, f"table, width: {2*table.x}, height: {2*table.y}")
        print("Ball 1 position:", initial_1.pos, "-- Ball 1 velocity:", initial_1.vel)
        print("Ball 2 position:", initial_2.pos, "-- Ball 2 velocity:", initial_2.vel)
        print("Lyapunov exponent:", lyapunov_avg)
        print("\n")
    
        plt.plot(list(range(n+1)), sep, ".")
        plt.title(f"Evolution of ball {sep_type} separation,\n{table.boundary_type} table")
        plt.ylabel(f"Separation ({unit})")
        plt.xlabel("Number of collisions")
        plt.ylim(0)
        plt.xlim(0, n)
        plt.show()
    
    
    return lyapunov_avg
    

        
#########################################
### Functions for individual sections ###
#########################################

### 3.1

def periodic_rectangle():
    """Plots a single periodic trajectory on a rectangular table, for section 3.1."""
    
    table = Rectangle(2, 1) # Create table.
    balls = get_balls(table, Ball([0, -0.5], [-2, 1]), 4) # Obtain collisions.
    table.plot(balls) # Show.

# All subsequent functions for individual sections follow the same 'create table/obtain collisions/show' structure.

def periodic_ellipse():
    """Plots a single periodic trajectory on an elliptical table, for section 3.1."""
    
    table = Ellipse(2, 1)
    balls = get_balls(table, Ball([sqrt(4/5), -table.fx(sqrt(4/5))], [0, 1]), 4)
    table.plot(balls)

def periodic_stadium():
    """Plots a single periodic trajectory on stadium table, for section 3.1."""
    
    table = Stadium(2, 1)
    balls = get_balls(table, Ball([0.5 + 1/(2*sqrt(2)), -1/(2*sqrt(2))], [0, 1]), 4)
    table.plot(balls)

def divergence_rectangle():
    """Plot divergence from expected periodic path over a large number of reflections,
       for the rectangular table. For section 3.1."""
    
    table = Rectangle(2, 1)
    init = Ball([0, -0.5], [-2, 1])
    rate_of_separation(table, init, 4, 1000)

def divergence_ellipse():
    """Plot divergence from expected periodic path over a large number of reflections,
       for the elliptical table. For section 3.1."""
    
    table = Ellipse(2, 1)
    init = Ball([sqrt(4/5), -table.fx(sqrt(4/5))], [0, 1])
    rate_of_separation(table, init, 4, 1000)

def divergence_stadium():
    """Plot divergence from expected periodic path over a large number of reflections,
       for the stadium table. For section 3.1."""
    
    table = Stadium(2, 1)
    init = Ball([0.5 + 1/(2*sqrt(2)), -1/(2*sqrt(2))], [0, 1])
    rate_of_separation(table, init, 4, 1000)


### 3.2

def phase_rectangle():
    """Plot the (x, vx)-space diagram for the Rectangular table when ball is initialised
       with diagonal velocity. For section 3.2."""
    
    table = Rectangle(2, 1)
    balls = get_balls(table, Ball([0, -0.5], [1, 2]), 100)
    phase_line(balls, table.boundary_type)

def phase_ellipse():
    """Plot the (x, vx)-space diagram for the Elliptical table when ball is initialised
       with diagonal velocity. For section 3.2."""
    
    table = Ellipse(2, 1)
    balls = get_balls(table, Ball([0, -0.5], [1, 2]), 100)
    phase_line(balls, table.boundary_type)

def phase_stadium():
    """Plot the (x, vx)-space diagram for the Stadium table when ball is initialised
       with diagonal velocity. For section 3.2."""
    
    table = Stadium(2, 1)
    balls = get_balls(table, Ball([0, -0.5], [1, 2]), 100)
    phase_line(balls, table.boundary_type)

def angle_ellipse():
    """Plots angle of rebound against angle of trajectory for Elliptical table. For section 3.2."""
    
    ellipse_eccentricity_rebound_angle([0, 0.2, 0.4, 0.6, 0.8, 0.999]) # x = 0
    ellipse_eccentricity_rebound_angle([0, 0.2, 0.4, 0.6, 0.8, 0.999], x_ball=0.25) # x = 0.25


### 3.3

def lyapunov_rectangle():
    """Calculate the lyapunov exponent for two similar trajectories on the Rectangular table,
       based on radial and angular separation, and plot the radial separation as a function of the number
       of reflections. For section 3.3."""
    
    table = Rectangle(2, 1)
    b1 = Ball([0.51, 0], [2, 3]) # Arbitrarily chosen.
    b2 = Ball([0.5, 0], [2, 3])
    
    radius = lambda ba, bb: sqrt((ba.pos[0] - bb.pos[0])**2 + (ba.pos[1] - bb.pos[1])**2)
    angle = lambda b1, b2: abs(arctan2(b1.pos[1], b1.pos[0]) - arctan2(b2.pos[1], b2.pos[0]))
    lyapunov(radius, table, b1, b2, n=2000, sep_type="radial", unit="m")
    lyapunov(angle, table, b1, b2, n=2000, sep_type="angular", unit="radians")

def lyapunov_ellipse():
    """Calculate the lyapunov exponent for two similar trajectories on the Elliptical table,
       based on radial and angular separation, and plot the radial separation as a function of the number
       of reflections. For section 3.3."""
    
    table = Ellipse(2, 1)
    b1 = Ball([0.51, 0], [2, 3]) # Arbitratily chosen.
    b2 = Ball([0.5, 0], [2, 3])
    
    radius = lambda ba, bb: sqrt((ba.pos[0] - bb.pos[0])**2 + (ba.pos[1] - bb.pos[1])**2)
    angle = lambda b1, b2: abs(arctan2(b1.pos[1], b1.pos[0]) - arctan2(b2.pos[1], b2.pos[0]))
    lyapunov(radius, table, b1, b2, n=2000, sep_type="radial", unit="m")
    lyapunov(angle, table, b1, b2, n=2000, sep_type="angular", unit="radians")


def lyapunov_stadium():
    """Calculate the lyapunov exponent for two similar trajectories on the Stadium table,
       based on radial and angular separation, and plot the radial separation as a function of the number
       of reflections. For section 3.3."""
    
    table = Stadium(2, 1)
    b1 = Ball([0.51, 0], [2, 3]) # Arbitratily chosen.
    b2 = Ball([0.5, 0], [2, 3])
    
    radius = lambda ba, bb: sqrt((ba.pos[0] - bb.pos[0])**2 + (ba.pos[1] - bb.pos[1])**2)
    angle = lambda b1, b2: abs(arctan2(b1.pos[1], b1.pos[0]) - arctan2(b2.pos[1], b2.pos[0]))
    lyapunov(radius, table, b1, b2, n=2000, sep_type="radial", unit="m")
    lyapunov(angle, table, b1, b2, n=2000, sep_type="angular", unit="radians")

def avg_lyapunov():
    """Return the average lyapunov exponent for each table type, over a number of trajectories, for
       both angular and radial separation. For section 3.3."""
    
    radius = lambda ba, bb: sqrt((ba.pos[0] - bb.pos[0])**2 + (ba.pos[1] - bb.pos[1])**2)
    angle = lambda b1, b2: abs(arctan2(b1.pos[1], b1.pos[0]) - arctan2(b2.pos[1], b2.pos[0]))
    
    x_range = [0.1*i for i in range(6)] # Range of x-coordinates to start at (y = 0 always).
    v_range = [ [1, 5], [2, 4], [3, 3], [4, 2], [5, 1] ] # Range of velocities.
    
    rect_lyap_rad, ell_lyap_rad, stad_lyap_rad = [], [], [] # Lists of computed radial Lyapunov exponents for tables.
    rect_lyap_ang, ell_lyap_ang, stad_lyap_ang = [], [], [] # Lists of computed angular Lyapunov exponents for tables.
    rect, ell, stad = Rectangle(2, 1), Ellipse(2, 1), Stadium(2, 1) # Tables.
    
    for x in x_range:
        for v in v_range:
            # Offset y position by 0.1 between balls to act as slight difference in initial conditions.
            rect_lyap_rad.append(lyapunov(radius, rect, Ball([x, 0], v), Ball([x, 0.1], v), n=2000, show=False))
            rect_lyap_ang.append(lyapunov(angle, rect, Ball([x, 0], v), Ball([x, 0.1], v), n=2000, show=False))
            
            ell_lyap_rad.append(lyapunov(radius, ell, Ball([x, 0], v), Ball([x, 0.1], v), n=2000, show=False))
            ell_lyap_ang.append(lyapunov(angle, ell, Ball([x, 0], v), Ball([x, 0.1], v), n=2000, show=False))
            
            stad_lyap_rad.append(lyapunov(radius, stad, Ball([x, 0], v), Ball([x, 0.1], v), n=2000, show=False))
            stad_lyap_ang.append(lyapunov(angle, stad, Ball([x, 0], v), Ball([x, 0.1], v), n=2000, show=False))
            
    
    print(f"Rectangle radial separation Lyapunov exponent: {mean(rect_lyap_rad)} +/- {std(rect_lyap_rad)}")
    print(f"Ellipse radial separation Lyapunov exponent: {mean(ell_lyap_rad)} +/- {std(ell_lyap_rad)}")
    print(f"Stadium radial separation Lyapunov exponent: {mean(stad_lyap_rad)} +/- {std(stad_lyap_rad)}")
    
    print(f"Rectangle angular separation Lyapunov exponent: {mean(rect_lyap_ang)} +/- {std(rect_lyap_ang)}")
    print(f"Ellipse angular separation Lyapunov exponent: {mean(ell_lyap_ang)} +/- {std(ell_lyap_ang)}")
    print(f"Stadium angular separation Lyapunov exponent: {mean(stad_lyap_ang)} +/- {std(stad_lyap_ang)}")

        
        
############
### Main ###
############
    


def main():
    """The main function, to be run when the script is executed directly. Produces all data in report. Asks for
       user to tell it which data to produce, and then produces it."""
    
    while True:
        print("Enter the number corresponding to the section you would like to access the tests for, or 0 to exit:")
        choice = input("(0) -- Exit\n(1) -- Periodic behaviour\n(2) -- Phase space plots\n(3) -- Lyapunov exponents\n")
        print("\n")
        
        if choice == "0":
            break # Exit.
        
        elif choice == "1": # Section 1.
            while True:
                print("Enter the number corresponding to the test you would like to run:")
                choice = input("(0) -- Back to selection\n(1) -- Plot rectangle periodic trajectory\n(2) -- Plot ellipse periodic trajectory\n(3) -- Plot stadium periodic trajectory\n(4) -- Plot rectangle divergence\n(5) -- Plot ellipse divergence\n(6) -- Plot stadium divergence\n")
                print("\n")
                
                if choice == "0":
                    break # Return.
                
                try:
                    [periodic_rectangle, periodic_ellipse, periodic_stadium,
                     divergence_rectangle, divergence_ellipse, divergence_stadium][int(choice) - 1]() # Call.
                
                except (ValueError, IndexError):
                    print("Oops, that wasn't a valid input. Try again.")
                   
        
        elif choice == "2": # Section 2.
            while True:
                print("Enter the number corresponding to the test you would like to run:")
                choice = input("(0) -- Back to selection\n(1) -- Plot rectangle phase diagram\n(2) -- Plot ellipse phase diagram\n(3) -- Plot stadium phase diagram\n(4) -- Plot ellipse rebound angle graphs\n")
                print("\n")
                
                if choice == "0":
                    break # Return.
                
                try:
                    [phase_rectangle, phase_ellipse, phase_stadium, angle_ellipse][int(choice) - 1]() # Call.
                
                except (ValueError, IndexError):
                    print("Oops, that wasn't a valid input. Try again.")
                    
        
        elif choice == "3": # Section 3.
            while True:
                print("Enter the number corresponding to the test you would like to run:")
                choice = input("(0) -- Back to selection\n(1) -- Get separation plots for rectangle\n(2) -- Get separation plots for ellipse\n(3) -- Get separation plots for stadium\n(4) -- Print average Lyapunov exponentials\n")
                print("\n")
                
                if choice == "0":
                    break # Return.
                
                try:
                    [lyapunov_rectangle, lyapunov_ellipse, lyapunov_stadium, avg_lyapunov][int(choice) - 1]() # Call.
                
                except (ValueError, IndexError):
                    print("Oops, that wasn't a valid input. Try again.")
                    
        
        else:
            print("Oops, that wasn't a valid input. Try again.")
        


if __name__ == "__main__":
    main() # Call main function
