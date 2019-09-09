# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:39:58 2018

@author: Jonathan Matthews (jxm1027)

Module containing classes ParticleData, ParticleDatabase, DiracMatrices, Particle, Integrate,
and Annihilate, to be used for calculating particle collision cross-sections.
"""

from shlex import split
from ps1 import Vector, Matrix, FourVector
from random import random, seed
from scipy.constants import pi, hbar, c
from math import sin, cos

class ParticleData:
    """An instance of this class will hold the data pertaining to a single particle type,
       and may also contain a ParticleData instance pertaining to its anti-particle,
       if it exists."""

    def __init__(self, pid=None, name=None, mass=None, tau=None, spin=None,
                 charge=None, colour=None):
        """The constructor for the ParticleData class. Stores particle ID, particle name, particle
           mass, lifetime, spin, charge and colour information as indepenedent member variables."""
        self.pid = pid
        self.name = name
        self.mass = mass
        self.tau = tau
        self.spin = spin
        self.charge = charge
        self.colour = colour

        self.anti = None

    def __str__(self):
        """Returns a string representation of a ParticleData instance."""
        return "pid = {}, name = {}, mass = {}, tau = {}, spin = {}, charge = {}, colour = {}".\
        format(self.pid, self.name, self.mass, self.tau, self.spin, self.charge, self.colour)

    def __repr__(self):
        """Returns a string which can be used to reproduce an instance of ParticleData."""
        return "ParticleData({}, \"{}\", {}, {}, {}, {}, {})".format(self.pid, self.name,\
                            self.mass, self.tau, self.spin, self.charge, self.colour)



class ParticleDatabase:
    """A particle database containing a list of ParticleData instances for all particles in the
       Pythia 8 database."""

    def __init__(self, path="ParticleData.xml"):
        """The constructor for this class takes as an argument the path to the Pythia 8 particle
           database file, but will assume a default value of a file named "ParticleData.xml"
           located in the current working directory.

           When called, the constructor will read all particle entries in the Pythia 8 database,
           and store the data as ParticleData instances inside the ParticleDatabase instance,
           creating setting anti-particles where they exist inside of these ParticleData
           instances."""

        xml = open(path)

        char = " "
        accumulator = ""
        data = []

        while char: # When char is empty string, end of file has been reached.
            char = xml.read(1)
            if char is "\n":
                char = " " # Separate items found on different lines.

            if not accumulator and char is "<": # Start new entry.
                accumulator += char

            elif accumulator and char is ">": # End of entry reached, add entry to data.
                accumulator = accumulator.strip("<")
                data.append(split(accumulator))
                accumulator = "" # Reset accumulator for new entry.

                if data[-1][0] != "particle":
                    del data[-1] # Remove entries that are not particle data.
                else:
                    data[-1] = data[-1][1:] # Get rid of the useless 'particle' string at start.

            elif accumulator: # End of entry has not been reached yet, continue adding.
                accumulator += char


        xml.close()
        self.particles = []

        for datum in data:
            particle = {}
            for i in range(len(datum)):
                datum[i] = datum[i].split("=")
                particle[datum[i][0]] = datum[i][1]

            try: # If lifetime is available.
                lifetime = float(particle['tau0'])
            except KeyError: # If lifetime doesn't exist.
                lifetime = 0.0

            self.particles.append(ParticleData(pid=int(particle['id']),
                                               name=particle['name'],
                                               mass=float(particle['m0']),
                                               tau=lifetime,
                                               spin=int(particle['spinType']),
                                               charge=int(particle['chargeType']),
                                               colour=int(particle['colType'])))

            try: # If antiname exists.
                self.particles.append(ParticleData(pid=-int(particle['id']),
                                                   name=particle['antiName'],
                                                   mass=float(particle['m0']),
                                                   tau=lifetime,
                                                   spin=int(particle['spinType']),
                                                   charge=-int(particle['chargeType']),
                                                   colour=int(particle['colType'])))

                self.particles[-1].anti = self.particles[-2]
                self.particles[-2].anti = self.particles[-1]

            except KeyError: # If antiname doesn't exist.
                pass

    def __getitem__(self, index):
        """Defines the magic method for retrieving particles by either name or particle ID.
           Returns an instance of ParticleData."""
        for particle in self.particles:
            if particle.pid == index or particle.name == index:
                return particle

        raise IndexError("Particle ID or name not found.")

    def __setitem__(self, index, new):
        """Defines the method for setting the value of the list entry at which a ParticleData
           instance is stored. Can be accessed either by PID or particle name."""
        for i in range(len(self.particles)):
            if self.particles[i].pid == index or self.particles[i].name == index:
                self.particles[i] = new



class DiracMatrices:
    """Class containing the four raised-index Dirac matrices, as separate Matrix instances."""

    def __init__(self):
        self.matrices = [Matrix([0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
                         Matrix([0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]),
                         Matrix([0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]),
                         Matrix([0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0])]

    def __getitem__(self, index):
        """Defines the magic method for retrieving one of the four raised-index Dirac matrices
           from an instance of the DiracMatrices class. Returns a Matrix instance."""
        return self.matrices[index]

    def __len__(self):
        """Magic method for obtaining the length of an instance of this class. Returns the int 4,
           given that there are four raised-index Dirac matrices."""
        return 4

class Particle:
    """A class representing a single particle for given ParticleData, with explicitly set
       momentum FourVector and helicity."""

    def __init__(self, data, p, h):
        """The constructor for the Particle class. Expects to receive ParticleData for the
           particle type the instance is to represent (data), as well as the momentum
           FourVector (p), and helicity eigenvalue (h). These values will be stored within
           the class instance."""

        self.data = data
        self.h = h
        if p[0] >= 0:
            self.p = p
        else:
            self.p = FourVector((p[1]**2 + p[2]**2 + p[3]**2 + data.mass**2)**0.5,
                                p[1], p[2], p[3])


    def w(self):
        """Returns the Dirac spinor for the particle represented by an instance of the this class.
           Object returned is of type Vector."""

        q = sum(map(lambda x: x*x, self.p[1:]))**0.5
        k = lambda hel: [q + self.p[3], 1j*self.p[2] + hel*self.p[1]][::int(hel)] # h is +1 or -1.

        try:
            xi = 1/(2*q*(q + self.p[3]))**0.5
            limit = False
        except ZeroDivisionError: # If p3 -> -q.
            xi = 1
            limit = True

        if self.data.pid < 0: # Anti-particle, so return v.
            if limit:
                k = [0, -self.h][::-int(self.h)] # h is +1 or -1.
            else:
                k = k(-self.h)

            a = -self.h*(self.p[0] + self.h*q)**0.5
            b = self.h*(self.p[0] - self.h*q)**0.5
            return xi*Vector(k[0]*a, k[1]*a, k[0]*b, k[1]*b)

        else: # Particle, so return u.
            if limit:
                k = [0, self.h][::int(self.h)] # h is +1 or -1.
            else:
                k = k(self.h)

            a = (self.p[0] - self.h*q)**0.5
            b = (self.p[0] + self.h*q)**0.5
            return xi*Vector(k[0]*a, k[1]*a, k[0]*b, k[1]*b)



    def wbar(self):
        """Returns ubar or vbar for a given instance of the Particle class. Return object is of
           type Vector."""
        wConj = ~self.w()
        return Vector(wConj[2], wConj[3], wConj[0], wConj[1])



class Integrator:
    """Defines a Monte Carlo integrator class for a function in two variables. Integrates over
       a given range for both variables."""

    def __init__(self, fn, xmin, xmax, ymin, ymax):
        """The constructor for the Integrator class. Takes as arguments the function to be
           integrated (fn), the lower and upper bounds of the range of the first variable of
           integration (xmin and xmax), and the lower and upper bounds of the range of the
           second variable of integration (ymin and ymax).
           These are then stored as class members."""

        self.fn = fn
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def mc(self, samples=1000):
        """This functions performs the integration defined by an instance of this class. It
           takes as an argument the number of samples to be used for the integration (samples),
           with a default value of 1000."""

        seed(0)
        xwidth = self.xmax - self.xmin
        ywidth = self.ymax - self.ymin
        total = 0

        for _ in range(samples):
            point = (self.xmax - random()*xwidth, self.ymax - random()*ywidth)
            total += self.fn(*point)

        return total*xwidth*ywidth/samples

def circle(x, y):
    """Defines a function which takes the coordinates for a point (x and y) as arguments and
       returns a 1 if that point is within the unit circle, or 0 otherwise."""
    return 1 if x*x + y*y <= 1 else 0


minkowski = Matrix([1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1])

class Annihilate:
    """Class defining an annihilation interaction between two particles, producing two particles,
       which calculates the cross-section for that interaction."""

    def __init__(self, p1, p2, p3, p4):
        """The constructor for the Annihilate method. Takes four instances of the Particle class,
           each representing one of the particles in the interaction.

           p1 and p2 -- The initial particles.
           p3 and p4 -- The resulting particles."""

        self.dmu = DiracMatrices()
        self.dml = DiracMatrices()
        self.dml.matrices = [minkowski[i, i]*dm for i, dm in enumerate(self.dmu.matrices)]

        self.p1, self.p2, self.p3, self.p4 = p1, p2, p3, p4

    def me(self):
        """Calculates and returns the matrix element for this interaction. Takes no arguments."""

        p0 = self.p1.p + self.p2.p
        total = 0
        for i in range(4):
            total += (self.p3.wbar()*self.dmu[i]*self.p4.w()) * (self.p2.wbar()*self.dml[i]*self.p1.w())

        return -4*pi*total / (137*abs(p0)**2)

    def xs(self, phi, theta):
        """Calculates and returns the cross-section element, d sigma, for the interaction defined
           by an instance of this class, for given phi and theta."""

        s = 0.5 if self.p3.data.pid == self.p4.data.pid else 1
        q = (self.p1.p[0]**2 - self.p3.data.mass**2)**0.5

        momentum3 = lambda phi2, theta2: FourVector(self.p1.p[0], q*sin(theta2)*cos(phi2),\
                                                 q*sin(theta2)*sin(phi2), q*cos(phi2))

        momentum4 = lambda phi2, theta2: FourVector(self.p1.p[0], -q*sin(theta2)*cos(phi2),\
                                                 -q*sin(theta2)*sin(phi2), -q*cos(phi2))

        constantStuff = ((hbar*c/(8*pi))**2)*s/(((self.p1.p[0] + self.p2.p[0])**2)*\
                         (sum(map(lambda x: x*x, self.p1.p[1:]))**0.5))
                            # The stuff that doesn't change with phi and theta.

        self.p3.p = momentum3(phi, theta) # Set momentum for particles 3 and 4, given phi and theta.
        self.p4.p = momentum4(phi, theta)

        M = self.me()
        return constantStuff*(sum(map(lambda x: x*x, self.p3.p[1:]))**0.5)*(M.conjugate()*M).real*sin(theta)



def calculateXS(A, B, C, D):
    """Function for obtaining the cross-section of the e+e- -> mu+mu- interaction. Intended for
       internal use only, helicity values should be given in the argument."""
    database = ParticleDatabase()
    e = Particle(database["e-"], FourVector(-1, 0, 0, 100), A) # Negative p0 will be recalculated.
    p = Particle(database["e+"], FourVector(-1, 0, 0, -100), B)
    muMinus = Particle(database["mu-"], FourVector(-1, 0, 0, 1), C) # Dummy FourVector.
    muPlus = Particle(database["mu+"], FourVector(-1, 0, 0, 1), D)

    collision = Annihilate(e, p, muMinus, muPlus)
    sigma = Integrator(collision.xs, 0, 2*pi, 0, pi).mc()

    return sigma
    