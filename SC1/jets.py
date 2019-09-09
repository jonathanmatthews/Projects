# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:00:40 2018

@author: Jonathan Matthews (jxm1027)

Python 3 script for reading in Pythia 8 event data and databases, and building jets
from that event data. Containing classes ParticleData, ParticleDatabase, Particle,
and Jet. 

To run this script as the main program, the Pythia 8 particle database should be
included in the working directory of this script, as well as the jets.dat events file.
This program expects partons to be assigned a status of 0, and final state particles
to be assigned a status of 3.

REQUIREMENTS:  matplotlib (and dependencies).
"""

from math import acos as arccos, sqrt, log, pi
from shlex import split
from matplotlib import pyplot as plt

#####################################################
###  ParticleDatabase class from problem sheet 2.  ##
#####################################################


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


##########################################
##  Jet building classes and functions. ##
##########################################

def phi(p):
    """Calculate and returns the angle phi for a given particle's fourmomentum, of some
       iterable type and length 4, in the form [E, px, py, pz]."""
    
    angle = arccos(p[2]/sqrt(p[1]**2 + p[2]**2 + p[3]**2))
    # Projected into the xy plane.
    return angle if p[1] > 0 else 2*pi - angle

def y(p): # p is four momentum (E, px, py, pz).
    """Calculates the rapidity of a particle for a given fourmomentum, of some iterable
       type and length 4, in form [E, px, py, pz]."""
    return 0.5*log((p[0]+p[3])/(p[0]-p[3]))

def pT(p): # p is four momentum (E, px, py, pz).
    """Calculates the transverse momentum for a given particle's fourmomentum, of some
       iterable type and length 4, in form [E, px, py, pz]."""
    return sqrt(p[1]**2 + p[2]**2)

def deltaRsq(p1, p2):
    """Calculates the delta R for any two particles. Takes two arguments, the fourmomenta of
       each particle."""
    
    dphi = abs(phi(p2) - phi(p1))
    dphi = dphi if dphi < pi else 2*pi - dphi
    return (y(p1) - y(p2))**2 + dphi**2

def dij(p1, p2, R=0.5, k=0): # p1 and p2 are four momenta for the two particles.
    """Calculates and returns the distance metric between two particles, whose fourmomenta should
       be given as the first two arguments, as iterables of length 4 (p1 and p2). The third
       argument (R) is the radius parameter, defaulting to 0.5. The fourth argument (k)
       defines the algorithm type used in jet clustering.
       
       k = -1 (anti-k clustering), 0 (Cambridge-Aachen), 1 (k clustering)."""
    
    return min(pT(p1)**(2*k), pT(p2)**(2*k))*deltaRsq(p1, p2)/R**2

def diB(p, k=0): # p is four momentum of a single particle.
    """Calculates the distance metric between a particle and the beam, for a given particle's
       fourmomentum (p) and algorithm type (k).
       
       k = -1 (anti-k clustering), 0 (Cambridge-Aachen), 1 (k clustering)."""
    return pT(p)**(2*k)




class Particle:
    """Particle class, containing particle status, pid and four-momentum, used in jet building."""
    
    def __init__(self, stat, pid, E, px, py, pz):
        """The constructor for the Particle class. Takes as arguments the status, pid, particle
           energy, x momentum, y momentum, and z momentum components."""
        self.stat = int(stat)
        self.pid = int(pid)
        self.p = [E, px, py, pz]
    
    def __repr__(self):
        """Defines the magic method to return a string which can be used to 
           recreate an instance of this class, should that string be evaluated."""
        return f"Particle({self.stat}, {self.pid}, {self.p[0]}, {self.p[1]}, {self.p[2]}, {self.p[3]})"


class Jet:
    """Class for reading in Pythia 8 particle events, and building jets from this event data."""
    
    def __init__(self, event=0,  particlesPath="jets.dat", databasePath="ParticleData.xml"):
        """Constructor for the Jet class. Takes as arguments the event to be read in from the
           Pythia 8 event file (to be separated by single blank lines), the path to the event
           file, and the path to the Pythia 8 particle database file. All of these have default
           values.
           
           When called, will read in the given event from the data at the given locations, and
           will store data as class members."""
        
        self.database = ParticleDatabase(databasePath)
        file = open(particlesPath)
        self.event = [] # List of particles in event.
        eventNum = -1 # Events start at 0.

        for line in file:
            if line[0] == "#": # Line is a comment, ignore. 
                continue
            
            if line == "\n" or line == "\r\n": # New event.
                eventNum += 1
                continue
            
            if eventNum == event: # Only read particles in desired event.
                self.event.append(self.newParticle(line))
            
            if eventNum > event: # Required event has been read, stop here.
                break

        file.close()
                
    
    def newParticle(self, line):
        """Method to create new Particle object from a line in the Pythia 8 event file. Takes
           the line itself, as a string, as the only argument.
           
           Data read from file is too inaccurate, so E must be recalculated when read in, using
           fourmomenta and Pythia 8 particle database."""
           
        line = line.split()
        px, py, pz = (float(i) for i in line[3:])
        pid = int(line[1])
        E = sqrt(px*px + py*py + pz*pz + self.database[pid].mass**2)
        return Particle(int(line[0]), pid, E, px, py, pz)

        
    def sjca(self, k=0):
        """Method to build jets from the event stored in an instance of this class. Jets are built
           only from final state particles, for which stat = 3. Jets are returned in a list, and
           not stored as members.
           
           Takes, as the only argument, the k value which defines the algorithm type. Defaults
           to Cambridge-Aachen. 
           
           k = -1 (anti-k clustering), 0 (Cambridge-Aachen), 1 (k clustering)."""
        
        particles = list(filter(lambda x: x.stat is 2, self.event)) # stat=2 only.
        jets = []
        
        for particle in particles: # Calculate all diB.
            particle.distances = {"beam" : diB(particle.p, k)}
            
        # Each particle will have its own distance dictionary, containing distances to
        # the beam and all other particles. For distance to the beam, the key will be
        # "beam", and for other particles, the key will be their position in the particles
        # list at the time of calculation.
        
        for i, particle1 in enumerate(particles): # Calculate all dij.
            for j, particle2 in enumerate(particles[1+i:]):
                dist = dij(particle1.p, particle2.p, k=k)
                particle1.distances.update({j+i+1 : dist})
                particle2.distances.update({i : dist})
        
        while particles:            
            #if not len(particles)%50: print(len(particles), "Particles remaining") # REMOVE
            
            minDist = float("inf") # value, particle 1, particle 2.
            key1 = 0
            key2 = 0
            
            for i, particle in enumerate(particles): # Get minimum distance.
                for j in particle.distances:
                    if particle.distances[j] < minDist:
                       minDist, key1, key2 = particle.distances[j], i, j
           
            if key2 is not "beam": # If some dij is smallest.
                fourmomentum = (particles[key1].p[i] + particles[key2].p[i]\
                                                  for i in range(4)) # Combine.
                particles.append(Particle(3, 0, *fourmomentum))
                
                del particles[max(key1, key2)] # Delete largest first.
                del particles[min(key1, key2)]
                
                particles[-1].distances = {"beam" : diB(particles[-1].p, k)} # Get diB for new particle.
            
            else: # If some diB is smallest.
                jets.append(particles[key1])
                del particles[key1]

            for particle in particles: # Reset distance dictionaries.
                particle.distances = {"beam" : particle.distances["beam"]}
            
            for i, particle1 in enumerate(particles): # Recalculate distances.
                for j, particle2 in enumerate(particles[1+i:]):
                    dist = dij(particle1.p, particle2.p, k=k)
                    particle1.distances.update({j+i+1 : dist})
                    particle2.distances.update({i : dist})
        
        return jets
        
        

        

# Functions to compare partons and jets.
# All expecet arguments (jet_fourmomentum, parton_fourmomentum).
# Want to minimise all of these.
deltaR = lambda j, p: sqrt(deltaRsq(p, j))
deltapT = lambda j, p: abs(pT(p) - pT(j))
deltaE = lambda j, p: abs(p[0] - j[0])
deltay = lambda j, p: abs(y(j) - y(p))


def compare(func, Jets, Partons):
    """Accepts a list of jets and partons, as Particle objects,
       as arguments 'Jets' and 'Partons'. Pairs individual partons
       and jets based on which pair out of the remaining particles
       produces the lowest delta R. Each pair then has function
       'func' applied to their momentum values. All the results of func
       are then returned as a list."""
       
    partons = Partons[:]
    jets = Jets[:]
    result = []
    
    while partons and jets:
        key = None
        low = float("inf")
        
        for p, parton in enumerate(partons):        
            for j, jet in enumerate(jets):
                curr = deltaR(parton.p, jet.p)
                if curr < low:
                    key = (p, j)
                    low = curr
        
        result.append(func(partons[key[0]].p, jets[key[1]].p))
        del partons[key[0]], jets[key[1]]
        
    return result


    
        
def compareHist(func, events, bins, titleFunction="Function"):
    """Function to plot histrogram for all algorithms, of values of 'func'
       applied to parton-jet pairs. Which events to calculate from should
       be given in the list 'events'."""
    
    heights = [[], [], []]
    
    for k in [0, 1, -1]:
        
        for n in events:
            print(f"\nCalculating event {n}, for k = {k}\n")
            event = Jet(n)
            allJets = event.sjca(k)
            jets = sorted(allJets, key=lambda p: pT(p.p))[-2:]
            partons = list(filter(lambda x: x.stat is 0, event.event))
            comparison = compare(func, jets, partons)
            for i in comparison:
                heights[k].append(i)
    
    for k in [0, 1, -1]:
        plt.hist(heights[k], bins=bins)
        plt.title(f"{titleFunction} for k = {k} jet-parton pairings, {len(events)} events")
        plt.xlabel(titleFunction)
        plt.ylabel("Number of occurrences")
        plt.show()
        
def compareHistPrecalculated(func, Events, AllJets, bins, titleFunction="Function"):
    """Function to plot histogram of 'func' applied to parton-jet pairs, for
       all algorithms. 'bins' should be the bins to use in the resulting
       histograms. 'titleFunction' is the function name which should appear in the
       histogram title. 'Events' and 'AllJets' should be precalculated jets
       and partons, in the form:
    
        Events : [[event1particles], [event2particles]...]
        AllJets : [[[jets1_CA], [jets1_kt], [jets1_antikt]], [[jets2_CA], ...]]"""
    
    events = Events[:]
    allJets = AllJets[:]
    
    hits = [[], [], []]
    
    for k in [0, 1, -1]:
        
        for e, event in enumerate(events):
            jets = sorted(allJets[e][k], key=lambda p: pT(p.p))[-2:] # Get largest 2 pT jets.
            partons = list(filter(lambda x: x.stat is 0, event))
            comparison = compare(func, jets, partons)
            for i in comparison:
                hits[k].append(i)
    
    for k in [0, 1, -1]:
        plt.hist(hits[k], bins=bins)
        plt.title(f"{titleFunction} for k = {k} jet-parton pairings, {len(events)} events")
        plt.xlabel(titleFunction)
        plt.ylabel("Number of occurrences")
        plt.show()
        

def compareSamples(event=0, eventData=["jets.dat", "qqbar.dat", "ccbar.dat", "gg.dat", "bbbar.dat"]):
    """Function to compare the distribution of jets for a given set of sample
       data files. Arguments passed include 'event', an integer indicating
       which event should be clustered (applies to all files), and 'eventData',
       which should be a list containing strings of the paths to the data files."""
    
    for file in eventData:
        jets = Jet(event, file).sjca(-1)
        plt.plot([y(i.p) for i in jets], [phi(i.p) for i in jets], "r.")
        plt.ylim(0)
        plt.title(f"Distribution of jets for {file}")
        plt.xlabel("y")
        plt.ylabel(r"$\phi$")
        plt.show()
    
    



if __name__ == "__main__":

    
    try: # Attempt to load precalculated data if it exists.
        file = open("myEvents.txt") # Load event data for plotting.
        myEvents = eval(file.readline())
        file.close()
        
        file = open("myJets.txt") # Load jet data for plotting.
        myJets = eval(file.readline())
        file.close()
        
    except:
        myEvents = []
        myJets = []
        
        for i in range(30): # [184, 3, 1]
            print(30-i, "remaining")
            Curr = Jet(i)
            myEvents.append(Curr.event)
            myJets.append([Curr.sjca(0), Curr.sjca(1), Curr.sjca(-1)])
            # This order helps with indexing, it's easier to remember.
        
        file = open("myEvents.txt", "w") # Save event data after calculating, to use later.
        file.write(repr(myEvents))
        file.close()
        
        file = open("myJets.txt", "w") # Save jet data after calculating, to use later.
        file.write(repr(myJets))
        file.close()


    
    compareHistPrecalculated(deltaR, myEvents, myJets, [0.5*i for i in range(19)], "Delta R")
    compareHistPrecalculated(deltapT, myEvents, myJets, [2*i for i in range(16)], "pT difference")
    compareHistPrecalculated(deltaE, myEvents, myJets, [200*i for i in range(12)], "Energy difference")
    compareHistPrecalculated(deltay, myEvents, myJets, [0.5*i for i in range(17)], "Rapidity difference")

#    compareSamples()
    
    
