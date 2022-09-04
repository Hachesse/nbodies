#!/usr/bin/python3
#################################################################################
# GRAVITY - a python version of gravity simulation                              #
#################################################################################
# Henri Souchay, 1986 - 2022                                                    #
# Python port of original GFA BASIC Code for Atari ST                           #
#################################################################################

import pygame
import sys
import random
import math
import numpy as np
from pygame.locals import KEYDOWN, KEYUP, MOUSEBUTTONDOWN, MOUSEBUTTONUP, QUIT


class Universe:
    def __init__(self,display,nplanets=2):
        """
        A universe consists of:
        - planets (a ragular array), the number of which may be passed at creation
        - 3 displays: one for each of planet discs, trajectories, and help
        - a set of physical constants that determine the its dynamics
        - an adaptive timestep for calculations
        - a table of mutual forces between planets, used for calculations
        """
        self.planets = []

        # displays
        self.display = display.copy() # main display layer
        self.trace = self.display.copy() # planet trace display layer
        self.helpdisp = self.display.copy() # help display layer
        self.traceattenuation=5 # speed of attenuation of the trace
        self.display.set_colorkey((0,0,0))
        self.displayHelp = 0
        self.helpdisp.set_colorkey((0,0,0))
        self.helpdisp.set_alpha(200)
        display_rect = self.display.get_rect()
        self.screenw=display_rect.width
        self.screenh=display_rect.height/2
        self.msg="" # info message displayed on top line
        self.fontsize = 24
        self.font = pygame.font.Font(pygame.font.match_font('Courier'), self.fontsize)

        # physics starts here
        self.G=1000.0 # gravity constant
        self.dCollision=5.0 # effective collision radius factor (ie divides radius to apply collision test)
        self.IntegrationMethod=1 # 0=Euler, 1=Verlet
        self.planet_maxid = 0
        self.maxInitVelocity=3.
        self.minPlanetMass=0.3
        self.maxPlanetInitialMass=400.
        self.planetexploding=-1
        self.bounds=self.screenw # size of universe, planets beyond that are lost
        self.timestep=0.1
        self.Time=0.
        self.step=0
        self.calcspeed=0.2
        self.forces=np.zeros((len(self.planets),3)) # array of mutual forces (indexess are nplanet and x,y,z)

        self.initPlanets(nplanets)
        

    def initPlanets(self,nplanets):
        """
        fills the universe with a variety of objects
        distributions and ranges are important for the overall 'naturalness' and balance of the program
        """
        self.planets=[]
        if nplanets>0:
            # initial position shall be within center half of screen
            sw=self.screenw/4
            sh=self.screenh/4
            # store momentum
            mv=np.array([0.,0.,0.])
            for i in range(nplanets):
                mass=random.lognormvariate(self.minPlanetMass*3, self.maxPlanetInitialMass/40)
                while(mass<self.minPlanetMass or mass>self.maxPlanetInitialMass): # repeat until mass within range
                    mass=random.lognormvariate(self.minPlanetMass*3, self.maxPlanetInitialMass/40)
                # initial positions pulled in centered uniform distribution 
                position = np.array([random.gauss(0,sw/2), random.gauss(0,sh), random.gauss(0,sh)])
                # initial speeds pulled in centered uniform distribution
                # gaussian might be interesting too!
                velocity = np.array([random.uniform(-self.maxInitVelocity,self.maxInitVelocity),
                        random.uniform(-self.maxInitVelocity,self.maxInitVelocity), 
                        random.uniform(-self.maxInitVelocity,self.maxInitVelocity)])
                # accumulate total momentum to correct for in the end
                mv=mv+mass*velocity # accumulate momentum
                self.planets.append(Planet(position,velocity,i,mass))

            # balance overall momentum with correction to the most massive planet (avoids doing so to a small planet, which gets thrown away)
            mostmassive=self.mostMassivePlanetIndex()
            pm = self.planets[mostmassive].mass
            pv = self.planets[mostmassive].v
            self.planets[mostmassive].v = pv-mv/pm

            self.planet_maxid = nplanets
            msg=f"FIATLUX - created {nplanets:d} planets - most massive ={mostmassive:d} - mass={pm:5.2f}"
        else:
            self.testPlanets(-nplanets)
        return self.planet_maxid


    def testPlanets(self, testsel):
        """
        example: define a fixed set of planets eg for tests purposes
        """
        if testsel==2:
            self.msg="TEMPLATE 2 * 5-bodies"
            self.planets=[  Planet(np.array([-100,  50, 150]),np.array([-1, 0  , 1. ]),0,16),
                            Planet(np.array([ 150,-250,-100]),np.array([ 2, 2  , 0. ]),1, 8),
                            Planet(np.array([-200,   0,-200]),np.array([-2, 2  , 1. ]),1, 4),
                            Planet(np.array([ -50,-200, 100]),np.array([ 2,-2  ,-1. ]),1, 4),
                            Planet(np.array([ 100, 400,   0]),np.array([ 0,-1  ,-1. ]),2,16)]
        elif testsel==3:
            self.msg="TEMPLATE 3 * 3-bodies"
            self.planets=[  Planet(np.array([-200,-300,-200]),np.array([ 0, 0  , 0. ]),0,4),
                            Planet(np.array([ 200, 200, 200]),np.array([ 2, 4  , 0. ]),1,2),
                            Planet(np.array([   0, 300,   0]),np.array([ 0,-1  ,-0.5]),2,8)]
        elif testsel==4:
            self.msg="TEMPLATE 4 * 3-bodies with orbit transfer"
            self.planets=[  Planet(np.array([ -50,-100,   0]),np.array([ 2,-1  ,-1. ]),0,4),
                            Planet(np.array([   0,  50,-100]),np.array([-4, 4  , 0. ]),1,1),
                            Planet(np.array([ 100,   0, 100]),np.array([-1, 0  , 1. ]),2,4)]
        elif testsel==5:
            self.msg="TEMPLATE 5 * 3-bodies with large mass"
            self.planets=[  Planet(np.array([   0,   0,   0]),np.array([  0,  0, 0. ]),2,500),
                            Planet(np.array([  50, 100,-200]),np.array([-20, 20,-20.]),1,  2),
                            Planet(np.array([-200,-100,   0]),np.array([ 10,-10, 10.]),0,  4)]
        elif testsel==6:
            self.msg="TEMPLATE 6 * 4-bodies with large mass"
            self.planets=[  Planet(np.array([   0,   0,   0]),np.array([  0,  0, 0. ]),2,500),
                            Planet(np.array([  50, 100,-200]),np.array([-20, 20,-20.]),1,  2),
                            Planet(np.array([-300,-200,   0]),np.array([ 10,-20, 10.]),0,  4),
                            Planet(np.array([-304,-204,   6]),np.array([  0,  0,  0.]),0,  0.099),
                            ]
            self.calcspeed=0.4

        elif testsel==7:
            self.msg="TEMPLATE * 7-bodies quasi stable"
            self.planets=[  Planet(np.array([   0,   0,   0]),np.array([  0,  0, 0. ]),2,500),
                            Planet(np.array([  50, 100,-200]),np.array([-20, 20,-20.]),1,0.2),
                            Planet(np.array([-300,-200,   0]),np.array([ 10,-20, 10.]),0,0.4),
                            Planet(np.array([-304,-204,   6]),np.array([ 40,  0,  0.]),0,0.1),
                            Planet(np.array([ -14, -24, 300]),np.array([ 30,-15,-10.]),0,0.1),
                            Planet(np.array([  45,  97,-205]),np.array([-10, 15,-15.]),0,0.1)
                            ]
            self.calcspeed=1
        else:
            self.msg="TEMPLATE 1 * 3-bodies highly chaotic with final escape"
            self.planets=[  Planet(np.array([-200,-200,-200]),np.array([ 1, 0  , 1. ]),0,4),
                            Planet(np.array([ 200, 200, 200]),np.array([-2, 4  , 0. ]),1,2),
                            Planet(np.array([   0, 200,   0]),np.array([ 0,-1  ,-0.5]),2,8)]


        self.planet_maxid = len(self.planets)-1


    def draw(self):
        """
        draw all planets and their trace    
        """
        self.display.fill((0, 0, 0))
        pygame.draw.line(self.display, (255,255,255), (0,self.screenh),(self.screenw, self.screenh))
        displaymessage = f"Nbodies {len(self.planets):0d} T{self.Time:8.2f} step {self.timestep:5.4f} " + self.msg
        text = self.font.render(displaymessage, True, (255,255,255))
        self.display.blit(text, (0,0))
        for planet in self.planets:
            planet.trace(self.trace,self.screenw,self.screenh)
            planet.draw(self.display,self.screenw,self.screenh)

    def drawHelp(self):
        """
        draw the help overlay
        """
        if self.displayHelp:
            print("drawHelp showing",self.displayHelp)
            self.helpdisp.fill((0,0,0))
            helprect = pygame.Rect(self.screenw/4,self.screenh/2,self.screenw/2,self.screenh)
            pygame.draw.rect(self.helpdisp,(200,200,200),helprect,1)
            pygame.draw.rect(self.helpdisp,(20,20,20),helprect.inflate(-2,-2))
            titlefont=pygame.font.Font(pygame.font.match_font('Arial Bold'), 48)
            textfont=pygame.font.Font(pygame.font.match_font('Arial'), 26)
            title= "GRAVITY Help"
            titlerender = titlefont.render(title, True, (255,255,255))
            w=titlefont.size(title)[0]
            self.helpdisp.blit(titlerender, (helprect.centerx - w/2, helprect.top + 5))
            text = ("?, h - displays this help",
                    "r - initialize planets and start simulation",
                    "1-9 or t - fixed set of planets (for tests)",
                    "a - toggles trace attenuation",
                    "F1,F2 - halves/double calculation step (or speed)",
                    "f, m - starts with few (<=5)or many (up to 100) planets",
                    "q, Esc - quits")
            for i,t in enumerate(text):
                render = textfont.render(t, True, (255,255,255))
                x = helprect.left+10
                y = helprect.top + 20 + (i+1)*textfont.get_linesize()
                self.helpdisp.blit(render, (x, y))
        else:
            #print("drawHelp masking")
            self.helpdisp.fill((0,0,0))
 
            
    def dampenTrace(self):
        """
        this trick makes the traces fade out according to the attenuation factor (alpha channel opacity) defined as constant
        typically 5 to 10 for attenuation factor (out of 256) does a nice job
        this is called from the main loop once every 50 steps or 100... whatever
        """
        tempDisp = self.trace.copy()
        tempDisp.set_alpha(self.traceattenuation)
        tempDisp.fill((0,0,0))
        pygame.draw.line(tempDisp, (255,255,255), (0,self.screenh),(self.screenw, self.screenh))
        self.trace.blit(tempDisp,(0,0))

    def printInfo(self):
        """
        computes and displays total momentum on standard output
        """
        mv=mp=mm=mv2=0.0
        masses=[]
        for planet in self.planets:
            pmv = planet.mass * planet.v
            mv += pmv
            mv2+= pmv * planet.v 
            mp += planet.mass * planet.p
            mm += planet.mass
            masses.append(planet.mass)

        KEnergy = np.sum(mv2)
        GEnergy = self.computegravitypotential()

        print(f"printInfo -{len(self.planets):2d} planets - dt={self.timestep:4.3f} - calcspeed {self.calcspeed:1.1f} ΣMass={np.sum(masses):4.2f} - ΣKE={KEnergy:6.2f} - ΣGE={GEnergy:6.2f} - TE={KEnergy+GEnergy:6.2f} - Σmv = ",mv)
        #        print("              - barycenter =", mp/mm )
        #        print("              - total mv   =",mv )
        #      print("              - masses     =", masses)


    def mostMassivePlanetIndex(self):
        mostmassive=-1
        largestmass=0.0
        for i,planeti in enumerate(self.planets):
            if planeti.mass>=largestmass:
                mostmassive = i
                largestmass = planeti.mass
        return mostmassive

    def computeDistances(self):
        """
        compute distances array, then compute forces
        this need be a high-performance routine
        """
        d2Array=np.zeros((len(self.planets),len(self.planets))) # todo: change 'zeros' to "empty" for speed
        for i,planeti in enumerate(self.planets):
            for j,planetj in enumerate(self.planets):
                dx=planetj.p[0]-planeti.p[0]
                dy=planetj.p[1]-planeti.p[1]
                dz=planetj.p[2]-planeti.p[2]
                d2=dx*dx+dy*dy+dz*dz
                d2Array[j,i] = d2Array[i,j] = d2 # love this python syntax
        return d2Array
        
    def computegravitypotential(self, index=-1):
        mg = 0.0
        if index>=0:
            planet1=self.planets[index]
            #print("cgp - planet ",index," - ",planet1)
            for i,planeti in enumerate(self.planets):
             #   print("cgp loop - planet ",i," - ",planeti)
                if planeti.id != planet1.id:
                    dist = np.linalg.norm(planet1.p - planeti.p)
                    mg -= planeti.mass * planet1.mass * self.G / dist # gravitational potential
            return mg
        else:
            for i,planetj in enumerate(self.planets):
                mg += self.computegravitypotential(i)
            return mg

    def computeforces(self):
        """
        compute distances array, then compute forces
        this need be a high-performance routine
        """
        nplanets = len(self.planets)
        #d2Array=np.zeros((nplanets,nplanets)) # todo: change 'zeros' to "empty" for speed
        dpArray=np.zeros((nplanets,nplanets,3))
        dmin=10000000
        d2coll_i=0
        d2coll_j=0
        self.planetexploding=-1
        for i,planeti in enumerate(self.planets):
            for j,planetj in enumerate(self.planets):
                if (j>i):
                    collisiondist = (planeti.radius+planetj.radius)/self.dCollision
                    dp = planetj.p - planeti.p
                    dist = np.linalg.norm(dp)
                    dpArray[i,j] = dp # WARNING: used to include "dpArray[j,i] =" but half the dpArray needs to be stored 
                    if dist<dmin:
                        dmin=dist
                    if dist<collisiondist:
                        d2coll_i=i
                        d2coll_j=j
                        break
            if d2coll_j:
                break

        if d2coll_j: # collision detected
            #    print("computeforces - collision detected between ",self.planets[d2coll_i].id," and ",self.planets[d2coll_j].id)
            self.planetsmerge(d2coll_i,d2coll_j) # note that indexes are passed rather than objects
            return -1
        else: # no collision, compute forces    
            self.forces=np.zeros((len(self.planets),3))
            for i,planeti in enumerate(self.planets):
                for j,planetj in enumerate(self.planets):
                    if (j>i):
                        # modulus
                        d=np.linalg.norm(dpArray[i,j])
                        fij=planeti.mass*planetj.mass*self.G/d/d
                        # project on axis
                        self.forces[i]=self.forces[i]+fij*dpArray[i,j]/d 
                        self.forces[j]=self.forces[j]-fij*dpArray[i,j]/d
            #   accel=np.linalg.norm(self.forces[i])/planeti.mass
            #    if accel>amax:
            #        amax=accel
            # check tidal force before applying force
            vmax = 0
            invvmax = 1000 # to avoid a NaN
            
            for i,planeti in enumerate(self.planets):
                v=np.linalg.norm(planeti.v)
                if v>vmax: vmax=v
                tidalForce = (np.linalg.norm(self.forces[i]))
                if tidalForce > planeti.tidalLimit:
                    self.planetexploding=i

            # cautiously adjust timestep to new geometry
            if vmax>0:
                invvmax=self.calcspeed/vmax

            newTimeStep = max(min(4,self.calcspeed*dmin/400, invvmax),0.0001)
            if newTimeStep > self.timestep :
                self.timestep=min(newTimeStep, self.timestep*2)
            else:
                self.timestep=newTimeStep
            return 0

    def makestep(self, dt=None):
        """
        core routine dispatches coputation according to choice of integration method
        """
        if dt==None:
            dt=self.timestep
        if self.IntegrationMethod==0:
            # Euler integration
            # amax=self.computeforces()
            if (self.computeforces()>=0): # no collision
                self.applyforces(dt)
                self.applymotion(dt)
        else: # Verlet method
            self.VerletStep(dt)
        if self.planetexploding != -1:
            self.explodeplanet(self.planetexploding)
        self.step = self.step + 1
        self.Time = self.Time + dt

        # new time step function of max speed
        # if amax>0:
        #     #    newTimeStep=max(min(2,4/vmax),0.001)
        #     newTimeStep=max(min(2,2/amax),0.001)
        #     if newTimeStep > self.timestep :
        #         self.timestep=min(newTimeStep, self.timestep*2)
        #     else:
        #         self.timestep=newTimeStep


    def applyforces(self, dt=None):
        """ 
        applies the forces (aka acceleration) over time interval dt to every planet
        note that it is useful that the timestep is not that of self when intermediate calculcations are performed such as in Verlet
        """
        # vmax=0
        if dt==None:
            dt=self.timestep
        for i,planeti in enumerate(self.planets):
            planeti.applyforce(dt, self.forces[i])
        #   vi=np.linalg.norm(planeti.v)
        #   if vmax<vi:
        #       vmax = vi
        return


    def applymotion(self, dt=None):
        """ 
        applies the motion over time interval dt to every planet
        note that it is useful that the timestep is not that of self when intermediate calculcations are performed such as in Verlet
        """
        if dt==None:
            dt=self.timestep
        for planeti in self.planets:
            planeti.applymotion(dt)



    def VerletStep(self,dt=None):
        """
        after //femto-physique/analyse-numerique/methode-de-verlet.php
        the verlet method adjusts for acceleration change on the trajectory over the step
        """
        # v0 serves as speed vector buffer
        if dt==None:
            dt=self.timestep

        v0=np.empty((len(self.planets),3))
        for i,planeti in enumerate(self.planets):
            v0[i]=planeti.v

        # computes forces (ie. accelerations) at original position
        if (self.computeforces()==0):
            # intermediate speed vector: v+dt/2*a
            self.applyforces(dt/2)
            #' next position is computed: p = p + dt.(v+dt/2*a)
            self.applymotion(dt)
            # restore speed vectors to their initial value
            for i,planeti in enumerate(self.planets):
                planeti.v = v0[i]
            # apply half-force from previous point
            self.applyforces(dt/2)
            #' compute force at new position
            if (self.computeforces()==0):
                # apply half force of new position 
                self.applyforces(dt/2)
            else:
                pass #' a collision occured, don't update anything before next step
        else:
            pass #' a collision occured, don't update anything before next step

    def testOutOfBounds(self):
        """
        test if a planet is too far away and can be discarded
        """
        index=-1
        for i,planeti in enumerate(self.planets):
            if ((abs(planeti.p[0])>self.bounds) or (abs(planeti.p[1])>self.bounds) or (abs(planeti.p[2])>self.bounds)):
                index=i 
        if (index!=-1):
            planet=self.planets[index]
            self.msg = f"EJECTION - Planet {planet.id:0d} is lost in space   "
            print(self.msg)
            self.removeplanet(index)

    def removeplanet(self,index):
        """
        remove one planet while preserving overall momentum by dispatching it to the rest so that the barycenter remains immobile
        """
        if ((index>=0) and (index<=len(self.planets))):
            # record momentum of lost planet
            removedpl = self.planets[index] # will remove that planet
            #print("removeplanet - Tmass",totalmass," pr.v=",removedpl.v.count(), " pr.mass=",removedpl.mass)
            mv = removedpl.mass * removedpl.v
            # remove it
            del self.planets[index]

            # dispatch momentum to rest of group
            totalmass=0.0
            for planet in self.planets:
                totalmass += planet.mass
            
            # MiV'i = MiVi + Mi/M't * MnVn (= MtVt = 0) --> V'i = Vi + 1/M't  MnVn 
            for planet in self.planets:
                planet.v = planet.v +  mv / totalmass # total momentum is preserved if sum is null 
        
    def planetsmerge(self,n1,n2):
        """ 
        merge planets while preserving momentum
        planets n1% and n2% are merged while preserving their momentum and not disrupting the rest
        need to secure that changes to tables don't affect merged planet
        so index of merged has to be lower than removed one
        """
        # retrieve planets
        planet1 = self.planets[n1]
        planet2 = self.planets[n2]
        # POSITIONS
        # update position with barycenter of n1% and n2%
        planet1.p = (planet1.mass*planet1.p + planet2.mass*planet2.p) / (planet1.mass + planet2.mass)
        # SPEEDS
        # let's keep the momentum
        # all momentum transfered to n1%
        planet1.v = (planet1.mass*planet1.v + planet2.mass*planet2.v) / (planet1.mass + planet2.mass)
        planet2.v = np.zeros(3)
        planet1.density=min(planet1.density,planet2.density)
        # MASS
        planet1.updatemass(planet1.mass + planet2.mass) # also updates radius and tidal limit
        self.msg = f" COLLISION - merged planet {planet1.id:d} and {planet2.id:d}"
        print(self.msg)
        self.removeplanet(n2)

    def explodeplanet(self,n1):
        """
        explode planets n1 while preserving momentum
        """
        # retrieve planets
        planet1 = self.planets[n1]
        mv0 = planet1.v*planet1.mass # record initial momentum as it will be preserved
        mg0 = self.computegravitypotential(n1) # record gravity potential for planet n1
        mp = np.array([0.,0.,0.])
        mv = np.array([0.,0.,0.])
        mm = 0 # cumulative mass of planetoids
        mass=0 # new mass
        i=0
        print(f"explodeplanet ORIGIN id={planet1.id:d} m={planet1.mass:3.2f} dt={self.timestep:6.5f} tl={planet1.tidalLimit:e} v=",planet1.v," p=",planet1.p)
        
        newplanets=[]
        while mm<planet1.mass:
        # create new planetoids until mass totals parent planet
            while mass<=0: # gaussian dist does not guarantee positive mass
                mass = random.uniform(planet1.mass/50,planet1.mass/20)
            if (mm + mass) > planet1.mass:
                mass = planet1.mass-mm # last iteration
                v = (mv0 - mv) / mass # overall added momentum shall be zero
            else:
                v = planet1.v * np.array([random.gauss(0.8,0.1),random.gauss(0.8,0.1),random.gauss(0.8,0.1)])

            # Let's scatter pieces around the original planet over a radius r in all directions to ensure new bodies are beyond immediate re-collision. 
            # However doing so will create additional gravitational potential energy, which will accumulate in the system unless 
            # we remove it (later) from the velocity...
            
            r = abs(random.gauss(planet1.radius,planet1.radius/3.)) + 6 * planet1.radius 
            theta = random.uniform(0,360)*np.pi/180
            phi=random.uniform(0,360)*np.pi/180
            dp = np.array([r * np.sin(phi)*np.cos(theta), r*np.sin(phi)*np.sin(theta), r*np.cos(phi)]) # x=ρsinφcosθ,y=ρsinφsinθ, and z=ρcosφ
            # p = planet1.p + self.screenw*np.array([random.gauss(0,0.003),random.gauss(0,0.003),random.gauss(0,0.003)])
            p = planet1.p + dp
            mp = mp + mass * p
            mv = mv + mass * v
            mm = mm + mass
            i=i+1
            self.planet_maxid=self.planet_maxid+1
            print(f"explodeplanet piece {i:d} id={self.planet_maxid:d} m={mass:3.2f} v=",v," p=",p)
            newplanets.append(Planet(p,v,self.planet_maxid,mass))
            mass=0

        planet1.v = np.zeros(3)
        self.msg = f" EXPLOSION - planet {planet1.id:d} had a tidal wreakage"
        print(self.msg)
        self.removeplanet(n1)

        index0 = len(self.planets)

        for newplanet in newplanets:
            self.planets.append(newplanet)

        mg1=0.0
        for i,planeti in enumerate(newplanets):
          #  print("explodeplanet ",i,"- delta mg=",mg1-mg0)
            mg1+=self.computegravitypotential(i+index0)

        print("explodeplanet delta mg=",mg1-mg0)

 
class Planet:
    def __init__(self, p = np.zeros(3), v = np.zeros(3), id=0, mass=1.0, temp=100):
    #    print("Planet(): p=",p," v=",v," m=",mass)
        self.p = p
        self.v = v
        self.a = np.zeros(3)
        self.mass = mass
        self.id = id
        self.color = (random.uniform(64,255),random.uniform(64,255),random.uniform(64,255))
        self.density = 1.5
        self.radius = self.getRadius()
        self.tidalConstant = abs(random.gauss(100000,20000))
        self.tidalLimit = self.getTidalLimit() # limit of acceleration before planet breaks in pieces
        self.Temperature = temp # temperature

    def updatemass(self,mass):
        self.mass = mass
        self.radius = self.getRadius()
        self.tidalLmit=self.getTidalLimit()  

    def getTidalLimit(self): # limit of acceleration before planet breaks in pieces
        return self.tidalConstant*self.density/(self.radius) # arbitraty formula

    def getRadius(self): # planet radius from its mass and density
        return (self.mass/self.density)**(1/3)

    def applyforce(self, dt, force=[0.,0.,0.]):
        self.a = force / self.mass
        self.v = self.v + dt * self.a

    def applymotion(self, dt):
        self.p = self.p + dt * self.v


    def draw(self, display, screenw, screenh):
        x=int(screenw/2+self.p[0]) 
        y=int(screenh/2+self.p[1]/2)
        z=int(screenh+screenh/2+self.p[2]/2)
        # print("drawing at ",x,y,z)
        pygame.draw.circle(display, self.color, (x, min(y,screenh)), int(self.radius*5))
        pygame.draw.circle(display, self.color, (x, max(z,screenh)), int(self.radius*5))

    def trace(self, display, screenw,screenh): 
        x=int(screenw/2+self.p[0]) 
        y=int(screenh/2+self.p[1]/2)
        z=int(screenh+screenh/2+self.p[2]/2)
        tracecolor=(self.color[0]*0.7,self.color[1]*0.7,self.color[2]*0.7)
        # print("drawing at ",x,y,z, " with color", tracecolor)
        pygame.draw.line(display, tracecolor, (x, min(y,screenh)),(x, min(y,screenh))) # see https://stackoverflow.com/questions/10354638/pygame-draw-single-pixel
        pygame.draw.line(display, tracecolor, (x, max(z,screenh)),(x, max(z,screenh)))



# -------------------------------------------------------------------------------------------------------------
# okay let's play
#
def main():
    # pygame inits, don't ask me why... none of this was needed owith GFA BASIC!
    pygame.init()
    clock = pygame.time.Clock()
    pygame.display.set_caption("GRAVITY")

    WINDOW_SIZE = (1600, 900)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    screen_rect = screen.get_rect()
    display = pygame.Surface(WINDOW_SIZE)
    display_rect = display.get_rect()

    maxNumberOfPlanets = 5
    attenuatetrace = 1 # whether to dampen the trace of every planet

    drawevery = 10 # display universe every 'drawevery' calculation steps - updated dynamically
    infoevery = 5000 # display universe info every 'invorevery' steps - static
    myUniverse = Universe(display, -1) # starts with example 1


    while True:
        #clock.tick(600)
        #    mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if (event.type == QUIT) or (event.type == KEYDOWN and (event.key == pygame.K_q or event.key==pygame.K_ESCAPE)):
                sys.exit()
                pygame.quit()
            if event.type == KEYDOWN:
                print("Keydown - key=",event.key)

            if event.type == KEYDOWN and event.key == pygame.K_r: # reset
                myUniverse = Universe(display, 1+int(random.uniform(1,maxNumberOfPlanets)))

            if event.type == KEYDOWN and event.key == pygame.K_t: # test planet set
                myUniverse = Universe(display, 0)

            if event.type == KEYDOWN and event.key == pygame.K_a: # toggle trace attenuation
                attenuatetrace = 1 - attenuatetrace

            if event.type == KEYDOWN and event.key == pygame.K_F1: # halve calcspeed
                myUniverse.calcspeed /= 2.0

            if event.type == KEYDOWN and event.key == pygame.K_F2: # double calcspeed
                myUniverse.calcspeed *= 2.0


            if event.type == KEYDOWN and event.key == pygame.K_f: # few planets
                maxNumberOfPlanets = 5
                myUniverse = Universe(display, 1+int(random.uniform(1,maxNumberOfPlanets)))

            if event.type == KEYDOWN and event.key >= pygame.K_1 and event.key <= pygame.K_9: # test setups
                myUniverse = Universe(display, -(event.key - pygame.K_1 + 1))

            if event.type == KEYDOWN and event.key == pygame.K_m: # many planets
                maxNumberOfPlanets = 100
                myUniverse = Universe(display, 1+int(random.uniform(1,maxNumberOfPlanets)))

            if event.type == KEYDOWN and ((event.key == pygame.K_h) or (event.unicode == "?")) :
                myUniverse.displayHelp = 1 - myUniverse.displayHelp # toggle help display
                myUniverse.drawHelp()


        #print("Time",myUniverse.Time)
        myUniverse.makestep()

        if (myUniverse.step % infoevery)==0: # test bounds & display info every 'infoevery' steps
            myUniverse.testOutOfBounds() # will remove planets out of universe
            if attenuatetrace:
                myUniverse.dampenTrace() # darken trace
            myUniverse.printInfo() # report on system in stdout

        if (myUniverse.step % drawevery)==0: # draws every 'drawevery' steps
            np=len(myUniverse.planets)
            if not np: # all planets have escaped; shouldn't happen but may
                sys.exit()
                pygame.quit()
            drawevery = 1 + int(200./(np*np)) # display universe every 'drawevery' calculation steps
            myUniverse.draw()
            screen.blit(myUniverse.trace, (0, 0))
            screen.blit(myUniverse.display, (0, 0))
            screen.blit(myUniverse.helpdisp, (0,0))
            pygame.display.update()
          
        
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()