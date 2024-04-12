from __future__ import print_function
from . import MonteCarlo
from random import random,randrange,uniform
from math import exp,isnan,inf
from shutil import copy
from time import time
from mpi4py import MPI
from lammps import lammps

class CanonicalMonteCarlo(MonteCarlo):
    def __init__(self):
        
        super().__init__()
            
    def run(self):
        self._lmp.command("print '-----------------------------------------------' file mc.log")
        self._lmp.command("print '        LAMMPS-powered NVT Monte Carlo         ' append mc.log")
        self._lmp.command("print '       developed by Prof. Roberto Gomes        ' append mc.log")
        self._lmp.command("print '   at the Federal University of ABC, Brazil    ' append mc.log")
        self._lmp.command("print '-----------------------------------------------' append mc.log")
        self._lmp.command("print '' append mc.log")
        self._lmp.command("print '*** MC simulation started! ***' append mc.log")
        self._lmp.command("print '' append mc.log")

        self._comm.barrier()	# Force synchronization
		
        self.params=self._comm.bcast(self.params,root=0)     # Broadcast dictionary values to all processors

        self._lmp.command("print '   Temperature: %f K.' append mc.log" % self.params['temp'])

        if self.me==0:  	# I/O tasks are performed by the first processor only				
            x=[]			# Cartesian positions (x1,x2,x3) of the sites on the lattice

            '''
			The lists containing the data of the system are populated
			with the data extracted from the lattice file
			'''
            try:	# Open (or try to open) the file containing the lattice
                f=open(self.params['latfile'],'r')
            except:
                raise FileNotFoundError("File '%s' not found!" % self.params['latfile'])

            for line in f:	# Loop over the lines of the lattice file, gathering the lattice data
                l=line.split()
			
                if len(l)>=3:
                    try:
                        x.append([float(l[0]),float(l[1]),float(l[2])])
                        self._t.append(1)
                    except:
                        raise TypeError("Wrong data types in '%s'!" % self.params['latfile'])
                else: 
                    raise IOError("Wrong number of columns in '%s'!" % (self.params['latfile']))

            f.close()

            self._nsites=len(self._t)

            '''
			In the following lines, the atoms of type 1 and atoms of other species are
			separated into distinct lists containing the indexes of the sites that
			they occupy
			'''
            if self.params['init']=='from_scratch':	# MC simulation starts from scratch
                self._inistep=0
                self.naccepted=0

                '''
				In a production run, the atomic species occupying each site are
				read from a file saved in the last iteration of an equilibration
				run
				'''
                if self.params['production']:
                    try:
                        f=open('last_equil.dat','r')
                    except:
                        raise FileNotFoundError("File 'last_equil.dat' not found!")

                    l=f.readline().split()	# Read the file header with information from the equilibration run

                    if len(l)>=3:
                        try:
                            self._ecurr=float(l[1])	# Energy of the last configuration in the equilibration run
                        except:
                            raise TypeError("Wrong data types at 'last_equil.dat' header!")
                    else:
                        raise IOError("Invalid number of columns at 'last_equil.dat' header!")

                    '''
					Read the atomic type of the sites from last_equil.dat and assign
					the site index to the appropriate list
					'''
                    l=f.readline().split()

                    if len(l)>=self._nsites:
                        for i in range(self._nsites):
                            try:
                                v=int(l[i])
                            except:
                                raise TypeError("Site types in 'last_equil.dat' must be integers!")
                            else:
                                if v>self.params['nspecsubs']+1: 
                                    raise ValueError("Site types in 'last_equil.dat' must be between 1 and "+str(self.params['nspecsubs']+1)+"!")

                            self._spec[v-1].append(i+1)
                            self._t[i]=v				
                    else:
                        raise IOError("Invalid number of sites represented in 'last_equil.dat'!")

                    f.close()

                    '''
					Read the information of the minimum energy configuration
					so far
					'''
                    try:
                        f=open('minconf.dat','r')
                    except:
                        raise FileNotFoundError("File 'minconf.dat' not found!")

                    l=f.readline().split()

                    if len(l)>=3:
                        try:
                            self.emin=float(l[1])
                        except:
                            raise TypeError("Wrong data types at 'minconf.dat' header!")
                    else:
                        raise IOError("Invalid number of columns at 'minconf.dat' header!")
                else:	# In an equilibration run, select randomly the indices to be occupied
                    for i in range(self._nsites):
                        self._spec[0].append(i+1)
					
                    for k in range(self.params['nspecsubs']):
                        for i in range(round(self._nsites*self.params['concentration'][k+1])):
                            j=randrange(0,len(self._spec[0]))
                            v=self._spec[0].pop(j)
                            self._spec[k+1].append(v)
                            self._t[self._spec[k+1][i]-1]=k+2
            elif self.params['init']=='restart':		# Restart a previous MC simulation
                try:
                    f=open('checkpoint.dat','r')
                except:
                    raise FileNotFoundError("File 'checkpoint.dat' not found!")

                '''
				First, read the first line of 'checkpoint.dat', which
				contains the last saved step of the previous run as well
				as its current total energy and number of accepted trial
				moves 
				'''		
                l=f.readline().split()

                if len(l)>=3:
                    try:
                        self._inistep=int(l[0])+1
                        self._ecurr=float(l[1])
                        self.naccepted=int(l[2])
                    except:
                        raise TypeError("Wrong data types at 'checkpoint.dat' header!")
                else:
                    raise IOError("Invalid number of columns at 'checkpoint.dat' header!")

                '''
				Read the atomic type of the sites from 'checkpoint.dat' and assign
				the site index to the appropriate list
				'''
                l=f.readline().split()

                if len(l)>=self._nsites:
                    for i in range(self._nsites):
                        try:
                            v=int(l[i])
                        except:
                            raise TypeError("Site types in 'checkpoint.dat' must be integers!")
                        else:
                            if v>self.params['nspecsubs']+1: 
                                raise ValueError("Site types in 'checkpoint.dat' must be between 1 and "+str(self.params['nspecsubs']+1)+"!")

                        self._spec[v-1].append(i+1)
                        self._t[i]=v		
                else:
                    raise IOError("Invalid number of sites represented in 'checkpoint.dat'!")

                f.close()

                '''
				Read the information of the minimum energy configuration
				so far
				'''
                try:
                    f=open('minconf.dat','r')
                except:
                    raise FileNotFoundError("File 'minconf.dat' not found!")

                l=f.readline().split()

                if len(l)>=3:
                    try:
                        self.emin=float(l[1])
                    except:
                        raise TypeError("Wrong data types at 'minconf.dat' header!")
                else:
                    raise IOError("Invalid number of columns at 'minconf.dat' header!")	
			
                f.close()
				
            '''
			Create the initial simulation box
			'''
            if self.params['production']==False and self.params['init']=='from_scratch':
                f=open('simbox.read_data','w')

                f.write("# Initial coordinate file provided to LAMMPS\n\n")
                f.write("%d atoms\n" % self._nsites)
				
                if self.params['nintersatoms']==0:
                    f.write("%d atom types\n" % (self.params['nspecsubs']+1))
                else:
                    f.write("%d atom types\n" % (self.params['nspecsubs']+2))

                f.write("\n%f %f xlo xhi\n" % (self.params['xlo'],self.params['xhi']))
                f.write("%f %f ylo yhi\n" % (self.params['ylo'],self.params['yhi']))
                f.write("%f %f zlo zhi\n" % (self.params['zlo'],self.params['zhi']))
                
                if self.params['xy']!=0.0 or self.params['xz']!=0.0 or self.params['yz']!=0.0:
                    f.write("\n%f %f %f xy xz yz\n" % (self.params['xy'],self.params['xz'],self.params['yz']))
                    
                if len(self.params['masses'])>0:
                    f.write("\nMasses\n\n")
                    
                    for i in range(self.params['masses']):
                        f.write("%d %f\n" % (i+1,self.params['masses'][i]))
                
                f.write("\nAtoms\n\n")

                for i in range(self._nsites):
                    f.write("%d %d %f %f %f\n" % (i+1,self._t[i],x[i][0],x[i][1],x[i][2]))

                f.close()		

        self._comm.barrier()	# Force synchronization

        '''
		Broadcast variables to all processors in the MPI pool
		'''
        self._inistep=self._comm.bcast(self._inistep,root=0)
        self._ecurr=self._comm.bcast(self._ecurr,root=0)
        self._nsites=self._comm.bcast(self._nsites,root=0)
        self.interstitial=self._comm.bcast(self.interstitial,root=0)

        '''
		Begin: Header of the input passed to the LAMMPS library
		'''
        self._lmp.command("atom_modify map hash")	# This must be set in order to use the scatter_atoms() function later
        self._lmp.command("dimension %d" % self.params['lmp_dimension'])
        self._lmp.command("units %s " % self.params['lmp_units'])
        self._lmp.command("atom_style %s" % self.params['lmp_atom_style'])
        
        boundary="boundary "
        
        if self.params['pbcx']:
            boundary+="p "
        else:
            boundary+="s "
            
        if self.params['pbcy']:
            boundary+="p "
        else:
            boundary+="s "
            
        if self.params['pbcz']:
            boundary+="p "
        else:
            boundary+="s "
        
        self._lmp.command(boundary)
        self._lmp.command("read_data simbox.read_data")
        self._lmp.command("pair_style %s" % self.params['lmp_pair_style'])
        
        if isinstance(self.params['lmp_pair_coeff'],(list,tuple)):
            for i in range(len(self.params['lmp_pair_coeff'])):
               self._lmp.command("pair_coeff %s" % self.params['lmp_pair_coeff'][i]) 
        else:
            self._lmp.command("pair_coeff %s" % self.params['lmp_pair_coeff'])
        '''
        End: Header of the input passed to the LAMMPS library
        '''

        if self.me==0:		
            if self.params['nintersatoms']>0:
                for i in range(self.params['nintersatoms']):
                    self._t.append(self.interstitial)

        if self.params['init']=='restart': 
            self._lmp.command("print '   Restarting the MC simulation from step %d.' append mc.log" % self._inistep)

        self._comm.barrier()     # Force synchronization

        self._t=self._comm.bcast(self._t,root=0)
        self._spec=self._comm.bcast(self._spec,root=0)

        '''
        Perform Monte Carlo iterations
        '''        
        for i in range(self._inistep,self.params['nsteps']):
            self.__move(i)
            self._comm.barrier()
        
        '''
        Finish the Monte Carlo simulation
        '''
        self._lmp.command("print '*** MC simulation finished! ***' append mc.log")
        self._lmp.close()
        
        self._comm.barrier()

        MPI.Finalize()
		
    def __move(self,step):
        r=None		# Initialize the variables involved in the 
        gamma=None	# acceptance condition to be broadcasted to the
        de=None		# MPI processors when running this function

        minconf=False	# Flag that defines whether this iteration has the minimum energy	

        tfirst=None	# Type of the first site in an exchange trial move
        tsecond=None	# Type of the second site in an exchange trial move
        first=None	# Store the LAMMPS atomic indices that the types of which
        second=None	# will be exchanged during an exchange trial move

        sel=None	# Store the index and the displacement of the atom
        dx=None		# that will be displaced during a displacement trial move
        dy=None
        dz=None

        trialprob=None	# Probability of choosing either a displacement or an exchange trial move
        dispprob=None	# Probability of choosing a slight atomic displacement or moving an interstitial atom to a new position

        xinternew=[]	# New position of the selected interstitial atom

        self._lmp.command("print '=> Step %d is running...' append mc.log" % step)
	
        if step>0 or self.params['production']:
            if self.me==0:	# Trial move selection is performed by the first processor only
                trialprob=random()	

                if self.params['nspecsubs']>0 and trialprob<self.params['xcprob']:	# Perform an exchange trial move
                    sellength=[]
                    chosen=False
                    totlength=0.0
                    r=random()

                    for i in range(self.params['nspecsubs']+1):	# Select the type of the first atom
                        length=totlength+self.params['concentration'][i]

                        if not chosen and r<length:
                            sellength.append(0.0)
                            tfirst=i
                            chosen=True
                        else:
                            sellength.append(length)
                            totlength=length		

                    old=randrange(0,len(self._spec[tfirst]))				
                    first=self._spec[tfirst][old]

                    r=uniform(0.0,totlength)
					
                    for i in range(self.params['nspecsubs']+1):	# Select the type of the other atom
                        if r<sellength[i]:
                            tsecond=i

                            break

                    new=randrange(0,len(self._spec[tsecond]))
                    second=self._spec[tsecond][new]
                else:	# Perform a displacement trial move
                    dispprob=random()

                    if self.params['nintersatoms']==0 or dispprob<self.params['dispprob']:	# Slightly displace an atom from its current position
                        sel=randrange(0,self._nsites+self.params['nintersatoms'])
                        dx=uniform(-self.params['maxdisp'],self.params['maxdisp'])
                        dy=uniform(-self.params['maxdisp'],self.params['maxdisp'])
                        dz=uniform(-self.params['maxdisp'],self.params['maxdisp'])
                    else:	# Move an interstitial atom to a random new position in the simulation box
                        sel=randrange(self._nsites,self.params['nintersatoms']+self._nsites)
                        xinternew.append(uniform(self.boxlo[0],self.boxhi[0]))
                        xinternew.append(uniform(self.boxlo[1],self.boxhi[1]))
                        xinternew.append(uniform(self.boxlo[2],self.boxhi[2]))

            self._comm.barrier()	# Force synchronization

            trialprob=self._comm.bcast(trialprob,root=0)	# Broadcast the trial probability to all processors
            dispprob=self._comm.bcast(dispprob,root=0)	# Broadcast the displacement probability to all processors

            if self.params['nspecsubs']>0 and trialprob<self.params['xcprob']:	# Perform an exchange trial move
                first=self._comm.bcast(first,root=0)
                second=self._comm.bcast(second,root=0)
                tfirst=self._comm.bcast(tfirst,root=0)
                tsecond=self._comm.bcast(tsecond,root=0)

                self._lmp.command("print '   Trial move: site %d -> %d, site %d -> %d' append mc.log" % (first,tsecond+1,second,tfirst+1))
                self._lmp.command("set atom %d type %d" % (first,tsecond+1))
                self._lmp.command("set atom %d type %d" % (second,tfirst+1))
            else:	# Perform a displacement trial move
                xold=self._lmp.gather_atoms("x",1,3)     # Get the atomic coordinates and store it into the old configuration
                x=self._lmp.gather_atoms("x",1,3)    	# Get the atomic coordinates and store it into the current, transient configuration

                self._comm.barrier()	# Force synchronization

                sel=self._comm.bcast(sel,root=0)

                if self.params['nintersatoms']==0 or dispprob<self.params['dispprob']:	# Slightly displace an atom from its current position
                    dx=self._comm.bcast(dx,root=0)
                    dy=self._comm.bcast(dy,root=0)
                    dz=self._comm.bcast(dz,root=0)

                    self._comm.barrier()	# Force synchronization

                    self._t=self._comm.bcast(self._t,root=0)

                    x[sel*3]+=dx
                    x[sel*3+1]+=dy
                    x[sel*3+2]+=dz			

                    self._lmp.command("print '   Trial move: atom %d, type %d, displaced by (%f,%f,%f)' append mc.log" % (sel+1,self._t[sel],dx,dy,dz))
                else:	# Move an interstitial site to a new position inside the simulation box
                    self._comm.barrier()	# Force synchronization

                    xinternew=self._comm.bcast(xinternew,root=0)

                    x[sel*3]=xinternew[0]
                    x[sel*3+1]=xinternew[1]
                    x[sel*3+2]=xinternew[2]

                    self._lmp.command("print '   Trial move: atom %d, interstitial, moved to (%f,%f,%f)' append mc.log" % (sel+1,xinternew[0],xinternew[1],xinternew[2]))

                self._lmp.scatter_atoms("x",1,3,x)								
        else:	# First MC step
            if self.params['nrunequil']>0:	# Relax the simulation box volume and positions with an initial MD run
                self._lmp.command("minimize 0 1e-4 5000 10000")
                self._lmp.command("reset_timestep 0")
                self._lmp.command("velocity all create %f %d dist gaussian" % (self.params['temp']*2,int(time())))
                
                if self.params['pbcx'] and self.params['pbcy'] and self.params['pbcz']:
                    self._lmp.command("fix 1 all npt temp %f %f $(100*dt) aniso %f %f $(1000*dt) drag 2.0" 
                                     % (self.params['temp'],self.params['temp'],self.params['press'],self.params['press']))                
                else:
                    pdef=""
                    
                    if self.params['pbcx']:
                        pdef=pdef+"x %f %f $(1000*dt) " % (self.params['press'],
                                                           self.params['press'])
                        
                    if self.params['pbcy']:
                        pdef=pdef+"y %f %f $(1000*dt) " % (self.params['press'],
                                                           self.params['press'])
                    if self.params['pbcz']:
                        pdef=pdef+"z %f %f $(1000*dt) " % (self.params['press'],
                                                           self.params['press'])
                        
                    if pdef=="":
                        self._lmp.command("fix 1 all nvt temp %f %f $(100*dt) drag 2.0" 
                                         % (self.params['temp'],self.params['temp']))             # Particle (0-dimension)
                    else:
                        self._lmp.command("fix 1 all npt temp %f %f $(100*dt) %s drag 2.0" 
                                         % (self.params['temp'],self.params['temp'],pdef))    # System with at least one non-periodic dimension

                self._lmp.command("run %d" % self.params['nrunequil'])
                self._lmp.command("unfix 1")
                self._lmp.command("reset_timestep 0")

                '''
				Assign the new values of box boundaries
				'''
                self.boxlo[0]=self._lmp.extract_global("boxxlo",1)
                self.boxlo[1]=self._lmp.extract_global("boxylo",1)
                self.boxlo[2]=self._lmp.extract_global("boxzlo",1)
                self.boxhi[0]=self._lmp.extract_global("boxxhi",1)
                self.boxhi[1]=self._lmp.extract_global("boxyhi",1)
                self.boxhi[2]=self._lmp.extract_global("boxzhi",1)

            if self.params['nintersatoms']>0:   # Create interstitial atoms at random positions
                self._lmp.command("create_atoms %d random %d %d NULL" % (self.interstitial,self.params['nintersatoms'],int(time())))
                self._lmp.command("group noninters type < %d" % self.interstitial)
                self._lmp.command("fix frozen noninters setforce 0 0 0")
                self._lmp.command("minimize 0 1e-4 5000 10000")
                self._lmp.command("unfix frozen")

            x=self._lmp.gather_atoms("x",1,3)        # Get the atomic coordinates and store it into the current, transient configuration

            if self.params['nrunequil']==0:	# Perturb atomic positions randomly
                for i in range(self._nsites+self.params['nintersatoms']):
                    if self.me==0:
                        dx=uniform(-self.params['maxdisp'],self.params['maxdisp'])
                        dy=uniform(-self.params['maxdisp'],self.params['maxdisp'])
                        dz=uniform(-self.params['maxdisp'],self.params['maxdisp'])

                    self._comm.barrier()	# Force synchronization

                    dx=self._comm.bcast(dx,root=0)
                    dy=self._comm.bcast(dy,root=0)
                    dz=self._comm.bcast(dz,root=0)

                    x[i*3]+=dx
                    x[i*3+1]+=dy
                    x[i*3+2]+=dz

                self._lmp.scatter_atoms("x",1,3,x) 

        self._lmp.command("variable e equal etotal")
        self._lmp.command("thermo_modify lost ignore")
        self._lmp.command("run 0")

        totatoms=int(self._lmp.get_natoms())

        if totatoms<(self._nsites+self.params['nintersatoms']):	
            e=inf
        else:
            e=float(self._lmp.extract_variable("e","all",0))

            if isnan(e):
                e=inf

        '''
		Apply the acceptance condition to decide whether the trial move
		will be accepted or not
		'''
        if step>0 or self.params['production']:
            if self.me==0:	# Serial computations are performed by the first processor only
                de=e-self._ecurr
                r=random()
                
                if de<=0.0:
                    gamma=1.0
                else:
                    gamma=exp(-self._beta*de)

            self._comm.barrier()	# Force synchronization

            de=self._comm.bcast(de,root=0)
            r=self._comm.bcast(r,root=0)
            gamma=self._comm.bcast(gamma,root=0)

            if r<gamma:
                self._lmp.command("print '   Energy variation: %f eV; trial move accepted.' append mc.log" % de)

                if self.me==0:	# I/O tasks are performed by the first processor only
                    if self.params['nspecsubs']>0 and trialprob<self.params['xcprob']:
                        '''
						Remove the entries from the corresponding lists, and then
						save the site indices in buffer variables
						'''
                        buffer1=self._spec[tsecond].pop(new)
                        buffer2=self._spec[tfirst].pop(old) 

                        '''
						Update permanently the class attributes changed during the trial move
						'''
                        self._t[buffer1-1]=tfirst+1
                        self._t[buffer2-1]=tsecond+1
                        self._spec[tsecond].append(buffer2)
                        self._spec[tfirst].append(buffer1)

                    self._ecurr=e
                    self.naccepted+=1

                    if e<self.emin:
                        self.emin=e
                        minconf=True

                self._comm.barrier()	# Force synchronization

                minconf=self._comm.bcast(minconf,root=0)
                self._ecurr=self._comm.bcast(self._ecurr,root=0)
            else:	# Undo the trial move
                self._lmp.command("print '   Energy variation: %f eV; trial move rejected.' append mc.log" % de)
                self._lmp.command("print '   Undoing the trial move...' append mc.log")

                if self.params['nspecsubs']>0 and trialprob<self.params['xcprob']:
                    self._lmp.command("set atom %d type %d" % (first,tfirst+1))
                    self._lmp.command("set atom %d type %d" % (second,tsecond+1))
                else:		
                    self._lmp.scatter_atoms("x",1,3,xold)
        else:	# What if this is the first MC step of the equilibration run?
            if self.me==0:
                self._ecurr=e
                self.emin=e
                minconf=True

            minconf=self._comm.bcast(minconf,root=0)
            self._ecurr=self._comm.bcast(self._ecurr,root=0)

            if totatoms<(self._nsites+self.params['nintersatoms']):
                self._lmp.scatter_atoms("x",1,3,x)

        self._lmp.command("reset_timestep 0")

        '''
		Write information about the minimum energy configuration
		so far
		'''
        if minconf:
            if self.me==0:	# I/O tasks are performed by the first processor only
                try:
                    copy("minconf.dat","minconf.tmp")
                except:
                    print("File 'minconf.dat' not found!")

                f=open('minconf.dat','w')
                f.write("%d %f %d\n" % (step,self.emin,self.naccepted))

                for i in range(self._nsites):
                    f.write("%d " % self._t[i])

                f.write("\n")
                f.close()

            self._lmp.command("write_data minconf.read_data")

        '''
		Save the current energy to energies.dat and the dump files
		'''
        if self.params['production']:
            if (step+1)%self.params['enesave']==0:	# Save the energy into energies.dat
                self._lmp.command("print '%d %f' append energies.dat screen no" % (step,self._ecurr))

            if self.params['ndump']>0:	# Save dump files during production runs
                if (step+1)%self.params['ndump']==0:	
                    self._lmp.command("write_dump all custom dump.%d.lammpstrj id type x y z" % step)

            if step==self.params['nsteps']-1:	# Write the last configuration in LAMMPS Data format
                self._lmp.command("write_data lastconf.read_data")
        else: # Save the current energy to energ_equil.dat and the initial configuration to iniconf.read_data
            if (step+1)%self.params['enesave']==0:
                self._lmp.command("print '%d %f' append energ_equil.dat screen no" % (step,self._ecurr))

            if step==0: 
                self._lmp.command("write_data iniconf.read_data")

        '''
		Write checkpoint at prescribed iterations
		'''
        if (step+1)%self.params['ncheckpoint']==0:
            '''
			Save the checkpoint coordinates
			'''
            self._lmp.command("write_data simbox.read_data")

            if self.me==0:	# I/O tasks are performed by the first processor only
                if (step+1)/self.params['ncheckpoint']>=2: 
                    try:
                        copy("checkpoint.dat","checkpoint.tmp")
                    except:
                        print("File 'checkpoint.dat' not found!")

                f=open('checkpoint.dat','w')
                f.write("%d %f %d\n" % (step,self._ecurr,self.naccepted))

                for i in range(self._nsites):
                    f.write("%d " % self._t[i])

                f.write("\n")
                f.close()

        '''
		Save the last accepted configuration in the equilibration run to be 
		later used in the production run
		'''
        if (not self.params['production']) and step==self.params['nsteps']-1:
            if self.me==0:	# I/O tasks are performed by the fist processor only
                f=open('last_equil.dat','w')
                f.write("%d %f %d\n" % (step,self._ecurr,self.naccepted))

                for i in range(self._nsites):
                    f.write("%d " % self._t[i])

                f.write("\n")
                f.close()

            self._lmp.command("write_data simbox.read_data")

        self._comm.barrier()