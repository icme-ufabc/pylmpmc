from abc import ABC,abstractmethod
from mpi4py import MPI
from lammps import lammps

class MonteCarlo(ABC):
    def __init__(self,params={}):  
        if not isinstance(params,dict):
            raise TypeError("'params' must be a Python dictionary!")

        self._comm=MPI.COMM_WORLD                           # MPI communicator
        self._comm.Get_rank()                               # Processor rank in a parallel simulation
        self._lmp=lammps()                                  # LAMMPS object that allows to drive LAMMPS for energy calculations
        self._inistep=None                                  # Initial step
        self._ecurr=None                                    # Energy of the current configuration
        self._nsites=None                                   # Number of sites
        self._interstitials=[]                              # Species of the interstitial atoms, if any	
        self._t=[]                                          # Store the type of the species occupying a given site
        self._boxlo=[]		                                # Store the lowest boundaries of the simulation box
        self._boxhi=[]		                                # Store the highest boundaries of the simulation box   
        self.MAXSUBIMP=7                                    # Define the maximum number of substitutional impurity species
        self.MAXINTIMP=3                                    # Define the maximum number of interstitial impurity species
        self._spec=[[] for i in range(self.MAXSUBIMP+1)]    # Store the sites occupied by an atomic species on the lattice
        self._params={}                                     # Python dictionary containing the MC simulation parameters        

        # Initial parameter values
        self._params['init']='from_scratch'
        self._params['temp']=300.0
        self._params['press']=0.0
        self._params['concentration']=[]
        self._params['nintersatoms']=[]
        self._params['nsteps']=1000000
        self._params['nrunequil']=0
        self._params['production']=False
        self._params['latfile']='lattice.ini'
        self._params['maxdisp']=0.02
        self._params['xcprob']=0.1
        self._params['dispprob']=0.9
        self._params['enesave']=1000
        self._params['ncheckpoint']=1000
        self._params['ndump']=1000
        self._params['xlo']=None
        self._params['ylo']=None
        self._params['zlo']=None
        self._params['xhi']=None
        self._params['yhi']=None
        self._params['zhi']=None
        self._params['xy']=0.0
        self._params['xz']=0.0
        self._params['yz']=0.0
        self._params['lmp_units']='metal'
        self._params['lmp_dimension']=3
        self._params['lmp_atom_style']='atomic'
        self._params['lmp_pair_style']=None
        self._params['lmp_pair_coeff']=None
        self._params['pbcx']=True
        self._params['pbcy']=True
        self._params['pbcz']=True
        self._params['masses']=[]
        
        # Updates parameter values
        self._params.update(params)
        
        # Is the initialization option OK?
        if isinstance(self._params['init'],str):
            self._params['init']=self._params['init'].lower()
            
            if not self._params['init'] in ('from_scratch','restart'):
                raise ValueError("'init' must be either 'from_scratch' or 'restart'!")
        else:
            raise TypeError("'init' must be a string!")
            
        # Is the lattice file name OK?
        if not isinstance(self._params['latfile'],str) and \
            len(self._params['latfile']):
            raise TypeError("'latfile' must be a non-empty string!")

        # Is the temperature OK?
        if not (isinstance(self._params['temp'],(int,float)) and 
                self._params['temp']>0.0):
            raise ValueError("'temp' must be greater than zero!")

        kb=8.6173303e-5		# Boltzmann constant in eV/K
        self._beta=1.0/(kb*self._params['temp'])

        # Is the pressure OK?
        if not (isinstance(self._params['press'],(int,float)) and self._params['press']>=0.0):
            raise ValueError("'press' must be greater than or equal to zero!")

        # Are the concentrations OK?
        if len(self._params['concentration'])==0:
            self._params['concentration'].append(1.0)
        else:
            for conc in self._params['concentration']:
                if not (isinstance(conc,float) and conc>0.0):
                    raise ValueError("Concentration must be greater than zero!")
                
            if sum(self._params["concentration"])>1.0:
                raise RuntimeError("Total concentration cannot be grater than one!")
                
        # Is the number of interstitial impurities OK?
        for nintersatoms in self._params['nintersatoms']:
            if not isinstance(nintersatoms,int) and nintersatoms>=0:
                raise ValueError("Number of interstitial atoms must be greater than or equal to zero!")
                
        # Are the atomic masses OK?
        if len(self._params['masses'])<(len(self._params['concentration'])+
                                        len(self._params['nintersatoms'])):
            raise TypeError("Atomic masses must be provided for all species in the simulation!")
        else:
            for mass in self._params['masses']:
                if not (isinstance(mass,(int,float)) and mass>0.0):
                    raise TypeError("Atomic masses must be greater than zero!")
		
        # Is the number of MC steps OK?
        if not (isinstance(self._params['nsteps'],int) and 
                self._params['nsteps']>0):
            raise ValueError("Number of MC steps must be greater than zero!")

        # Is the number of MD steps for volume equilibration OK?
        if not (isinstance(self._params['nrunequil'],int) and 
                self._params['nrunequil']>=0): 
            raise ValueError("Number of MD steps for volume equilibration must be greater than or equal to zero!")

        # Is the maximum displacement of lattice atoms OK?
        if not (isinstance(self._params['maxdisp'],(int,float)) and
                self._params['maxdisp']>0.0):
            raise ValueError("Maximum atomic displacement must be greater than zero!")
            
        # Is the energy save interval OK?
        if not (isinstance(self._params['enesave'],int) and self._params['enesave']>0):
            ValueError("Energy save interval must be greater than zero!") 

        # Is the checkpoint interval OK?
        if not (isinstance(self._params['ncheckpoint'],int) and self._params['ncheckpoint']>=0):
            raise TypeError("Checkpoint interval must greater than or equal to zero!")

        # Is the dump interval OK?
        if not (isinstance(self._params['ndump'],int) and self._params['ndump']>0):
            raise TypeError("Dump interval must be greater than zero!")        

        # Is the exchange probability OK?
        if len(self._params['concentration'])>0:
            if not (isinstance(self._params['xcprob'],float) and 
                    self._params['xcprob']>=0.0 and self._params['xcprob']<1.0):
                raise ValueError("'xcprob' must be positive and less than one!")

        # Is the displacement probability of interstitial atoms OK?
        if len(self._params['nintersatoms'])>0:
            if not (isinstance(self._params['dispprob'],float) and
                    self._params['dispprob']>=0.0 and self._params['dispprob']<1.0):
                raise ValueError("'dispprob' must be positive and less than one!")

        # Have the system boundaries been correctly set?
        if isinstance(self._params['xlo'],(int,float)) and \
            isinstance(self._params['xhi'],(int,float)) and \
            isinstance(self._params['ylo'],(int,float)) and \
            isinstance(self._params['yhi'],(int,float)) and \
            isinstance(self._params['zlo'],(int,float)) and \
            isinstance(self._params['zhi'],(int,float)):
            if(self._params['xhi']<=self._params['xlo']): 
                raise ValueError("'xhi' must be greater than 'xlo!")
            elif(self._params['yhi']<=self._params['ylo']): 
                raise ValueError("'yhi' must be greater than 'ylo'!")
            elif(self._params['zhi']<=self._params['zlo']): 
                raise ValueError("'zhi' must be greater than 'zlo'!")
        else:
            raise TypeError("System boundaries must be numbers!")
                
        # Have the simulation box tilt factors been correctly set?
        if not (isinstance(self._params['xy'],(int,float)) and
                isinstance(self._params['xz'],(int,float)) and
                isinstance(self._params['yz'],(int,float))):
            raise TypeError("Tilt factors must be numbers!")
        
        # Are LAMMPS header commands OK? AQUI
        if not isinstance(self._params['lmp_units'],str):
            raise TypeError("LAMMPS 'units' must be a string!")
            
        if not isinstance(self._params['lmp_dimension'],int):
            raise TypeError("LAMMPS 'dimension' must be an integer!")
            
        if not isinstance(self._params['lmp_atom_style'],str):
            raise TypeError("LAMMPS 'atom_style' must be a string!")
            
        if not isinstance(self._params['lmp_pair_stye'],str):
            raise TypeError("LAMMPS 'pair_style' must be a string!")
            
        if not isinstance(self._params['lmp_pair_coeff'],(str,list,tuple)):
            raise TypeError("LAMMPS 'pair_coeff' must be a string or a list/tuple of strings!")
        
        # Are the boolean parameters OK?
        if not isinstance(self._params['production'],bool):
            raise TypeError("True or False is expected to define if this is a production run!")
        
        if not isinstance(self._params['pbcx'],bool):
            raise TypeError("True or False is expected to define if the simulation is periodic along X!")
            
        if not isinstance(self._params['pbcy'],bool):
            raise TypeError("True or False is expected to define if the simulation is periodic along Y!")
            
        if not isinstance(self._params['pbcz'],bool):
            raise TypeError("True or False is expected to define if the simulation is periodic along Z!")

    def run(self):
        pass
    
    def _move(self,step,move_type="canonical",*args):
        '''
        Performs a trial move at step 'step'.

        Parameters
        ----------
        step : integer
            Monte Carlo step.
        move_type : string, optional
            Type of trial move. It must be either "canonical" or "non-canonical". 
            The default is "canonical".
        *args : Tuple
            Tuple of arguments to be used for non-canonical trial moves.
        '''
        if move_type.lower()=="canonical":
            self._canonical_move(step)
        elif move_type.lower()=="non-canonical":
            self._non_canonical_move(step,args)
        else:
            raise ValueError("Only 'canonical' and 'non-canonical' are allowed!")

    def _canonical_move(self,step):
        pass
    
    def _non_canonical_move(self,*args):
        pass
    