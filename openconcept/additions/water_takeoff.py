from openmdao.api import ImplicitComponent, ExplicitComponent, Group, ExternalCodeComp, IndepVarComp, BalanceComp, ExecComp, IndepVarComp, KrigingSurrogate, MetaModelUnStructuredComp
from openmdao.api import NewtonSolver, BoundsEnforceLS
from openmdao.api import DirectSolver, SqliteRecorder, ArmijoGoldsteinLS
from openmdao.api import Problem
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp, SumComp
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from openconcept.utilities.dvlabel import DVLabel
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.analysis.aerodynamics import Lift, StallSpeed
from openconcept.utilities.math import ElementMultiplyDivideComp, AddSubtractComp
from openconcept.utilities.math.integrals import Integrator
from openconcept.utilities.linearinterp import LinearInterpolator
import numpy as np
import openmdao.api as om

class Groundspeeds(ExplicitComponent):
    """
    Computes groundspeed for vectorial true airspeed and true vertical speed.

    This is a helper function for the main mission analysis routines
    and shouldn't be instantiated directly.

    Inputs
    ------
    fltcond|vs : float
        Vertical speed for all mission phases (vector, m/s)
    fltcond|Utrue : float
        True airspeed for all mission phases (vector, m/s)

    Outputs
    -------
    fltcond|groundspeed : float
        True groundspeed for all mission phases (vector, m/s)
    fltcond|cosgamma : float
        Cosine of the flght path angle for all mission phases (vector, dimensionless)
    fltcond|singamma : float
        Sine of the flight path angle for all mission phases (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of points to run
    """
    def initialize(self):

        self.options.declare('num_nodes',default=1,desc="Number of Simpson intervals to use per seg (eg. climb, cruise, descend). Number of analysis points is 2N+1")

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('fltcond|vs', units='m/s',shape=(nn,))
        self.add_input('fltcond|Utrue', units='m/s',shape=(nn,))
        self.add_output('fltcond|groundspeed', units='m/s',shape=(nn,))
        self.add_output('fltcond|cosgamma', shape=(nn,), desc='Cosine of the flight path angle')
        self.add_output('fltcond|singamma', shape=(nn,), desc='sin of the flight path angle' )
        self.declare_partials(['fltcond|groundspeed','fltcond|cosgamma','fltcond|singamma'], ['fltcond|vs','fltcond|Utrue'], rows=range(nn), cols=range(nn))

    def compute(self, inputs, outputs):

        nn = self.options['num_nodes']
        #compute the groundspeed on climb and desc
        inside = inputs['fltcond|Utrue']**2-inputs['fltcond|vs']**2
        groundspeed =  np.sqrt(inside)
        groundspeed_fixed = np.sqrt(np.where(np.less(inside, 0.0), 0.01, inside))
        #groundspeed =  np.sqrt(inputs['fltcond|Utrue']**2-inputs['fltcond|vs']**2)
        #groundspeed_fixed= np.where(np.isnan(groundspeed),0,groundspeed)
        outputs['fltcond|groundspeed'] = groundspeed_fixed
        outputs['fltcond|singamma'] = np.where(np.isnan(groundspeed),1,inputs['fltcond|vs'] / inputs['fltcond|Utrue'])
        outputs['fltcond|cosgamma'] = groundspeed_fixed / inputs['fltcond|Utrue']

    def compute_partials(self, inputs, J):
        inside = inputs['fltcond|Utrue']**2-inputs['fltcond|vs']**2
        groundspeed =  np.sqrt(inside)
        groundspeed_fixed = np.sqrt(np.where(np.less(inside, 0.0), 0.01, inside))
        J['fltcond|groundspeed','fltcond|vs'] = np.where(np.isnan(groundspeed),0,(1/2) / groundspeed_fixed * (-2) * inputs['fltcond|vs'])
        J['fltcond|groundspeed','fltcond|Utrue'] = np.where(np.isnan(groundspeed),0, (1/2) / groundspeed_fixed * 2 * inputs['fltcond|Utrue'])
        J['fltcond|singamma','fltcond|vs'] = np.where(np.isnan(groundspeed), 0, 1 / inputs['fltcond|Utrue'])
        J['fltcond|singamma','fltcond|Utrue'] = np.where(np.isnan(groundspeed), 0, - inputs['fltcond|vs'] / inputs['fltcond|Utrue'] ** 2)
        J['fltcond|cosgamma','fltcond|vs'] = J['fltcond|groundspeed','fltcond|vs'] / inputs['fltcond|Utrue']
        J['fltcond|cosgamma','fltcond|Utrue'] = (J['fltcond|groundspeed','fltcond|Utrue'] * inputs['fltcond|Utrue'] - groundspeed_fixed) / inputs['fltcond|Utrue']**2

def hullModel(hull_datapath):
    # Surrogate input format: [C_vr, C_r]
    # , [C_vt, Trim] 
    # cvr, cr, cvt, trim = [], [], [], []
    training = []
    with open(hull_datapath) as resistance:
        for row in resistance:
            row = row.split(',')
            if (row[0] != '') or (row[1] != '') or (row[2] != '') or (row[3] != ''):
                # cvr.append(float(row[0]))
                # cr.append(float(row[1]))
                # cvt.append(float(row[2]))
                # trim.append(float(row[3]))
                training.append(list(map(float, row)))
                
    # training = np.transpose([cvr, cvt, cr, trim])

    return training

hull_datapath = '/home/godot/Academia/Amphy/Aircraft Data/TN2481 - Planing Hull.csv'

hull_data = hullModel(hull_datapath)

class HullSurrogate(MetaModelUnStructuredComp):

    def initialize(self):
        self.options.declare('default_surrogate', default=KrigingSurrogate())
        self.options.declare('vec_size', default=1)
        self.options.declare('dataset', default=np.zeros((4,1)))
        self.options.declare('train:CVR')
        self.options.declare('train:CVT')
        self.options.declare('train:CR')
        self.options.declare('train:alphat')
        self.options.declare('num_nodes',default=1)

    def setup(self):
        nn = self.options['num_nodes']
        dataset = self.options['dataset']
        self.add_input('CVR', shape=(nn,), val=[0]*nn, training_data=[ x[0] for x in dataset ])
        self.add_input('CVT', shape=(nn,), val=[0]*nn, training_data=[ x[2] for x in dataset ])
        self.add_output('CR', shape=(nn,), val=[0]*nn, training_data=[ x[1] for x in dataset ])
        self.add_output('alphat', shape=(nn,), val=[0]*nn, training_data=[ x[3] for x in dataset ])

class VelocityCoefficient(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes',default=1)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('fltcond|Utrue',  shape=(nn,), val=[0]*nn, units='m/s')
        self.add_input('B', val=0., units='m')

        self.add_output('CVR', shape=(nn,))
        self.add_output('CVT', shape=(nn,))

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        g = 9.81
        outputs['CVR'] = inputs['fltcond|Utrue']/(g * inputs['B'])**0.5
        outputs['CVT'] = inputs['fltcond|Utrue']/(g * inputs['B'])**0.5

    def compute_partials(self, inputs, partials):
        g = 9.81
        partials['CVR', 'fltcond|Utrue'] = 1./(g * inputs['B'])**0.5
        partials['CVR', 'B'] = - 0.5 * g * inputs['fltcond|Utrue']/(g * inputs['B'])**1.5
        partials['CVT', 'fltcond|Utrue'] = 1./(g * inputs['B'])**0.5
        partials['CVT', 'B'] = - 0.5 * g * inputs['fltcond|Utrue']/(g * inputs['B'])**1.5

class Resistance(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes',default=1)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('CR', shape=(nn,), val=[0]*nn)
        self.add_input('B', units='m')

        self.add_output('resistance', shape=(nn,), units='N')

        self.declare_partials('*', '*')
    def compute(self, inputs, outputs):
        g = 9.81
        rho_w = 998.2
        outputs['resistance'] = rho_w * g * inputs['B']**3 * inputs['CR']

    def compute_partials(self, inputs, partials):
        g = 9.81
        rho_w = 998.2
        partials['resistance', 'B'] = 3 * rho_w * g * inputs['B']**2 * inputs['CR'] 
        partials['resistance', 'CR'] = rho_w * g * inputs['B']**3 

class HorizontalAcceleration(ExplicitComponent):
    """
    Computes acceleration during takeoff run and effectively forms the T-D residual.

    Inputs
    ------
    weight : float
        Aircraft weight (scalar, kg)
    drag : float
        Aircraft drag at each analysis point (vector, N)
    lift : float
        Aircraft lift at each analysis point (vector, N)
    thrust : float
        Thrust at each TO analysis point (vector, N)
    fltcond|singamma : float
        The sine of the flight path angle gamma (vector, dimensionless)
    braking : float
        Effective rolling friction multiplier at each point (vector, dimensionless)

    Outputs
    -------
    accel_horiz : float
        Aircraft horizontal acceleration (vector, m/s**2)

    Options
    -------
    num_nodes : int
        Number of analysis points to run
    """
    def initialize(self):
        self.options.declare('num_nodes',default=1)

    def setup(self):
        nn = self.options['num_nodes']
        g = 9.80665 #m/s^2
        self.add_input('weight', units='kg', shape=(nn,))
        self.add_input('drag', units='N',shape=(nn,))
        self.add_input('thrust', units='N',shape=(nn,))

        self.add_output('accel_horiz', units='m/s**2', shape=(nn,))
        arange=np.arange(nn)
        self.declare_partials(['accel_horiz'], ['weight','drag','thrust'], rows=arange, cols=arange)


    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        g = 9.80665 #m/s^2
        m = inputs['weight']
        accel = inputs['thrust']/m - inputs['drag']/m
        outputs['accel_horiz'] = accel

    def compute_partials(self, inputs, J):
        g = 9.80665 #m/s^2
        m = inputs['weight']
        J['accel_horiz','thrust'] = 1/m
        J['accel_horiz','drag'] = -1/m
        J['accel_horiz','weight'] = (inputs['drag']-inputs['thrust'])/m**2

class HullResistance(Group):

    def setup(self):
        nn = 81
        indeps = self.add_subsystem('const', IndepVarComp(), promotes=['*'])
        indeps.add_output('thrust', shape=(nn,), val=[10000]*nn, units='N')
        indeps.add_output('weight', shape=(nn,), val=[5600]*nn, units='kg')
        indeps.add_output('fltcond|Utrue_initial', val=2., units='m/s')
        indeps.add_output('dt', val=0.5, units='s')
        self.add_subsystem('vel_coeff', VelocityCoefficient(num_nodes=nn), promotes=['*'])
        self.add_subsystem('surrogate', HullSurrogate(dataset=hull_data, num_nodes=nn), promotes=['*'])
        self.add_subsystem('res', Resistance(num_nodes=nn), promotes=['*'])
        self.add_subsystem('haccel',HorizontalAcceleration(num_nodes=nn), promotes_inputs=[('drag', 'resistance'), '*'],promotes_outputs=['*'])
        nn_simpson = int((nn-1)/2)
        self.add_subsystem('intvelocity',Integrator(num_intervals=nn_simpson, 
                                                    method='simpson', 
                                                    quantity_units='m/s', diff_units='s', 
                                                    time_setup='dt', lower=1.5),
                                                    promotes_inputs=[('dqdt','accel_horiz'),'dt',('q_initial','fltcond|Utrue_initial')],
                                                    promotes_outputs=[('q','fltcond|Utrue'),('q_final','fltcond|Utrue_final')])

# Define a Trajectory object
# traj = dm.Trajectory()
# p.model.add_subsystem('traj', subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription
# phase = dm.Phase(ode_class=BrachistochroneODE,
#                  transcription=dm.GaussLobatto(num_segments=10, order=3))

# traj.add_phase(name='phase0', phase=phase)

prob = om.Problem()

prob.model = HullResistance()

# prob.model.nonlinear_solver = NewtonSolver()
# prob.model.options['assembled_jac_type'] = 'csc'
# prob.model.linear_solver = DirectSolver(assemble_jac=True)
# prob.model.nonlinear_solver.options['solve_subsystems'] = True
# prob.model.nonlinear_solver.options['maxiter'] = 10
# prob.model.nonlinear_solver.options['atol'] = 1e-6
# prob.model.nonlinear_solver.options['rtol'] = 1e-6
# prob.model.set_solver_print()

prob.model.add_recorder(om.SqliteRecorder('hull_takeoff_solver_solution.sql'))

indeps = prob.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
# indeps.add_output('fltcond|Utrue', val=0., units='m/s')
indeps.add_output('B', val=0., units='m')

prob.setup()

# N2 generation
om.n2(prob, 'hull_takeoff.html', show_browser=False)

prob['fltcond|Utrue'] = 8.
prob['B'] = 1.71

# prob.check_partials(compact_print=True)

# prob.set_val('climb.fltcond|vs', np.ones((num_nodes_nodes,))*1500, units='ft/min')

prob.run_model()
cr = om.CaseReader('hull_takeoff_solver_solution.sql')
final_case = cr.get_case(-1)
# final_case.list_inputs(print_arrays=True)
final_case.list_outputs(print_arrays=True)

