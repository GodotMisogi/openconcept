from openmdao.api import ImplicitComponent, ExplicitComponent, Group, ExternalCodeComp, IndepVarComp, BalanceComp, ExecComp, IndepVarComp
from openmdao.api import NewtonSolver, BoundsEnforceLS
from openmdao.api import DirectSolver, SqliteRecorder
from openmdao.api import Problem
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp, SumComp
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from openconcept.utilities.dvlabel import DVLabel
import numpy as np
import openmdao.api as om

from examples.StabilityTest import data as acdata

# class Stability_HorizontalTail(Group):

#     def initialize(self):
#         self.options.declare('off-design', default=False)

#     def setup(self, inputs, outputs):
#         # self.add_input('ac|geom|wing|S_ref', units='m**2')
#         # self.add_input('ac|geom|wing|mac', units='m')
#         # self.add_input('l_h', units='m')
#         # self.add_input('V_h', units=None)

#         self.add_output('')
#     def solve_nonlinear(self, inputs, outputs):
#         outputs['ac|geom|hstab|S_ref'] = inputs['V_h'] * inputs['ac|geom|wing|S_ref'] * inputs['ac|geom|wing|mac'] / inputs['l_h']


class Lister(ExplicitComponent):

    def initialize(self):
        self.options.declare('num', default=1, types=int)
        self.options.declare('units', default=None)

    def setup(self):
        nn = self.options['num']
        unit = self.options['units']
        for i in range(nn):
            self.add_input(f'{i}', units=unit)

        self.add_output('list', shape=(nn, 1), units=unit)

    def compute(self, inputs, outputs):
        nn = self.options['num']
        outputs['list'] = [ inputs[f'{i}'] for i in range(nn) ]

class LongitudinalCenterOfGravity(Group):
    """

    """
    def initialize(self):
        self.options.declare('num', default=1, types=int)

    def setup(self):
        nn = self.options['num']

        # TODO: Group formulation
        self.add_subsystem('moments', ElementMultiplyDivideComp(output_name='xWs', input_names=['weights', 'positions'], input_units=['N', 'm'], vec_size=nn, length=1), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('sum_weights', SumComp(output_name='W', input_name='weights', vec_size=nn, length=1, units='N'), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('sum_moments', SumComp(output_name='xW', input_name='xWs', vec_size=nn, length=1, units='N*m'), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('center_of_gravity', ExecComp('x_cg = xW/W', x_cg={'units': 'm'}, xW={'units': 'N*m'}, W={'units': 'N'}), promotes_inputs=['*'], promotes_outputs=['*'])


class AerodynamicPitching(ExplicitComponent):
    
    def setup(self):
        # Aircraft/Global
        self.add_input('ac|cl', units=None,
                        desc='Aircraft lift coefficient.')
        
        # Wing
        self.add_input('wing|S_ref', units='m**2',
                        desc='Wing area')
        self.add_input('wing|coords|x', units='m',
                        desc='Wing AC x-location')
        self.add_input('wing|cm', units=None,
                        desc='Wing pitching moment coefficient')
        self.add_input('ref_chord', units='m',
                        desc='Reference chord length, e.g. MAC, MGC or root chord')

        # Tail
        self.add_input('hstab|coords|x', units='m',
                        desc='Horizontal tail AC x-location')
        self.add_input('hstab|cl', units=None,
                        desc='Horizontal tail lift coefficient')
        self.add_input('hstab|cm', units=None,
                        desc='Horizontal tail pitching moment coefficient')
        self.add_input('hstab|S_ref', units='m**2',
                        desc='Horizontal tail area')

        # Fuselage
        self.add_input('fuselage|cmvf', units=None,
                        desc='Fuselage pitching moment coefficient')

        # Outputs
        self.add_output('ac|cm', units=None)
        self.add_output('x_cp', units='m')

        # Jacobians
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        Sh_Sw = inputs['hstab|S_ref'] / inputs['wing|S_ref']
        clh = inputs['hstab|cl']
        cl = inputs['ac|cl']
        c0 = inputs['ref_chord']
        wing_ac = inputs['wing|coords|x']
        tail_ac = inputs['hstab|coords|x']

        # Moment coefficient
        outputs['ac|cm'] = inputs['wing|cm'] \
                         - wing_ac / c0 * (cl - Sh_Sw * clh) \
                         + inputs['hstab|cm'] \
                         + tail_ac / c0 * Sh_Sw * clh \
                         + inputs['fuselage|cmvf']
        # Center of pressure 
        outputs['x_cp'] = -c0 * (inputs['wing|cm'] \
                         - wing_ac / c0 * (cl - Sh_Sw * clh) \
                         + inputs['hstab|cm'] \
                         + tail_ac / c0 * Sh_Sw * clh \
                         + inputs['fuselage|cmvf']) / cl

    def compute_partials(self, inputs, partials):
        Sh = inputs['hstab|S_ref']
        Sw = inputs['wing|S_ref']
        clh = inputs['hstab|cl']
        cl = inputs['ac|cl']
        c0 = inputs['ref_chord']
        wing_ac = inputs['wing|coords|x']
        tail_ac = inputs['hstab|coords|x']

        # Aircraft Cm derivatives
        partials['ac|cm', 'ac|cl'] = -wing_ac / c0 * (1 - Sh/Sw * clh)
        partials['ac|cm', 'fuselage|cmvf'] = 1
        partials['ac|cm', 'hstab|S_ref'] = - wing_ac / c0 * (cl - 1/Sw * clh) \
                                         + tail_ac / c0 * 1/Sw * clh
        partials['ac|cm', 'hstab|cl'] = - wing_ac / c0 * (cl - Sh/Sw) \
                                      + tail_ac / c0 * Sh/Sw
        partials['ac|cm', 'hstab|cm'] = 1                     
        partials['ac|cm', 'hstab|coords|x'] = 1 / c0 * Sh/Sw * clh
        partials['ac|cm', 'ref_chord'] = (wing_ac * (cl - Sh/Sw * clh) - tail_ac * Sh/Sw * clh) / c0**2
        partials['ac|cm', 'wing|S_ref'] = - 1 / c0 * Sh/Sw**2 * clh * (wing_ac + tail_ac)
        partials['ac|cm', 'wing|cm'] = 1
        partials['ac|cm', 'wing|coords|x'] = -1 / c0 * (cl - Sh/Sw * clh)


        # CoP derivatives
        partials['x_cp', 'ac|cl'] = c0 / cl**2 * (inputs['wing|cm'] \
                                  - wing_ac / c0 * Sh/Sw * clh \
                                  + inputs['hstab|cm'] \
                                  + tail_ac / c0 * Sh/Sw * clh \
                                  + inputs['fuselage|cmvf'])
        partials['x_cp', 'fuselage|cmvf'] = -c0 / cl
        partials['x_cp', 'hstab|S_ref'] = c0 / cl  * (wing_ac / c0 * (cl - 1/Sw * clh) \
                                        - tail_ac / c0 * 1/Sw * clh)
        partials['x_cp', 'hstab|cl'] = c0 / cl * (wing_ac / c0 * (cl - Sh/Sw) \
                                     - tail_ac / c0 * Sh/Sw)
        partials['x_cp', 'hstab|cm'] = -c0 / cl                  
        partials['x_cp', 'hstab|coords|x'] = - 1 / cl * Sh/Sw * clh
        partials['x_cp', 'ref_chord'] = - (inputs['fuselage|cmvf'] + inputs['wing|cm'] + inputs['hstab|cm']) / cl
        partials['x_cp', 'wing|S_ref'] = 1 / cl * Sh/Sw**2 * clh * (wing_ac + tail_ac)
        partials['x_cp', 'wing|cm'] = -c0 / cl
        partials['x_cp', 'wing|coords|x'] = 1 / cl * (cl - Sh/Sw * clh)

class NeutralPoint(ExplicitComponent):
    
    def setup(self):
        # Aircraft/Global
        # self.add_input('ac|cl', units=None,
        #                 desc='Aircraft lift coefficient.')
        
        # Wing
        self.add_input('wing|S_ref', units='m**2',
                        desc='Wing area')
        self.add_input('wing|coords|x', units='m',
                        desc='Wing AC x-location')
        # self.add_input('wing|cm', units=None,
                        # desc='Wing pitching moment coefficient')
        # self.add_input('ref_chord', units='m',
                        # desc='Reference chord length, e.g. MAC, MGC or root chord')

        # Tail
        self.add_input('hstab|coords|x', units='m',
                        desc='Horizontal tail AC x-location')
        # self.add_input('hstab|cl', units=None,
        #                 desc='Horizontal tail lift coefficient')
        # self.add_input('hstab|cm', units=None,
        #                 desc='Horizontal tail pitching moment coefficient')
        self.add_input('hstab|S_ref', units='m**2',
                        desc='Horizontal tail area')

        # Fuselage
        self.add_input('fuselage|cmvf', units=None,
                        desc='Fuselage pitching moment coefficient')
        self.add_input('dCLh_dCL', units=None)

        self.add_output('x_np', units='m')

        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        dCLh_dCL = inputs['dCLh_dCL']
        cmvf = inputs['fuselage|cmvf']
        tail_ac = inputs['hstab|coords|x']
        Sh = inputs['hstab|S_ref']
        wing_ac = inputs['wing|coords|x']
        Sw = inputs['wing|S_ref']
        cmvf = inputs['fuselage|cmvf']
        # c0 = inputs['ref_chord']

        outputs['x_np'] = wing_ac * (1 - Sh/Sw * dCLh_dCL) \
                        + tail_ac * Sh/Sw * dCLh_dCL \
                        - cmvf / Sw

    def compute_partials(self, inputs, partials):
        dCLh_dCL = inputs['dCLh_dCL']
        cmvf = inputs['fuselage|cmvf']
        tail_ac = inputs['hstab|coords|x']
        Sh = inputs['hstab|S_ref']
        wing_ac = inputs['wing|coords|x']
        Sw = inputs['wing|S_ref']
        
        partials['x_np', 'dCLh_dCL'] = Sh/Sw * (tail_ac - wing_ac) 
        partials['x_np', 'fuselage|cmvf'] = -1/Sw
        partials['x_np', 'hstab|coords|x'] = Sh/Sw * dCLh_dCL
        partials['x_np', 'hstab|S_ref'] = 1/Sw * dCLh_dCL * (tail_ac - wing_ac)
        partials['x_np', 'wing|coords|x'] = (1 - Sh/Sw * dCLh_dCL)
        partials['x_np', 'wing|S_ref'] = Sh/Sw**2 * dCLh_dCL * (wing_ac - tail_ac) + cmvf / Sw**2


class dCLhdCL(ExplicitComponent):

    def setup(self):
        self.add_input('mach', units=None)
        self.add_input('wing|AR', units=None)
        self.add_input('wing|c4sweep', units='deg')
        self.add_input('hstab|AR', units=None)
        self.add_input('hstab|c4sweep', units='deg')
        self.add_input('dEps_dAlpha', units=None)
        
        self.add_output('dCLh_dCL', units=None)

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        dEps_dAlpha = inputs['dEps_dAlpha']
        beta = (1 - inputs['mach']**2)**0.5
        wing_AR = inputs['wing|AR']
        tail_AR = inputs['hstab|AR']
        la_w = inputs['wing|c4sweep']
        la_ht = inputs['hstab|c4sweep']

        outputs['dCLh_dCL'] = (beta + 2/wing_AR) / (beta + 2/tail_AR) * ((beta**2 + np.tan(la_w)**2)/(beta**2 + np.tan(la_ht)**2))**0.5 * (1 - dEps_dAlpha)

    # def compute_partials(self, inputs, partials):
    #     dEps_dAlpha = inputs['dEps_dAlpha']
    #     beta = (1 - inputs['mach']**2)**0.5
    #     wing_AR = inputs['wing|AR']
    #     tail_AR = inputs['hstab|AR']
    #     la_w = inputs['wing|c4sweep']
    #     la_ht = inputs['hstab|c4sweep']

    #     partials['dCLh_dCL', 'mach'] = 

class Weights(Group):

    def setup(self):
        ac_model = self.add_subsystem('ac_model', DictIndepVarComp(acdata, separator='|'), promotes_outputs=["*"])
        ac_model.add_output_from_dict('ac|geom|wing|weight')
        ac_model.add_output_from_dict('ac|geom|hstab|weight')
        ac_model.add_output_from_dict('ac|geom|vstab|weight')
        ac_model.add_output_from_dict('ac|geom|fuselage|weight')

        self.connect('ac|geom|wing|weight', 'wts.0')
        self.connect('ac|geom|hstab|weight', 'wts.1')
        self.connect('ac|geom|fuselage|weight', 'wts.2')

class PitchTrim(Group):

    def initialize(self):
        self.options.declare('num', default=1, types=int)
        self.options.declare('off_design', default=False)

    def setup(self):
        nn = self.options['num']
        design = not(self.options['off_design'])

        # Subsystems
        self.add_subsystem('center_of_pressure', AerodynamicPitching(), 
                            promotes_inputs=['*'], 
                            promotes_outputs=['*'])

        self.add_subsystem('center_of_gravity', LongitudinalCenterOfGravity(num=nn), 
                            promotes_inputs=['*'], 
                            promotes_outputs=['*'])


        # Balance component 
        bal = self.add_subsystem('solve', BalanceComp(), promotes=['*'])
        
        # Solve for horizontal tail area and wing position
        if design:
            bal.add_balance('hstab|S_ref', units='m**2', eq_units='m')
            bal.add_balance('wing|coords|x', units='m', eq_units='m')

            # Set LHS and RHS
            self.connect('x_cg', ['lhs:hstab|S_ref', 'lhs:wing|coords|x'])
            self.connect('x_cp', ['rhs:hstab|S_ref', 'rhs:wing|coords|x'])
        # Solve for horizontal tail lift coefficient
        else:
            bal.add_balance('hstab|cl', units=None, eq_units='m')

            # Set LHS and RHS
            self.connect('x_cg', 'lhs:hstab|cl')
            self.connect('x_cp', 'rhs:hstab|cl')

class PitchStability(Group):

    def initialize(self):
        self.options.declare('num', default=1, types=int)

    def setup(self):
        nn = self.options['num']

        ac_model = self.add_subsystem('ac_model', DictIndepVarComp(acdata, separator='|'), promotes_outputs=["*"])
        ac_model.add_output_from_dict('ac|stability|static_margin')
        ac_model.add_output_from_dict('ac|geom|ref_chord')

        # Subsystems
        self.add_subsystem('center_of_gravity', LongitudinalCenterOfGravity(num=nn), 
                            promotes_inputs=['*'],  
                            promotes_outputs=['*'])

        self.add_subsystem('hstab_cl_gradient', dCLhdCL(), 
                            promotes_inputs=['*'], 
                            promotes_outputs=['*'])
        
        self.add_subsystem('neutral_point', NeutralPoint(), 
                            promotes_inputs=['*'], 
                            promotes_outputs=['*'])

        self.add_subsystem('sm_lhs', ExecComp('dx = x_np - x_cg', x_cg={'units': 'm'}, x_np={'units': 'm'}))

        self.add_subsystem('sm_rhs', ExecComp('fsm_mac = static_margin * mac', static_margin={'units': None}, mac={'units': 'm'}))
        
        # Balance component 
        bal = self.add_subsystem('solve', BalanceComp(), promotes=['*'])
        bal.add_balance('hstab|S_ref', units='m**2')
        bal.add_balance('wing|coords|x', units='m')

        # Connections
        self.connect('ac|stability|static_margin', 'sm_rhs.static_margin')
        self.connect('ac|geom|ref_chord', 'sm_rhs.mac')
        self.connect('x_np', 'sm_lhs.x_np')
        self.connect('x_cg', 'sm_lhs.x_cg')
        # self.connect('solve.hstab|S_ref', 'hstab|S_ref')
        # self.connect('solve.wing|coords|x', 'wing|coords|x')

        # Set LHS and RHS
        self.connect('sm_lhs.dx', ['lhs:hstab|S_ref', 'lhs:wing|coords|x', ])
        self.connect('sm_rhs.fsm_mac', ['rhs:hstab|S_ref', 'rhs:wing|coords|x'])


class StabilityDesignHorizontalTail(Group):

    def initialize(self):
        self.options.declare('num', default=1, types=int)

    def setup(self):
        nn = self.options['num']

        # Define a bunch of design variables and airplane-specific parameters
        ac_model = self.add_subsystem('ac_model', DictIndepVarComp(acdata, separator='|'), promotes_outputs=["*"])
        ac_model.add_output_from_dict('ac|aero|cl')
        ac_model.add_output_from_dict('ac|geom|ref_chord')
        ac_model.add_output_from_dict('ac|stability|static_margin')

        ac_model.add_output_from_dict('ac|geom|wing|S_ref')
        ac_model.add_output_from_dict('ac|geom|wing|cm')
        ac_model.add_output_from_dict('ac|geom|wing|AR')
        ac_model.add_output_from_dict('ac|geom|wing|coords|x')
        ac_model.add_output_from_dict('ac|geom|wing|c4sweep')

        ac_model.add_output_from_dict('ac|geom|hstab|S_ref')
        ac_model.add_output_from_dict('ac|geom|hstab|AR')
        ac_model.add_output_from_dict('ac|geom|hstab|cl')
        ac_model.add_output_from_dict('ac|geom|hstab|cm')
        ac_model.add_output_from_dict('ac|geom|hstab|coords|x')
        ac_model.add_output_from_dict('ac|geom|hstab|c4sweep')

        ac_model.add_output_from_dict('ac|geom|fuselage|cmvf')

        self.add_subsystem('acmodel',self.options['aircraft_model'](num_nodes=nn, flight_phase=self.options['flight_phase']),promotes_inputs=['*'],promotes_outputs=['*'])

        indeps = self.add_subsystem('const', IndepVarComp(), promotes_outputs=['*'])
        indeps.add_output('dEps_dAlpha', val=0.01, units=None)
        indeps.add_output('mach', val=0.8)
        indeps.add_output('weights', val=[2000., 1000., 500])
        indeps.add_output('positions', val=[0.5, -0.25, 25])


        # Cycle subsystem
        # cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
        # cycle.add_subsystem('trim', PitchTrim(num=nn, off_design=False), 
                            # promotes_inputs=['*'])
            
        self.add_subsystem('stability', PitchStability(num=nn), 
                            promotes_inputs=['*'])
        # cycle.connect('')
        # cycle.connect('trim.solve.hstab|S_ref', 'stability.solve.hstab|S_ref')
        # cycle.connect('trim.solve.wing|coords|x', 'stability.solve.wing|coords|x')


        # self.connect('ac|aero|cl', 'ac|cl')
        # self.connect('ac|geom|ref_chord', 'ref_chord')

        self.connect('ac|geom|wing|S_ref', 'wing|S_ref')
        # self.connect('ac|geom|wing|cm', 'wing|cm')
        self.connect('ac|geom|wing|AR', 'wing|AR')
        self.connect('ac|geom|wing|c4sweep', 'wing|c4sweep')

        self.connect('ac|geom|hstab|AR', 'hstab|AR')
        # self.connect('ac|geom|hstab|cl', 'hstab|cl')
        # self.connect('ac|geom|hstab|cm', 'hstab|cm')
        self.connect('ac|geom|hstab|coords|x', 'hstab|coords|x')
        self.connect('ac|geom|hstab|c4sweep', 'hstab|c4sweep')

        self.connect('ac|geom|fuselage|cmvf', 'fuselage|cmvf')


prob = Problem()
prob.model = StabilityDesignHorizontalTail(num=3)

prob.model.nonlinear_solver = NewtonSolver()
prob.model.options['assembled_jac_type'] = 'csc'
prob.model.linear_solver = DirectSolver(assemble_jac=True)
prob.model.nonlinear_solver.options['solve_subsystems'] = True
prob.model.nonlinear_solver.options['maxiter'] = 10
prob.model.nonlinear_solver.options['atol'] = 1e-6
prob.model.nonlinear_solver.options['rtol'] = 1e-6
prob.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='scalar', print_bound_enforce=True)
prob.model.set_solver_print()
prob.setup(check=True, mode='fwd')

# N2 generation
om.n2(prob, 'stability.html', show_browser=False)

prob['hstab|S_ref'] = 5.0
prob['wing|coords|x'] = 0.25


prob.check_partials(compact_print=True)
prob.model.list_inputs(prom_name=True)
prob.model.list_outputs(prom_name=True)

print(prob['xW'])
print(prob['hstab|S_ref'])
print(prob['wing|coords|x'])

# prob.set_val('climb.fltcond|vs', np.ones((num_nodes,))*1500, units='ft/min')

prob.run_model()

print(prob['hstab|S_ref'])
print(prob['wing|coords|x'])
    