from openmdao.api import ImplicitComponent, ExplicitComponent, Group, ExternalCodeComp, IndepVarComp, BalanceComp, ExecComp, IndepVarComp
from openmdao.api import NewtonSolver, BoundsEnforceLS
from openmdao.api import DirectSolver, SqliteRecorder, ArmijoGoldsteinLS
from openmdao.api import Problem
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp, SumComp
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from openconcept.utilities.dvlabel import DVLabel
import numpy as np
import openmdao.api as om

from wing_model import Wing, HorizontalStab
from examples.StabilityTest import data as acdata


class Lister(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('units', default=None)
        self.options.declare('name')

    def setup(self):
        nn = self.options['num_nodes']
        unit = self.options['units']
        name = self.options['name']
        for i in range(nn):
            self.add_input(f'{name}_{i}', units=unit)

        self.add_output(f'{name}', shape=(nn, 1), units=unit)

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        name = self.options['name']
        outputs[f'{name}'] = [ inputs[f'{name}_{i}'] for i in range(nn) ]

# class LongitudinalCenterOfGravity(ExplicitComponent):

#     def initialize(self):
#         self.options.declare('num_nodes_nodes', default=1, types=int)
#         self.options.declare('r_payload', default=1, types=float)
#         self.options.declare('r_fuel', default=1, types=float)

#     def setup(self):

#         self.add_input('payload|coords|x', units='m')
#         self.add_input('wing|coords|x', units='m')
#         self.add_input('hstab|coords|x', units='m')
#         self.add_input('vstab|coords|x', units='m')
#         self.add_input('fuselage|coords|x', units='m')

#         self.add_input('weight|payload', units='N')
#         self.add_input('weight|wing', units='N')
#         self.add_input('weight|hstab', units='N')
#         self.add_input('weight|vstab', units='N')
#         self.add_input('weight|fuselage', units='N')
#         z
#         self.add_output('x_cg')

#         self.declare_partials('*', '*')

    # def compute(self, inputs, outputs):
    #     components = ['wing', 'hstab', 'vstab', 'fuselage']
    #     weights = [ self.options[f'r_{name}']*inputs[f'weight|{name}'] if name == 'payload' or name == 'fuel' else inputs[f'weight|{name}'] for name in components ]
    #     positions = [ inputs[f'{name}|coords|x'] for name in components ]

    #     outputs['x_cg'] = sum([ x*W for (x,W) in zip(positions, weights) ])/sum(weights)

    # def compute_partials(self, inputs, partials):
    #     components = ['wing', 'hstab', 'fuselage']
    #     weights = [ inputs[f'weight|{name}'] for name in components ]
    #     positions = [ inputs[f'{name}|coords|x'] for name in components ]
    #     sum_weights = sum(weights)

    #     for name in components:
    #         partials['x_cg', f'{name}|coords|x'] = self.options[f'r_{name}']*inputs[f'{name}|coords|x']/sum_weights if name == 'payload' or name == 'fuel' else inputs[f'{name}|coords|x']/sum_weights
    #         partials['x_cg', f'weight|{name}'] = 1



class LongitudinalCenterOfGravity(Group):
    """

    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # TODO: Group formulation
        self.add_subsystem('moments', 
                            ElementMultiplyDivideComp(output_name='xWs', input_names=['weights', 'positions'], input_units=['N', 'm'], vec_size=nn, length=1), 
                            promotes=['*'])
        self.add_subsystem('sum_weights', 
                            SumComp(output_name='W', input_name='weights', vec_size=nn, length=1, units='N'), 
                            promotes=['*'])
        self.add_subsystem('sum_moments', 
                            SumComp(output_name='xW', input_name='xWs', vec_size=nn, length=1, units='N*m'), 
                            promotes=['*'])
        self.add_subsystem('center_of_gravity', 
                            ExecComp('x_cg = xW/W', x_cg={'units': 'm'}, xW={'units': 'N*m'}, W={'units': 'N'}), 
                            promotes=['*'])


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
        # Wing
        self.add_input('wing|S_ref', units='m**2',
                        desc='Wing area')
        self.add_input('wing|coords|x', units='m',
                        desc='Wing AC x-location')

        # Tail
        self.add_input('hstab|coords|x', units='m',
                        desc='Horizontal tail AC x-location')
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

class PitchTrim(Group):

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('off_design', default=False)

    def setup(self):
        nn = self.options['num_nodes']
        design = not(self.options['off_design'])

        # Subsystems
        self.add_subsystem('center_of_pressure', AerodynamicPitching(), 
                            promotes_inputs=['*'], 
                            promotes_outputs=['*'])
       
        # Compute trim value for solving for horizontal tail area and wing position
        if design:
            self.add_subsystem('trim', ExecComp('trim_eq = x_cg - x_cp', x_cg={'units': 'm'}, x_cp={'units': 'm'}), promotes=['*'])

        # Solve for horizontal tail lift coefficient
        else:

            self.add_subsystem('center_of_gravity', LongitudinalCenterOfGravity(num_nodes=nn), 
                                promotes_inputs=['*'], 
                                promotes_outputs=['*'])

            # Balance component 
            bal = self.add_subsystem('solve', BalanceComp(), promotes=['*'])
            bal.add_balance('hstab|cl', units=None, eq_units='m')

            # Set LHS and RHS
            self.connect('x_cg', 'lhs:hstab|cl')
            self.connect('x_cp', 'rhs:hstab|cl')

class PitchStability(Group):

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options

    def setup(self):
        nn = self.options['num_nodes']

        ac_model = self.add_subsystem('ac_model', DictIndepVarComp(acdata, separator='|'), promotes_outputs=["*"])
        ac_model.add_output_from_dict('ac|stability|static_margin')
        ac_model.add_output_from_dict('ac|geom|ref_chord')

        # Subsystems
        # self.add_subsystem('center_of_gravity', LongitudinalCenterOfGravity(num_nodes=nn), 
        #                     promotes_inputs=['*'],  
        #                     promotes_outputs=['*'])

        self.add_subsystem('dCLh_dCL', dCLhdCL(), 
                            promotes_inputs=['*'], 
                            promotes_outputs=['*'])
        
        self.add_subsystem('neutral_point', NeutralPoint(), 
                            promotes_inputs=['*'], 
                            promotes_outputs=['*'])


        self.add_subsystem('sm_eq', ExecComp('sm_eq = x_cg - x_np + static_margin * mac', x_cg={'units': 'm'}, x_np={'units': 'm'}, static_margin={'units': None}, mac={'units': 'm'}), promotes=['*'])

        # Connections
        self.connect('ac|stability|static_margin', 'static_margin')
        self.connect('ac|geom|ref_chord', 'mac')



class StabilityOffDesignHorizontalTail(Group):

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('static_margin', default=0.25, types=float)
        

    def setup(self):
        nn = self.options['num_nodes']

        # Define a bunch of design variables and airplane-specific parameters
        dv_comp = self.add_subsystem('dv_comp', DictIndepVarComp(acdata, separator='|'), promotes_outputs=["*"])
        dv_comp.add_output_from_dict('ac|aero|cl')
        dv_comp.add_output_from_dict('ac|geom|ref_chord')

        # dv_comp.add_output_from_dict('ac|geom|wing|S_ref')
        dv_comp.add_output_from_dict('ac|geom|wing|cm')
        dv_comp.add_output_from_dict('ac|geom|wing|coords|x')

        # dv_comp.add_output_from_dict('ac|geom|hstab|S_ref')
        dv_comp.add_output_from_dict('ac|geom|hstab|cm')

        dv_comp.add_output_from_dict('ac|weights|fuselage')
        dv_comp.add_output_from_dict('ac|geom|fuselage|cmvf')

        wing = self.add_subsystem('wing', Wing())

        hstab = self.add_subsystem('hstab', HorizontalStab())

        weights = ['hstab.weight', 'ac|weights|fuselage']

        positions = ['ac|geom|hstab|coords|x', 'ac|geom|fuselage|coords|x']

        [ dv_comp.add_output_from_dict(f'{name}') for name in positions ]

        self.add_subsystem('weights', Lister(num_nodes=nn, units='N', name='weights'), promotes=['*'])

        self.add_subsystem('positions', Lister(num_nodes=nn, units='m', name='positions'), promotes=['*'])

        for (i, name) in enumerate(weights):
            self.connect(f'{name}', f'weights_{i}')

        for (i, name) in enumerate(positions):
            self.connect(f'{name}', f'positions_{i}')

        self.add_subsystem('trim', PitchTrim(num_nodes=nn, off_design=True), 
                            promotes_inputs=['*'])

        self.connect('ac|aero|cl', 'ac|cl')
        self.connect('ac|geom|ref_chord', 'ref_chord')


        self.connect('wing.S_ref', 'wing|S_ref')
        # self.connect('wing.AR', 'wing|AR')
        # self.connect('wing.sweep', 'wing|c4sweep')
        self.connect('wing.weight', f'weights_{nn-1}')
        # self.connect('wing|coords|x', f'positions_{nn-1}')

        self.connect('hstab.S_ref', 'hstab|S_ref')
        # self.connect('hstab.sweep', 'hstab|c4sweep')

        # self.connect('ac|geom|wing|S_ref', 'wing|S_ref')
        self.connect('ac|geom|wing|cm', 'wing|cm')
        self.connect('ac|geom|wing|coords|x', ['wing|coords|x', f'positions_{nn-1}']) 
        
        # self.connect('ac|geom|hstab|S_ref', 'hstab|S_ref')
        self.connect('ac|geom|hstab|cm', 'hstab|cm')
        self.connect('ac|geom|hstab|coords|x', ['hstab|coords|x'])


        self.connect('ac|geom|fuselage|cmvf', 'fuselage|cmvf')

        self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True)

class StabilityDesignHorizontalTail(Group):

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('aircraft_model',default=None)

    def setup(self):
        nn = self.options['num_nodes']

        ac_model = self.add_subsystem('ac_model', DictIndepVarComp(acdata, separator='|'), promotes_outputs=["*"])

        ac_model.add_output_from_dict('ac|geom|wing|S_ref')
        ac_model.add_output_from_dict('ac|geom|wing|AR')
        ac_model.add_output_from_dict('ac|geom|wing|c4sweep')

        ac_model.add_output_from_dict('ac|geom|hstab|AR')
        # ac_model.add_output_from_dict('ac|geom|hstab|coords|x')
        ac_model.add_output_from_dict('ac|geom|hstab|c4sweep')

        ac_model.add_output_from_dict('ac|geom|fuselage|cmvf')

        ac_model.add_output_from_dict('ac|aero|cl')
        ac_model.add_output_from_dict('ac|geom|ref_chord')
        ac_model.add_output_from_dict('ac|geom|wing|cm')
        ac_model.add_output_from_dict('ac|geom|hstab|cl')
        ac_model.add_output_from_dict('ac|geom|hstab|cm')
        
        ac_model.add_output_from_dict('ac|weights|fuselage')

        wing = self.add_subsystem('wing', Wing())

        hstab = self.add_subsystem('hstab', HorizontalStab())
        
        weights = ['hstab.weight', 'ac|weights|fuselage']

        positions = ['ac|geom|hstab|coords|x', 'ac|geom|fuselage|coords|x']

        [ ac_model.add_output_from_dict(f'{name}') for name in positions ]
        
        self.add_subsystem('weights', Lister(num_nodes=nn, units='N', name='weights'), promotes=['*'])

        self.add_subsystem('positions', Lister(num_nodes=nn, units='m', name='positions'), promotes=['*'])

        indeps = self.add_subsystem('const', IndepVarComp(), promotes_outputs=['*'])
        indeps.add_output('dEps_dAlpha', val=0.01, units=None)
        indeps.add_output('mach', val=0.8)

        self.add_subsystem('center_of_gravity', LongitudinalCenterOfGravity(num_nodes=nn), 
                            promotes_inputs=['*'], 
                            promotes_outputs=['*'])

        cycle = self.add_subsystem('cycle', om.Group(), promotes=['*']) 
        cycle.add_subsystem('trim', PitchTrim(num_nodes=nn, off_design=False), promotes_inputs=['*'], promotes_outputs=['trim_eq'])
        cycle.add_subsystem('stability', PitchStability(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['sm_eq'])

        # Balance component
        bal = self.add_subsystem('solve', BalanceComp(), promotes=['*'])
        bal.add_balance('hstab|S_ref', units='m**2', rhs_val=0.0)
        bal.add_balance('wing|coords|x', units='m', rhs_val=0.0)

        for (i, name) in enumerate(weights):
            self.connect(f'{name}', [ f'weights_{i}'])

        for (i, name) in enumerate(positions):
            self.connect(f'{name}', [ f'positions_{i}'])

        # OpenMDAO automatically couples on the group level
        self.connect('trim_eq', 'lhs:wing|coords|x')
        self.connect('sm_eq', 'lhs:hstab|S_ref')   

        self.connect('wing.S_ref', 'wing|S_ref')
        self.connect('wing.AR', 'wing|AR')
        self.connect('wing.sweep', 'wing|c4sweep')
        self.connect('wing.weight', f'weights_{nn-1}')
        self.connect('wing|coords|x', f'positions_{nn-1}')

        self.connect('hstab.AR', 'hstab|AR')
        self.connect('hstab.sweep', 'hstab|c4sweep')
        self.connect('ac|geom|hstab|coords|x', 'hstab|coords|x')

        self.connect('ac|geom|fuselage|cmvf', 'fuselage|cmvf')

        self.connect('ac|aero|cl', 'ac|cl')
        self.connect('ac|geom|ref_chord', 'ref_chord')
        self.connect('ac|geom|wing|cm', 'wing|cm')
        self.connect('ac|geom|hstab|cl', 'hstab|cl')
        self.connect('ac|geom|hstab|cm', 'hstab|cm')

        self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True)

def trim():
    prob = Problem()
    prob.model = StabilityOffDesignHorizontalTail(num_nodes=3)

    prob.model.nonlinear_solver = NewtonSolver()
    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 10
    prob.model.nonlinear_solver.options['atol'] = 1e-6
    prob.model.nonlinear_solver.options['rtol'] = 1e-6
    prob.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='scalar', print_bound_enforce=True)
    prob.model.set_solver_print()

    prob.model.add_recorder(om.SqliteRecorder('trim_solver_solution.sql'))

    prob.setup()

    # N2 generation
    om.n2(prob, 'trim.html', show_browser=False)

    prob['trim.hstab|cl'] = -2.0

    prob.check_partials(compact_print=True)

    # prob.set_val('climb.fltcond|vs', np.ones((num_nodes_nodes,))*1500, units='ft/min')

    prob.run_model()

    cr = om.CaseReader('trim_solver_solution.sql')
    final_case = cr.get_case(-1)
    # final_case.list_inputs(print_arrays=True)
    final_case.list_outputs(print_arrays=True)

def stability():
    prob = Problem()
    prob.model = StabilityDesignHorizontalTail(num_nodes=3, aircraft_model=acdata)

    prob.model.nonlinear_solver = NewtonSolver()
    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 10
    prob.model.nonlinear_solver.options['atol'] = 1e-6
    prob.model.nonlinear_solver.options['rtol'] = 1e-6
    prob.model.set_solver_print()

    prob.model.add_recorder(om.SqliteRecorder('stability_solver_solution.sql'))

    prob.setup()

    # N2 generation
    om.n2(prob, 'stability.html', show_browser=False)

    prob['hstab|S_ref'] = 8
    prob['wing|coords|x'] = -2

    # prob.check_partials(compact_print=True)

    # prob.set_val('climb.fltcond|vs', np.ones((num_nodes_nodes,))*1500, units='ft/min')

    prob.run_model()
    cr = om.CaseReader('stability_solver_solution.sql')
    final_case = cr.get_case(-1)
    # final_case.list_inputs(print_arrays=True)
    final_case.list_outputs(print_arrays=True)

if __name__ == '__main__':

    trim()
    stability()