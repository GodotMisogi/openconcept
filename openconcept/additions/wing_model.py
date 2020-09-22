from openmdao.api import ImplicitComponent, ExplicitComponent, Group, ExternalCodeComp, IndepVarComp, BalanceComp, ExecComp, IndepVarComp
from openmdao.api import NewtonSolver, BoundsEnforceLS
from openmdao.api import DirectSolver, SqliteRecorder
from openmdao.api import Problem
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp, SumComp
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from openconcept.utilities.dvlabel import DVLabel

import numpy as np
import openmdao.api as om

class TrapezoidalLiftingSurface(ExplicitComponent):
    """
    Defines a half-wing as a sequence of trapeziums (trapezoids if you're 'murican) from root to tip.

    Inputs
    ------
    chords : [float]
        Respective taper ratios of each section with respect to the root chord length, with the tip chord ratio for the last entry. Automatically constrained as a result. (vector, m)
    spans : [float]
        Spans of each section (vector, m)
    c4sweeps : float
        Quarter-chord sweep angles, TBD how to use them. (vector, dimensionless)

    Outputs
    -------
    S_ref : float
        Reference area of the wing (scalar, m)
    AR : float
        Aspect ratio of the wing (scalar, dimensionless)

    Options
    -------
    num_sections : int
        Number of airfoil sections (sets vec length) (default 1)
    """
    def initialize(self):
        self.options.declare('num_sections', default=1, types=int, desc='Number of airfoil sections')

    def setup(self):
        self.add_input('chords', shape=(self.options['num_sections']+1,), units='m')
        self.add_input('spans', shape=(self.options['num_sections'],), units='m')
        self.add_input('c4sweeps', units='deg')

        self.add_output('S_ref', units='m**2')
        self.add_output('AR', units='m')

    def compute(self, inputs, outputs):
        chords = inputs['chords']
        spans = inputs['spans']
        outputs['S_ref'] = sum([ (c1 + c2) * s / 2 for (c2, c1, s) in zip(chords[:-1], chords[1:], spans) ])
        outputs['AR'] = sum(inputs['spans'])**2 / outputs['S_ref']
        # outputs['x_mac'] =
        # outputs['y_mac'] =

# class GlobalCoordinates(ExplicitComponent):

#     def setup(self):
#         self.add_input()

class LiftingSurface(ExplicitComponent):

    def initialize(self):
        self.options.declare('name', default='wing', types=str)

    def setup(self):
        name = self.options['name']
        self.add_input('span', units='m')
        self.add_input('AR', units=None)
        self.add_input('taper', units=None)
        self.add_input('incidence', units='deg')
        self.add_input('dihedral', units='deg')
        self.add_input('sweep', units='deg')

        self.add_output('S_ref', units='m**2')
        self.add_output('chord|mean', units='m')
        self.add_output('chord|root', units='m')
        self.add_output('chord|tip', units='m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        span = inputs['span']
        AR = inputs['AR']
        taper_ratio = inputs['taper']
        dihedral = inputs['dihedral']
        incidence = inputs['incidence']
        taper_scale = (3.0/2.0)*(1.0+taper_ratio)/(1.0+taper_ratio+taper_ratio**2)

        outputs['S_ref'] = span**2/AR*np.cos(dihedral)**2
        outputs['chord|mean'] = span/AR
        outputs['chord|root'] = outputs['chord|mean'] * taper_scale
        outputs['chord|tip'] = outputs['chord|root'] * taper_ratio

    def compute_partials(self, inputs, partials):
        span = inputs['span']
        AR = inputs['AR']
        taper_ratio = inputs['taper']
        dihedral = inputs['dihedral']
        incidence = inputs['incidence']
        taper_scale = (3.0/2.0)*(1.0+taper_ratio)/(1.0+taper_ratio+taper_ratio**2)

        partials['S_ref', 'span'] = 2*span/AR*np.cos(dihedral)**2
        partials['S_ref', 'AR'] = -(span/AR*np.cos(dihedral))**2
        partials['S_ref', 'dihedral'] = span**2/AR*(-2*np.cos(dihedral)*np.sin(dihedral))
        
        partials['chord|mean', 'span'] = 1/AR
        partials['chord|mean', 'AR'] = -span/AR**2

        partials['chord|root', 'span'] = taper_scale * 1/AR
        partials['chord|root', 'AR'] = -taper_scale * span/AR**2
        partials['chord|root', 'taper'] = taper_scale*(1./(1. + taper_ratio) - (1. + 2*taper_ratio)/(1. + taper_ratio + taper_ratio**2))*span/AR 

        partials['chord|tip', 'span'] = taper_ratio * taper_scale * 1/AR
        partials['chord|tip', 'AR'] = -taper_ratio * taper_scale * span/AR**2
        partials['chord|tip', 'taper'] = taper_ratio*taper_scale*(1./taper_ratio + 1./(1. + taper_ratio) - (1. + 2*taper_ratio)/(1. + taper_ratio + taper_ratio**2))*span/AR 

class LiftingSurfaceWeight(ExplicitComponent):

    def setup(self):
        self.add_input('loading', units='N/m**2', desc='Loading')
        self.add_input('S_ref', units='m**2')

        self.add_output('weight', units='N')

    def compute(self, inputs, outputs):
        outputs['weight'] = inputs['loading'] * inputs['S_ref']

class Wing(Group):

    def setup(self):
        # const = self.add_subsystem('const', IndepVarComp(), promotes_inputs=['*'], promotes_outputs=['*'])
        # const.add_output('wing_loading', units='N/m**2')
        # TODO: Airfoil modules

        indeps = self.add_subsystem('dv_comp', IndepVarComp(), promotes_outputs=['*'])
        indeps.add_output('span', val=19.8, units='m')
        indeps.add_output('AR', val=10, units=None)
        indeps.add_output('taper', val=0.5, units=None)
        indeps.add_output('incidence', val=3., units='deg')
        indeps.add_output('dihedral', val=3., units='deg')
        indeps.add_output('sweep', val=0., units='deg')
        indeps.add_output('loading', val=800, units='N/m**2')

        self.add_subsystem('geom', LiftingSurface(name='wing'), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('weight', LiftingSurfaceWeight(), promotes_inputs=['*'], promotes_outputs=['*'])

class HorizontalStab(Group):

    def setup(self):
        # const = self.add_subsystem('const', IndepVarComp(), promotes_inputs=['*'], promotes_outputs=['*'])
        # const.add_output('wing_loading', units='N/m**2')
        # TODO: Airfoil modules

        indeps = self.add_subsystem('dv_comp', IndepVarComp(), promotes_outputs=['*'])
        indeps.add_output('span', val=4, units='m')
        indeps.add_output('AR', val=3, units=None)
        indeps.add_output('taper', val=0.8, units=None)
        indeps.add_output('incidence', val=0., units='deg')
        indeps.add_output('dihedral', val=0., units='deg')
        indeps.add_output('sweep', val=15., units='deg')
        indeps.add_output('loading', val=800, units='N/m**2')

        self.add_subsystem('geom', LiftingSurface(name='wing'), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('weight', LiftingSurfaceWeight(), promotes_inputs=['*'], promotes_outputs=['*'])

prob = Problem()
prob.model = Wing()

prob.setup(check=True, mode='fwd')

# N2 generation
om.n2(prob, 'wing.html', show_browser=False)

prob.check_partials(compact_print=True)
prob.model.list_inputs(prom_name=True,
                       hierarchical=True,
                       print_arrays=True)
prob.model.list_outputs(prom_name=True,
                        hierarchical=True,
                        print_arrays=True)
prob.run_model()
