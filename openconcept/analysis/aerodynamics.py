"""Aerodynamic analysis routines usable for multiple purposes / flight phases"""
from __future__ import division
from openmdao.api import ExplicitComponent, MetaModelUnStructuredComp, Group, KrigingSurrogate
import numpy as np

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
# print(hull_data)

class HullSurrogate(MetaModelUnStructuredComp):

    def initialize(self):
        self.options.declare('default_surrogate', default=KrigingSurrogate())
        self.options.declare('vec_size', default=1)
        self.options.declare('dataset', default=np.zeros((4,1)))
        self.options.declare('train:CVR')
        self.options.declare('train:CVT')
        self.options.declare('train:CR')
        self.options.declare('train:alphat')

    def setup(self):
        dataset = self.options['dataset']
        self.add_input('CVR', 0., training_data=[ x[0] for x in dataset ])
        self.add_input('CVT', 0., training_data=[ x[2] for x in dataset ])
        self.add_output('CR', 0., training_data=[ x[1] for x in dataset ])
        self.add_output('alphat', 0., training_data=[ x[3] for x in dataset ])

class VelocityCoefficient(ExplicitComponent):

    def setup(self):
        self.add_input('fltcond|Utrue', val=0., units='m')
        self.add_input('B', val=0., units='m')

        self.add_output('CVR')
        self.add_output('CVT')

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

    def setup(self):
        self.add_input('CR', val=0)
        self.add_input('B', units='m')

        self.add_output('resistance', units='N')

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

class HullResistance(Group):

    def setup(self):
        self.add_subsystem('vel_coeff', VelocityCoefficient(), promotes=['*'])
        self.add_subsystem('surrogate', HullSurrogate(dataset=hull_data), promotes=['*'])
        self.add_subsystem('res', Resistance(), promotes=['*'])

class PolarDrag(ExplicitComponent):
    """
    Calculates drag force based on drag polar formulation

    Inputs
    ------
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)
    fltcond|q : float
        Dynamic pressure (vector, Pascals)
    ac|geom|wing|S_ref : float
        Reference wing area (scalar, m**2)
    ac|geom|wing|AR : float
        Wing aspect ratio (scalar, dimensionless)
    CD0 : float
        Zero-lift drag coefficient (scalar, dimensionless)
    e : float
        Wing Oswald efficiency (scalar, dimensionless)

    Outputs
    -------
    drag : float
        Drag force (vector, Newtons)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")

    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('fltcond|CL', shape=(nn,))
        self.add_input('fltcond|q', units='N * m**-2', shape=(nn,))
        self.add_input('ac|geom|wing|S_ref', units='m **2')
        self.add_input('CD0')
        self.add_input('e')
        self.add_input('ac|geom|wing|AR')
        self.add_output('drag', units='N', shape=(nn,))

        self.declare_partials(['drag'], ['fltcond|CL', 'fltcond|q'], rows=arange, cols=arange)
        self.declare_partials(['drag'],
                              ['ac|geom|wing|S_ref', 'ac|geom|wing|AR', 'CD0', 'e'],
                              rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        outputs['drag'] = (inputs['fltcond|q'] * inputs['ac|geom|wing|S_ref'] *
                           (inputs['CD0'] + inputs['fltcond|CL']**2 / np.pi / inputs['e'] /
                            inputs['ac|geom|wing|AR']))

    def compute_partials(self, inputs, J):
        J['drag', 'fltcond|q'] = (inputs['ac|geom|wing|S_ref'] *
                                  (inputs['CD0'] + inputs['fltcond|CL']**2 / np.pi /
                                   inputs['e'] / inputs['ac|geom|wing|AR']))
        J['drag', 'fltcond|CL'] = (inputs['fltcond|q'] * inputs['ac|geom|wing|S_ref'] *
                                   (2 * inputs['fltcond|CL'] / np.pi / inputs['e'] /
                                    inputs['ac|geom|wing|AR']))
        J['drag', 'CD0'] = inputs['fltcond|q'] * inputs['ac|geom|wing|S_ref']
        J['drag', 'e'] = - (inputs['fltcond|q'] * inputs['ac|geom|wing|S_ref'] *
                            inputs['fltcond|CL']**2 / np.pi /
                            inputs['e']**2 / inputs['ac|geom|wing|AR'])
        J['drag', 'ac|geom|wing|S_ref'] = (inputs['fltcond|q'] * (inputs['CD0'] +
                                           inputs['fltcond|CL']**2 / np.pi / inputs['e'] /
                                           inputs['ac|geom|wing|AR']))
        J['drag', 'ac|geom|wing|AR'] = - (inputs['fltcond|q'] * inputs['ac|geom|wing|S_ref'] *
                                          inputs['fltcond|CL']**2 / np.pi / inputs['e'] /
                                          inputs['ac|geom|wing|AR']**2)


class Lift(ExplicitComponent):
    """
    Calculates lift force based on CL, dynamic pressure, and wing area

    Inputs
    ------
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)
    fltcond|q : float
        Dynamic pressure (vector, Pascals)
    ac|geom|wing|S_ref : float
        Reference wing area (scalar, m**2)

    Outputs
    -------
    lift : float
        Lift force (vector, Newtons)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length) (default 1)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of nodes to compute")

    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('fltcond|CL', shape=(nn,))
        self.add_input('fltcond|q', units='N * m**-2', shape=(nn,))
        self.add_input('ac|geom|wing|S_ref', units='m **2')

        self.add_output('lift', units='N', shape=(nn,))
        self.declare_partials(['lift'], ['fltcond|CL', 'fltcond|q'], rows=arange, cols=arange)
        self.declare_partials(['lift'], ['ac|geom|wing|S_ref'], rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        outputs['lift'] = inputs['fltcond|q'] * inputs['ac|geom|wing|S_ref'] * inputs['fltcond|CL']

    def compute_partials(self, inputs, J):
        J['lift', 'fltcond|q'] = inputs['ac|geom|wing|S_ref'] * inputs['fltcond|CL']
        J['lift', 'fltcond|CL'] = inputs['fltcond|q'] * inputs['ac|geom|wing|S_ref']
        J['lift', 'ac|geom|wing|S_ref'] = inputs['fltcond|q'] * inputs['fltcond|CL']


class StallSpeed(ExplicitComponent):
    """
    Calculates stall speed based on CLmax, wing area, and weight

    Inputs
    ------
    CLmax : float
        Maximum lfit coefficient (scalar, dimensionless)
    weight : float
        Dynamic pressure (scalar, kg)
    ac|geom|wing|S_ref : float
        Reference wing area (scalar, m**2)

    Outputs
    -------
    Vstall_eas : float
        Stall speed (scalar, m/s)
    """

    def setup(self):
        self.add_input('weight', units='kg')
        self.add_input('ac|geom|wing|S_ref', units='m**2')
        self.add_input('CLmax')
        self.add_output('Vstall_eas', units='m/s')
        self.declare_partials(['Vstall_eas'], ['weight', 'ac|geom|wing|S_ref', 'CLmax'])

    def compute(self, inputs, outputs):
        g = 9.80665  # m/s^2
        rho = 1.225  # kg/m3
        outputs['Vstall_eas'] = np.sqrt(2 * inputs['weight'] * g / rho /
                                        inputs['ac|geom|wing|S_ref'] / inputs['CLmax'])

    def compute_partials(self, inputs, J):
        g = 9.80665  # m/s^2
        rho = 1.225  # kg/m3
        J['Vstall_eas', 'weight'] = (1 / np.sqrt(2 * inputs['weight'] * g / rho /
                                                 inputs['ac|geom|wing|S_ref'] / inputs['CLmax']) *
                                     g / rho / inputs['ac|geom|wing|S_ref'] / inputs['CLmax'])
        J['Vstall_eas', 'ac|geom|wing|S_ref'] = - (1 / np.sqrt(2 * inputs['weight'] * g / rho /
                                                               inputs['ac|geom|wing|S_ref'] /
                                                               inputs['CLmax']) *
                                                   inputs['weight'] * g / rho /
                                                   inputs['ac|geom|wing|S_ref'] ** 2 /
                                                   inputs['CLmax'])
        J['Vstall_eas', 'CLmax'] = - (1 / np.sqrt(2 * inputs['weight'] * g / rho /
                                                  inputs['ac|geom|wing|S_ref'] /
                                                  inputs['CLmax']) *
                                      inputs['weight'] * g / rho /
                                      inputs['ac|geom|wing|S_ref'] / inputs['CLmax'] ** 2)
