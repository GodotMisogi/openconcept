from __future__ import division
import numpy as np
from openmdao.api import ExplicitComponent
from openmdao.api import Group


class SimpleMotor(ExplicitComponent):
    """
    A simple motor which creates shaft power and draws electrical load

    Input Vars
    ----------
    throttle : float
        (n vector, dimensionless) Should be [0, 1]
    elec_power_rating: float
        (scalar, W) Electric (not mech) design power

    Output Vars
    -----------
    shaft_power_out : float
        (n vector, W)
    elec_load : float
        (n vector, W)
    heat_out : float
        (n vector, W)
    component_cost : float
        (scalar, USD)
    component_weight : float
        (scalar, kg)
    component_sizing_margin : float
        (n vector, dimensionless)


    Options
    -------
    num_nodes : int
        (default 1) Number of analysis points to run (sets vec length)
    efficiency : float
        (default 1.0) Shaft power efficiency. Sensible range 0.0 to 1.0
    weight_inc : float
        (default 1/5000, kg/W) Weight per unit rated power
    weight_base : float
        (default 0, kg) Base weight
    cost_inc : float
        (default 0.134228, USD/W) Cost per unit rated power
    cost_base : float
        (default 1 USD) Base cost
    """

    def initialize(self):
        # define technology factors
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('efficiency', default=1., desc='Efficiency (dimensionless)')
        self.options.declare('weight_inc', default=1 / 5000, desc='kg/W')  # 5kW/kg
        self.options.declare('weight_base', default=0., desc='kg base weight')
        self.options.declare('cost_inc', default=100 / 745, desc='$ cost per watt')
        self.options.declare('cost_base', default=1., desc='$ cost base')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('throttle', desc='Throttle input (Fractional)', shape=(nn,))
        self.add_input('elec_power_rating', units='W', desc='Rated electrical power (load)')

        # outputs and partials
        eta_m = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        self.add_output('shaft_power_out', units='W', desc='Output shaft power', shape=(nn,))
        self.add_output('heat_out', units='W', desc='Waste heat out', shape=(nn,))
        self.add_output('elec_load', units='W', desc='Electrical load consumed', shape=(nn,))
        self.add_output('component_cost', units='USD', desc='Motor component cost')
        self.add_output('component_weight', units='kg', desc='Motor component weight')
        self.add_output('component_sizing_margin', desc='Fraction of rated power', shape=(nn,))
        self.declare_partials(["*"], ["*"], dependent=False)
        self.declare_partials('shaft_power_out', 'elec_power_rating')
        self.declare_partials('shaft_power_out', 'throttle', 'elec_power_rating',
                              rows=range(nn), cols=range(nn))
        self.declare_partials('heat_out', 'elec_power_rating')
        self.declare_partials('heat_out', 'throttle', 'elec_power_rating',
                              rows=range(nn), cols=range(nn))
        self.declare_partials('elec_load', 'elec_power_rating')
        self.declare_partials('elec_load', 'throttle', rows=range(nn), cols=range(nn))
        self.declare_partials('component_cost', 'elec_power_rating', val=cost_inc)
        self.declare_partials('component_weight', 'elec_power_rating', val=weight_inc)
        self.declare_partials('component_sizing_margin', 'throttle',
                              val=1.0 * np.ones(nn), rows=range(nn), cols=range(nn))

    def compute(self, inputs, outputs):
        eta_m = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        outputs['shaft_power_out'] = inputs['throttle'] * inputs['elec_power_rating'] * eta_m
        outputs['heat_out'] = inputs['throttle'] * inputs['elec_power_rating'] * (1 - eta_m)
        outputs['elec_load'] = inputs['throttle'] * inputs['elec_power_rating']
        outputs['component_cost'] = inputs['elec_power_rating'] * cost_inc + cost_base
        outputs['component_weight'] = inputs['elec_power_rating'] * weight_inc + weight_base
        outputs['component_sizing_margin'] = inputs['throttle']

    def compute_partials(self, inputs, J):
        eta_m = self.options['efficiency']
        J['shaft_power_out', 'throttle'] = inputs['elec_power_rating'] * eta_m
        J['shaft_power_out', 'elec_power_rating'] = inputs['throttle'] * eta_m
        J['heat_out', 'throttle'] = inputs['elec_power_rating'] * (1 - eta_m)
        J['heat_out', 'elec_power_rating'] = inputs['throttle'] * (1 - eta_m)
        J['elec_load', 'throttle'] = inputs['elec_power_rating']
        J['elec_load', 'elec_power_rating'] = inputs['throttle']
