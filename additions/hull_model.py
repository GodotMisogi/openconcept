from openmdao.api import ImplicitComponent, ExplicitComponent, Group, ExternalCodeComp
import numpy as np
        
class Hull(ExplicitComponent):
    """
    Defines a hull based on parameters.

    Inputs
    ------


    Outputs
    -------


    Options
    -------

    """
    def initialize(self):
        self.options.declare('num_sections', default=2, types=int,
                             desc='Number of sections corresponding to sharpedges or steps of the hull.')
        self.options.declare('reserve', default=200, types=float,
                             desc='Weight multiplier for factor of safety.')

    def setup(self, inputs, outputs):
        self.add_input('c_delta', units=None,
                        desc='Load coefficient')
        self.add_input('slenderness', units=None,
                        desc='Slenderness ratio, defined as L/B. Starts after the first section, i.e. the afterbody.')
        self.add_input('weight', units='N',
                        desc='Aircraft weight.')
 

        self.add_output('beam_width', units='m',
                        desc='Beam width of the hull.')
        self.add_output('volume', units='m**3',
                        desc="Volume required to maintain buoyancy based on Archimedes' principle.")
        self.add_output('length', units='m',
                        desc="Length of the hull.")

    def compute(self, inputs, outputs):
        rho, g = 1000, 9.81
        weight = self.options['reserve']*inputs['weight']/100
        outputs['volume'] = weight / (rho * g)
        outputs['beam_width'] = (weight/(inputs['c_delta'] * rho * g))**(1./3)
        outputs['length'] = inputs['slenderness'] * outputs['volume'] / outputs['beam_width']** 2
        
    # def compute_partials(self, inputs, outputs):
