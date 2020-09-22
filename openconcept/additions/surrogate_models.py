from openmdao.api import ImplicitComponent, ExplicitComponent, Group, ExternalCodeComp, IndepVarComp, BalanceComp, ExecComp, IndepVarComp
from openmdao.api import NewtonSolver, BoundsEnforceLS
from openmdao.api import DirectSolver, SqliteRecorder
from openmdao.api import Problem, MetaModelUnStructuredComp
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp, SumComp
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from openconcept.utilities.dvlabel import DVLabel

import numpy as np

import openmdao.api as om

class FoilSurrogate(MetaModelUnStructuredComp):

    def initialize(self):
        self.options.declare('default_surrogate', default=om.KrigingSurrogate())
        self.options.declare('vec_size', default=1)
        self.options.declare('dataset', default=np.zeros((5,1)))
        self.options.declare('train:alpha')
        self.options.declare('train:speed')
        self.options.declare('train:CD')
        self.options.declare('train:CL')
        self.options.declare('train:CM')

    def setup(self):
        dataset = self.options['dataset']
        self.add_input('alpha', 0., training_data=[ x[0] for x in dataset ])
        self.add_input('speed', 0., training_data=[ x[1] for x in dataset ])

        self.add_output('CD', 0., training_data=[ x[2] for x in dataset ])
        self.add_output('CL', 0., training_data=[ x[3] for x in dataset ])
        self.add_output('CM', 0., training_data=[ x[4] for x in dataset ])


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
print(hull_data)

class HullSurrogate(MetaModelUnStructuredComp):

    def initialize(self):
        self.options.declare('default_surrogate', default=om.KrigingSurrogate())
        self.options.declare('vec_size', default=1)
        self.options.declare('dataset', default=np.zeros((2,1)))
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

prob = om.Problem()

hydrofoil_data = [[15.0, 45.0, 0.18014105411251088, 0.5871405269194366, 0.09388405883237283], 
        [11.25, 45.0, 0.11025254400190256, 0.4433999087726861, 0.07294918296166375], 
        [7.5, 45.0, 0.054045576221288597, 0.26212441928703384, 0.043911113054372335], 
        [3.75, 45.0, 0.02221737496116033, 0.09061210714467337, 0.021296157191180155], 
        [0.0, 45.0, 0.01868686651715304, 0.013524231456119508, 0.02079947333249812], 
        [15.0, 34.0, 0.20285859445954924, 0.6784965331243692, 0.11457262624655475], 
        [11.25, 34.0, 0.12329497719580368, 0.5144769211311389, 0.08997984615384615], 
        [7.5, 34.0, 0.06524714937942443, 0.3467650134424518, 0.06598620313204712], 
        [3.75, 34.0, 0.02577366894041013, 0.1448902533177221, 0.03680872956652468], 
        [0.0, 34.0, 0.01947210219917277, 0.03749711544082959, 0.03603127863192181], 
        [15.0, 23.0, 0.2331393964014045, 0.8105435625121147, 0.19717770706704232], 
        [11.25, 23.0, 0.145975351993266, 0.6530511436113461, 0.12022875834001505], 
        [7.5, 23.0, 0.08202840962526786, 0.5026800758874596, 0.10984182032072162], 
        [3.75, 23.0, 0.03635839994659825, 0.3255932167996114, 0.09044248417188674], 
        [0.0, 23.0, 0.015505613287414738, 0.16380582789623707, 0.10284945300425959], 
        [15.0, 12.0, 0.28543433513557864, 1.2033250782507747, 0.15110494254322226], 
        [11.25, 12.0, 0.15428162327892703, 0.9757691898762281, 0.12406471873991481], 
        [7.5, 12.0, 0.06579780364381302, 0.6903379257527372, 0.10959421894815836], 
        [3.75, 12.0, 0.028382987073440124, 0.4273581791060729, 0.10313450231270357], 
        [0.0, 12.0, 0.012236829521134496, 0.18352618577642058, 0.09615879223502882], 
        [15.0, 1.0, 0.22642877211972098, 1.285253496056226, 1.189094742144826], 
        [11.25, 1.0, 0.11474785479019635, 0.9777204430882056, 0.957187953470308], 
        [7.5, 1.0, 0.052219415094759594, 0.699949034015138, 0.7194158465046354], 
        [3.75, 1.0, 0.02632417546107841, 0.44001202869355377, 0.4696727477073416], 
        [0.0, 1.0, 0.013411673104587062, 0.17936520753943141, 0.2663711022049612]]
        
# hydrofoil = prob.model.add_subsystem('hydrofoil', FoilSurrogate(dataset=hydrofoil_data, default_surrogate=om.KrigingSurrogate()))
hull = prob.model.add_subsystem('hull', HullSurrogate(dataset=hull_data, default_surrogate=om.KrigingSurrogate()))

prob.setup()

prob.final_setup()

# prob['hydrofoil.alpha'] = 2.1
# prob['hydrofoil.speed'] = 15

prob['hull.CVR'] = prob['hull.CVT'] = 2.

prob.check_partials(compact_print=True)
prob.model.list_inputs(prom_name=True,
                       hierarchical=True,
                       print_arrays=True)
prob.model.list_outputs(prom_name=True,
                        hierarchical=True,
                        print_arrays=True)

prob.run_model()
prob.model.list_inputs(print_arrays=True)
prob.model.list_outputs(print_arrays=True)