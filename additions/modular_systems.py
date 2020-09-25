import openmdao.api as om


class fComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('c', default=1.0, types=(float, int))

    def setup(self):
        self.add_input('x', shape=(2,))
        self.add_output('f', shape=(1,))

        self.declare_partials(of='f', wrt='x', method='cs')

    def compute(self, inputs, outputs):
        xs = inputs['x']
        c = self.options['c']

        outputs['f'] = -xs[0]**2 + xs[1] + c

class gComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('d', default=1.0, types=(float, int))

    def setup(self):
        self.add_input('x', shape=(2,))
        self.add_output('g', shape=(1,))

        self.declare_partials(of='g', wrt='x', method='cs')

    def compute(self, inputs, outputs):
        xs = inputs['x']
        d = self.options['d']

        outputs['g'] = -xs[0]**3 + xs[1]**2 - d


class hComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('d', default=1.0, types=(float, int))

    def setup(self):
        self.add_input('x', shape=(2,))
        self.add_output('h', shape=(1,))

        self.declare_partials(of='h', wrt='x', method='cs')

    def compute(self, inputs, outputs):
        xs = inputs['x']
        d = self.options['d']

        outputs['h'] = xs[0] + xs[1]**2 + d


class FComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('x', shape=(2,))
        self.add_output('F', shape=(1,))

        self.declare_partials(of='F', wrt='x', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        xs = inputs['x']
        outputs['F'] = xs[0] ** 2 - xs[1] ** 2 - 5


class ResidComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('m', default=5.0, types=(float, int))
        self.options.declare('n', default=6.0, types=(float, int))
        self.options.declare('resid_name', default='R', types=(str,))
        self.options.declare('f1_name', default='f1', types=(str,))
        self.options.declare('f2_name', default='f2', types=(str,))

    def setup(self):
        f1_name = self.options['f1_name']
        f2_name = self.options['f2_name']
        resid_name = self.options['resid_name']

        self.add_input(f1_name, shape=(1,))
        self.add_input(f2_name, shape=(1,))
        self.add_output(resid_name, shape=(1,))

        self.const = self.options['m'] * self.options['n']

        self.declare_partials(of=resid_name, wrt=f1_name, val=1.0)
        self.declare_partials(of=resid_name, wrt=f2_name, val=-1.0)

    def compute(self, inputs, outputs):
        f1 = inputs[self.options['f1_name']]
        f2 = inputs[self.options['f2_name']]
        outputs[self.options['resid_name']] = f1 - f2 + self.const


class ResidualerGroup(om.Group):

    def initialize(self):
        self.options.declare('use_solver', default=False, types=(bool,))

        self._residuals = []

    def add_residual(self, name, f1, f2, f1_name, f2_name, m, n):
        self._residuals.append({'name': name,
                                'f1': f1,
                                'f2': f2,
                                'f1_name': f1_name,
                                'f2_name': f2_name,
                                'm': m,
                                'n': n})

    def setup(self):


        if self.options['use_solver']:
            bal = om.BalanceComp()
            mux = om.MuxComp(vec_size=2)
            mux.add_var('x', shape=(1,), axis=1)

        for i, resid in enumerate(self._residuals):
            g = om.Group()

            name = resid['name']

            f1_name = resid['f1_name']
            f2_name = resid['f2_name']

            m = resid['m']
            n = resid['n']

            g.add_subsystem(f'{f1_name}_comp', subsys=resid['f1'], promotes_inputs=['x'], promotes_outputs=[f1_name])
            g.add_subsystem(f'{f2_name}_comp', subsys=resid['f2'], promotes_inputs=['x'], promotes_outputs=[f2_name])

            g.add_subsystem(f'{name}_comp',
                            subsys=ResidComp(resid_name=name, f1_name=f1_name, f2_name=f2_name, m=m, n=n),
                            promotes_inputs=[f1_name, f2_name],
                            promotes_outputs=[name])

            self.add_subsystem(f'r{i}', g, promotes_inputs=['x'], promotes_outputs=[name])

            if self.options['use_solver']:
                bal.add_balance(name=f'x{i}', lhs_name=name, normalize=True)
                self.connect(f'bal.x{i}', f'mux.x_{i}')

        if self.options['use_solver']:
            self.add_subsystem('bal', bal, promotes_inputs=[r['name'] for r in self._residuals])
            self.add_subsystem('mux', mux, promotes_outputs=['x'])

            self.linear_solver = om.DirectSolver()
            self.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=False)


def problem1_optimizer():

    p = om.Problem(model=om.Group())

    #
    # Subsystems
    #

    ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['x'])
    ivc.add_output('x', shape=(2,))

    residular = ResidualerGroup()

    residular.add_residual('R', f1=fComp(c=1), f2=gComp(d=2), m=0, n=0, f1_name='f', f2_name='g')
    # residular.add_residual('S', f1=fComp(c=3), f2=gComp(d=4), m=5, n=6, f1_name='f', f2_name='g')

    p.model.add_subsystem('residualer', subsys=residular, promotes_inputs=['x'])

    p.model.add_subsystem('obj_comp', subsys=FComp(), promotes_inputs=['x'])

    #
    # Optimization problem
    #
    p.model.add_objective('obj_comp.F', ref=1.0)
    p.model.add_design_var('x', ref=1.0)
    p.model.add_constraint('residualer.R', equals=0.0)

    #
    # Driver
    #
    p.driver = om.ScipyOptimizeDriver()

    #
    # Record the driver iterations
    #
    p.driver.add_recorder(om.SqliteRecorder('problem1_optimizer_solution.sql'))

    #
    # Setup
    #
    p.setup()
    om.n2(p, 'problem_2.html')

    #
    # Set initial guess for design variables
    #
    p.set_val('x', [-2, 2])

    #
    # Run optimization driver
    #
    p.run_driver()

    #
    # Print results (using recorders is good practice)
    #
    cr = om.CaseReader('problem1_optimizer_solution.sql')
    final_case = cr.get_case(-1)
    final_case.list_outputs(print_arrays=True)

def problem2_optimizer():

    p = om.Problem(model=om.Group())

    #
    # Subsystems
    #

    ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['x'])
    ivc.add_output('x', shape=(2,))

    residular = ResidualerGroup()

    residular.add_residual('R', f1=fComp(c=6), f2=gComp(d=4), m=0, n=0, f1_name='f', f2_name='g')
    residular.add_residual('S', f1=fComp(c=4), f2=hComp(d=3), m=5, n=6, f1_name='f', f2_name='h')

    p.model.add_subsystem('residualer', subsys=residular, promotes_inputs=['x'])

    p.model.add_subsystem('obj_comp', subsys=FComp(), promotes_inputs=['x'])

    #
    # Optimization problem
    #
    p.model.add_objective('obj_comp.F', ref=1.0)
    p.model.add_design_var('x', ref=1.0)
    p.model.add_constraint('residualer.R', equals=0.0)
    p.model.add_constraint('residualer.S', equals=0.0)

    #
    # Driver
    #
    p.driver = om.ScipyOptimizeDriver()

    #
    # Record the driver iterations
    #
    p.driver.add_recorder(om.SqliteRecorder('problem2_optimizer_solution.sql'))

    #
    # Setup
    #
    p.setup()

    #
    # Set initial guess for design variables
    #
    p.set_val('x', [-2, 2])

    #
    # Run optimization driver
    #
    p.run_driver()

    #
    # Print results (using recorders is good practice)
    #
    cr = om.CaseReader('problem2_optimizer_solution.sql')
    final_case = cr.get_case(-1)
    final_case.list_outputs(print_arrays=True)

def problem2_solver():

    p = om.Problem(model=om.Group())

    #
    # Subsystems
    #
    residular = ResidualerGroup(use_solver=True)

    residular.add_residual('R', f1=fComp(c=6), f2=gComp(d=4), m=0, n=0, f1_name='f', f2_name='g')
    residular.add_residual('S', f1=fComp(c=4), f2=hComp(d=3), m=5, n=6, f1_name='f', f2_name='h')

    p.model.add_subsystem('residualer', subsys=residular, promotes_outputs=['x'])
    p.model.add_subsystem('obj_comp', subsys=FComp(), promotes_inputs=['x'])

    #
    # Record the model directly since theres no driver
    #
    p.model.add_recorder(om.SqliteRecorder('problem2_solver_solution.sql'))

    #
    # Setup
    #
    p.setup()
    om.n2(p, 'problem_2.html')
    #
    # Set initial guess for design variables
    #
    p.set_val('x', [[-2, 2]])

    #
    # Run optimization driver
    #
    # p.run_driver()
    p.run_model()

    #
    # Print results (using recorders is good practice)
    #
    cr = om.CaseReader('problem2_solver_solution.sql')
    final_case = cr.get_case(-1)
    final_case.list_outputs(print_arrays=True)


if __name__ == '__main__':

    # problem1_optimizer()
    problem2_optimizer()
    # problem2_solver()