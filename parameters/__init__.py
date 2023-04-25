from pymap_elites_multiobjective.parameters.parameters01 import Parameters as p01
from pymap_elites_multiobjective.parameters.parameters02 import Parameters as p02
from pymap_elites_multiobjective.parameters.parameters03 import Parameters as p03
from pymap_elites_multiobjective.parameters.parameters04 import Parameters as p04
from pymap_elites_multiobjective.parameters.parameters10 import Parameters as p10
from pymap_elites_multiobjective.parameters.parameters11 import Parameters as p11
from pymap_elites_multiobjective.parameters.parameters12 import Parameters as p12
from pymap_elites_multiobjective.parameters.parameters13 import Parameters as p13
from pymap_elites_multiobjective.parameters.parameters21 import Parameters as p21
from pymap_elites_multiobjective.parameters.parameters22 import Parameters as p22
from pymap_elites_multiobjective.parameters.parameters23 import Parameters as p23

# these counterfactual agents don't move
batch_0cf = [p10]
batch_no_move = [p11, p12, p13]
# These do move (obv. except the case with 0 agents)
batch_move = [p21, p22, p23]

