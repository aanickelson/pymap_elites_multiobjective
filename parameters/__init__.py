from pymap_elites_multiobjective.parameters.parameters010 import Parameters as p010
from pymap_elites_multiobjective.parameters.parameters011 import Parameters as p011
from pymap_elites_multiobjective.parameters.parameters012 import Parameters as p012
from pymap_elites_multiobjective.parameters.parameters013 import Parameters as p013
from pymap_elites_multiobjective.parameters.parameters021 import Parameters as p021
from pymap_elites_multiobjective.parameters.parameters022 import Parameters as p022
from pymap_elites_multiobjective.parameters.parameters023 import Parameters as p023
from pymap_elites_multiobjective.parameters.parameters031 import Parameters as p031
from pymap_elites_multiobjective.parameters.parameters032 import Parameters as p032
from pymap_elites_multiobjective.parameters.parameters033 import Parameters as p033
from pymap_elites_multiobjective.parameters.parameters041 import Parameters as p041
from pymap_elites_multiobjective.parameters.parameters042 import Parameters as p042
from pymap_elites_multiobjective.parameters.parameters043 import Parameters as p043
from pymap_elites_multiobjective.parameters.parameters121 import Parameters as p121
from pymap_elites_multiobjective.parameters.parameters122 import Parameters as p122
from pymap_elites_multiobjective.parameters.parameters123 import Parameters as p123
from pymap_elites_multiobjective.parameters.parameters141 import Parameters as p141
from pymap_elites_multiobjective.parameters.parameters142 import Parameters as p142
from pymap_elites_multiobjective.parameters.parameters143 import Parameters as p143
from pymap_elites_multiobjective.parameters.parameters231 import Parameters as p231
from pymap_elites_multiobjective.parameters.parameters233 import Parameters as p233
from pymap_elites_multiobjective.parameters.parameters235 import Parameters as p235
from pymap_elites_multiobjective.parameters.parameters237 import Parameters as p237
from pymap_elites_multiobjective.parameters.parameters239 import Parameters as p239
from pymap_elites_multiobjective.parameters.parameters241 import Parameters as p241
from pymap_elites_multiobjective.parameters.parameters243 import Parameters as p243
from pymap_elites_multiobjective.parameters.parameters245 import Parameters as p245
from pymap_elites_multiobjective.parameters.parameters247 import Parameters as p247
from pymap_elites_multiobjective.parameters.parameters249 import Parameters as p249
from pymap_elites_multiobjective.parameters.parameters341 import Parameters as p341
from pymap_elites_multiobjective.parameters.parameters343 import Parameters as p343
from pymap_elites_multiobjective.parameters.parameters345 import Parameters as p345
from pymap_elites_multiobjective.parameters.parameters347 import Parameters as p347
from pymap_elites_multiobjective.parameters.parameters349 import Parameters as p349

no_cf = [p010, ]
no_close = [p011, p012, p013, ]
move_close = [p021, p022, p023, ]
no_far = [p031, p032, p033, ]
move_far = [p041, p042, p043, ]
poi_close = [p121, p122, p123, ]
poi_far = [p141, p142, p143, ]
no_far_new = [p231, p233, p235, p237, p239, ]
no_far_new_sm = [p231, p235, p239]

move_far_new = [p241, p243, p245, p247, p249, ]
move_far_new_sm = [p241, p245, p249]
poi_far_new = [p341, p343, p345, p347, p349, ]
poi_far_new_sm = [p341, p345, p349]
test_sm = [[p245, p345]]

new_batches = [poi_far_new_sm, no_far_new_sm, move_far_new_sm]
just_one = [poi_far_new_sm]
