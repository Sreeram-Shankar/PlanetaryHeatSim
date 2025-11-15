from functools import partial
from .bdf import solve_be, solve_bdf2, solve_bdf3, solve_bdf4, solve_bdf5, solve_bdf6
from .rk import solve_rk1, solve_rk2, solve_rk3, solve_rk4, solve_rk5
from .ab import solve_ab2, solve_ab3, solve_ab4, solve_ab5
from .am import solve_am2, solve_am3, solve_am4, solve_am5
from .irk import solve_collocation
from .sdirk import solve_sdirk2, solve_sdirk3, solve_sdirk4
solver_map = {
    "BE": solve_be,
    "BDF2": solve_bdf2,
    "BDF3": solve_bdf3,
    "BDF4": solve_bdf4,
    "BDF5": solve_bdf5,
    "BDF6": solve_bdf6,
    "RK1": solve_rk1,
    "RK2": solve_rk2,
    "RK3": solve_rk3,
    "RK4": solve_rk4,
    "RK5": solve_rk5,
    "AB2": solve_ab2,
    "AB3": solve_ab3,
    "AB4": solve_ab4,
    "AB5": solve_ab5,
    "AM2": solve_am2,
    "AM3": solve_am3,
    "AM4": solve_am4,
    "AM5": solve_am5,
    "GL1": partial(solve_collocation, family="gauss", s=1),
    "GL2": partial(solve_collocation, family="gauss", s=2),
    "GL3": partial(solve_collocation, family="gauss", s=3),
    "GL4": partial(solve_collocation, family="gauss", s=4),
    "GL5": partial(solve_collocation, family="gauss", s=5),
    "R1": partial(solve_collocation, family="radau", s=1),
    "R2": partial(solve_collocation, family="radau", s=2),
    "R3": partial(solve_collocation, family="radau", s=3),
    "R4": partial(solve_collocation, family="radau", s=4),
    "R5": partial(solve_collocation, family="radau", s=5),
    "L1": partial(solve_collocation, family="lobatto", s=1),
    "L2": partial(solve_collocation, family="lobatto", s=2),
    "L3": partial(solve_collocation, family="lobatto", s=3),
    "L4": partial(solve_collocation, family="lobatto", s=4),
    "L5": partial(solve_collocation, family="lobatto", s=5),
    "SDIRK2": solve_sdirk2,
    "SDIRK3": solve_sdirk3,
    "SDIRK4": solve_sdirk4,
}
