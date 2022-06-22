from typing import Dict, Any

from solvers.base_solver import BaseSolver
from solvers.speech_enhancement.phonetic_aware_solver import PhoneticAwareSolver


class SolverFactory:

    supported_solvers: Dict[str, Any] = {
        'base': BaseSolver,
        "phonetic_aware_solver": PhoneticAwareSolver
    }

    @staticmethod
    def get_solver(data, model, optimizer, args):
        if args.solver_name not in SolverFactory.supported_solvers.keys():
            raise ValueError(f"Solver: {args.solver_name} is not supported by SolverFactory.\nPlease make sure implementation is valid.")
        return SolverFactory.supported_solvers[args.solver_name](data, model, optimizer, args)