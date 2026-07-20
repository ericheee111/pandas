"""Shared machinery for platform aggregate ASV benchmarks."""


def select_case_params(case_type, method_name, param_indices):
    """Select one parameter combination from an upstream ASV benchmark."""
    method = getattr(case_type, method_name)
    raw_params = getattr(method, "params", getattr(case_type, "params", ()))
    params = list(raw_params)
    if params and not isinstance(params[0], (tuple, list)):
        params = [params]
    else:
        params = [list(axis) for axis in params]
    return tuple(params[axis][index] for axis, index in enumerate(param_indices))


class _AggregateBenchmark:
    """Run selected upstream ASV cases as one platform-weighted benchmark."""

    def setup(self):
        lengths = {
            len(self.case_params),
            len(self.run_repeat),
            len(self.case_methods),
            len(self.case_types),
        }
        if lengths != {len(self.case_params)}:
            raise ValueError("aggregate case metadata lengths differ")

        self.cases = []
        caches = {}
        try:
            for case_type, method_name, params in zip(
                self.case_types, self.case_methods, self.case_params
            ):
                if case_type not in caches:
                    cache_owner = case_type()
                    setup_cache = getattr(cache_owner, "setup_cache", None)
                    cache = setup_cache() if setup_cache is not None else None
                    caches[case_type] = cache
                cache = caches[case_type]
                call_params = ((cache,) if cache is not None else ()) + tuple(params)
                case = case_type()
                case_setup = getattr(case, "setup", None)
                if case_setup is not None:
                    case_setup(*call_params)
                self.cases.append((case, method_name, call_params))
        except BaseException:
            self.teardown()
            raise

    def time_aggregate(self):
        for (case, method_name, call_params), repeat in zip(
            self.cases, self.run_repeat
        ):
            method = getattr(case, method_name)
            for _ in range(repeat):
                method(*call_params)

    def teardown(self):
        cases = getattr(self, "cases", [])
        while cases:
            case, _method_name, call_params = cases.pop()
            case_teardown = getattr(case, "teardown", None)
            if case_teardown is not None:
                case_teardown(*call_params)
