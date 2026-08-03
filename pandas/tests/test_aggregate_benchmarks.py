def test_aggregate_950_tslibs_datetime_imports() -> None:
    from asv_bench.benchmarks.aggregate_950 import tslibs3

    offsets_case = tslibs3._Official_tslibs_offsets_OffestDatetimeArithmetic()
    offsets_case.setup(tslibs3.offset_objs[0])
    assert offsets_case.date.year == 2011

    timedelta_case = tslibs3._Official_tslibs_timedelta_TimedeltaConstructor()
    timedelta_case.setup()
    assert timedelta_case.dttimedelta.total_seconds() == 3600
