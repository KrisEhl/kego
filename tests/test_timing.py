from kego.timing import Timings


def test_add_accumulates_total_and_count():
    t = Timings()
    t.add("x", 1.0)
    t.add("x", 2.0)
    assert t.as_dict()["x"] == (3.0, 2)


def test_merge_combines_registries():
    a = Timings()
    a.add("x", 1.0)
    a.add("y", 2.0)
    b = Timings()
    b.add("x", 3.0)
    a.merge(b)
    d = a.as_dict()
    assert d["x"] == (4.0, 2)
    assert d["y"] == (2.0, 1)


def test_merge_accepts_raw_dict():
    a = Timings()
    a.add("x", 1.0)
    a.merge({"x": (2.0, 3)})
    assert a.as_dict()["x"] == (3.0, 4)


def test_timer_context_records_one_call():
    t = Timings()
    with t.timer("blk"):
        pass
    secs, n = t.as_dict()["blk"]
    assert n == 1
    assert secs >= 0.0


def test_reset_clears():
    t = Timings()
    t.add("x", 1.0)
    t.reset()
    assert t.as_dict() == {}
