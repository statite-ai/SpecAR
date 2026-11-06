"""Backend selection logic; runs on any machine (no GPU, no torch): python -m pytest tests  or  python tests/test_attention.py"""
import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location("specar_attention", os.path.join(os.path.dirname(__file__), "..", "specar", "attention.py"))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
ALL = {"fa3": True, "fa4": True, "fa2": True}
NONE = {"fa3": False, "fa4": False, "fa2": False}


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f"expected {exc.__name__}")


def test_auto_picks_by_architecture():
    assert A.choose(9, ALL) == "fa3"                                   # Hopper: the reference backend
    assert A.choose(10, ALL) == "fa4"                                  # Blackwell
    assert A.choose(8, ALL) == "fa2"                                   # anything else
    assert A.choose(12, ALL) == "fa2"


def test_auto_falls_back_to_what_is_installed():
    assert A.choose(9, {"fa3": False, "fa4": True, "fa2": True}) == "fa4"
    assert A.choose(9, {"fa3": False, "fa4": False, "fa2": True}) == "fa2"
    assert A.choose(10, {"fa3": True, "fa4": False, "fa2": True}) == "fa2"   # FA3 is never used on Blackwell


def test_fails_clearly_when_nothing_fits():
    msg = raises(RuntimeError, A.choose, 10, NONE, None, "NVIDIA B200")
    assert "NVIDIA B200" in msg and "flash-attn-4" in msg
    msg = raises(RuntimeError, A.choose, 10, {"fa3": True, "fa4": False, "fa2": False})
    assert "flash-attn-4" in msg
    assert "kernels" in raises(RuntimeError, A.choose, 9, NONE)


def test_override():
    assert A.choose(9, ALL, "fa4") == "fa4"
    assert A.choose(10, ALL, "FA2") == "fa2"
    assert A.choose(10, ALL, "auto") == "fa4"
    assert "does not support" in raises(RuntimeError, A.choose, 10, ALL, "fa3", "NVIDIA B200")
    assert "not installed" in raises(RuntimeError, A.choose, 9, {"fa3": True, "fa4": False, "fa2": True}, "fa4")
    raises(ValueError, A.choose, 9, ALL, "sdpa")


def test_unsupported_operations_are_refused():
    raises(NotImplementedError, A._only_plain, "fa4", 0.1, (-1, -1))
    raises(NotImplementedError, A._only_plain, "fa4", 0.0, (8, 8))
    raises(NotImplementedError, A._only_plain, "fa4", 0.0, (-1, -1), True)
    A._only_plain("fa3", 0.0, (-1, -1))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
