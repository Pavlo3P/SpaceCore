import pytest

from spacecore import checked_method


class _RecordingSpace:
    def __init__(self, valid):
        self.valid = valid
        self.calls = []

    def _check_member(self, value):
        self.calls.append(value)
        if value != self.valid:
            raise ValueError(f"invalid member: {value!r}")


class _CheckedDemo:
    def __init__(self, enable_checks=True):
        self._enable_checks = enable_checks
        self.dom = _RecordingSpace("x")
        self.cod = _RecordingSpace("y")
        self.space = _RecordingSpace("z")
        self.apply_result = "y"
        self.rapply_result = "x"
        self.value_result = 1.0
        self.grad_result = "z"

    @checked_method(in_space="dom", out_space="cod")
    def apply(self, x):
        """Apply docstring."""
        return self.apply_result

    @checked_method(in_space="cod", out_space="dom")
    def rapply(self, y):
        return self.rapply_result

    @checked_method(in_space="space")
    def value(self, x):
        return self.value_result

    @checked_method(in_space="space", out_space="space")
    def grad(self, x):
        return self.grad_result


def test_checked_method_validates_apply_input_and_output():
    demo = _CheckedDemo()

    assert demo.apply("x") == "y"
    assert demo.dom.calls == ["x"]
    assert demo.cod.calls == ["y"]


def test_checked_method_validates_rapply_input_and_output():
    demo = _CheckedDemo()

    assert demo.rapply("y") == "x"
    assert demo.cod.calls == ["y"]
    assert demo.dom.calls == ["x"]


def test_checked_method_validates_value_input():
    demo = _CheckedDemo()

    assert demo.value("z") == 1.0
    assert demo.space.calls == ["z"]


def test_checked_method_validates_grad_input_and_output():
    demo = _CheckedDemo()

    assert demo.grad("z") == "z"
    assert demo.space.calls == ["z", "z"]


def test_checked_method_invalid_input_raises_when_enabled():
    demo = _CheckedDemo(enable_checks=True)

    with pytest.raises(ValueError, match="invalid member"):
        demo.apply("bad")


def test_checked_method_invalid_output_raises_when_enabled():
    demo = _CheckedDemo(enable_checks=True)
    demo.apply_result = "bad"

    with pytest.raises(ValueError, match="invalid member"):
        demo.apply("x")


def test_checked_method_skips_checks_when_disabled():
    demo = _CheckedDemo(enable_checks=False)
    demo.apply_result = "bad"

    assert demo.apply("bad") == "bad"
    assert demo.dom.calls == []
    assert demo.cod.calls == []


def test_checked_method_preserves_metadata():
    assert _CheckedDemo.apply.__name__ == "apply"
    assert _CheckedDemo.apply.__doc__ == "Apply docstring."
    assert _CheckedDemo.apply.__wrapped__ is not None
