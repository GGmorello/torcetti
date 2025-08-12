_grad_enabled = True


def is_grad_enabled():
    return _grad_enabled


def set_grad_enabled(mode: bool):
    global _grad_enabled
    _grad_enabled = bool(mode)


class _GradModeContext:
    def __init__(self, mode: bool):
        self.mode = bool(mode)
        self.prev = None

    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = self.mode
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _grad_enabled
        _grad_enabled = self.prev
        # propagate exceptions, do not suppress
        return False


def no_grad():
    """Context manager that disables gradient tracking inside its scope."""
    return _GradModeContext(False)


def grad_enabled():
    """Context manager that enables gradient tracking inside its scope (opposite of no_grad)."""
    return _GradModeContext(True) 