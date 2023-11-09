from __future__ import annotations


class QadenceException(Exception):
    pass


class NotSupportedError(QadenceException):
    pass


class NotPauliBlockError(QadenceException):
    pass
