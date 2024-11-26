from __future__ import annotations

from qadence.types import StrEnum


class TrainingStage(StrEnum):
    """Different stages in the training, validation, and testing process."""

    IDLE = "idle"
    """An 'idle' stage for scenarios where no training, validation, or testing is involved."""

    TRAIN_START = "train_start"
    """Marks the start of the training process."""

    TRAIN_END = "train_end"
    """Marks the end of the training process."""

    TRAIN_EPOCH_START = "train_epoch_start"
    """Indicates the start of a training epoch."""

    TRAIN_EPOCH_END = "train_epoch_end"
    """Indicates the end of a training epoch."""

    TRAIN_BATCH_START = "train_batch_start"
    """Marks the start of processing a training batch."""

    TRAIN_BATCH_END = "train_batch_end"
    """Marks the end of processing a training batch."""

    VAL_EPOCH_START = "val_epoch_start"
    """Indicates the start of a validation epoch."""

    VAL_EPOCH_END = "val_epoch_end"
    """Indicates the end of a validation epoch."""

    VAL_BATCH_START = "val_batch_start"
    """Marks the start of processing a validation batch."""

    VAL_BATCH_END = "val_batch_end"
    """Marks the end of processing a validation batch."""

    TEST_BATCH_START = "test_batch_start"
    """Marks the start of processing a test batch."""

    TEST_BATCH_END = "test_batch_end"
    """Marks the end of processing a test batch."""
