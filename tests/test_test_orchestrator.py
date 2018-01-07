import pytest

from mleap.experiments.test_orchestrator import TestOrchestrator

def test_inputs():
    test_toch = TestOrchestrator(hdf5_input_io =4, hdf5_output_io=3)