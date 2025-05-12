import pytest
import time
from mini_bdx_runtime.projector import Projector
projector_delay = 1

def test_projector_switch(projector):
    """Test projector switching on and off"""
    # Initial state should be off
    assert projector.on == False
    
    # Switch on
    projector.switch()
    assert projector.on == True
    time.sleep(projector_delay)  # Allow time for GPIO to update
    
    # Switch off
    projector.switch()
    assert projector.on == False
    time.sleep(projector_delay)  # Allow time for GPIO to update

