import pytest
import numpy as np
import time

def test_feet_contacts_initialization(feet_contacts):
    """Test that feet contacts are properly initialized"""
    assert feet_contacts is not None

def test_feet_contacts_get(feet_contacts):
    """Test getting feet contact states"""
    # Get contact states
    contacts = feet_contacts.get()
    
    # Check return type and shape
    assert isinstance(contacts, np.ndarray)
    assert contacts.shape == (2,)
    assert contacts.dtype == bool
    
    # Check that values are boolean
    assert isinstance(contacts[0], bool)  # Left foot
    assert isinstance(contacts[1], bool)  # Right foot

def test_feet_contacts_continuous_reading(feet_contacts):
    """Test continuous reading of feet contacts"""
    # Take multiple readings to ensure stability
    readings = []
    for _ in range(5):
        readings.append(feet_contacts.get())
        time.sleep(0.1)
    
    # Convert readings to numpy array for analysis
    readings = np.array(readings)
    assert readings.shape == (5, 2)

def test_feet_contacts_response_time(feet_contacts):
    """Test response time of feet contacts"""
    start_time = time.time()
    feet_contacts.get()
    end_time = time.time()
    
    # Reading should be very quick (under 10ms)
    assert end_time - start_time < 0.01 