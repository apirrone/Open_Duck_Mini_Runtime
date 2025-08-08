import pytest
import os
import time
import pygame
from unittest import mock
from conftest import get_assets_dir
from mini_bdx_runtime.sounds import Sounds

sound_delay = 0.1
sound_volume = 0.5

def test_sounds_initialization(sounds):
    """Test that sounds system is properly initialized"""
    assert sounds is not None
    assert sounds.ok == True
    assert len(sounds.sounds) > 0

def test_play_specific_sound(sounds):
    """Test playing a specific sound"""
    # Test playing happy sound
    sounds.play_happy()
    time.sleep(sound_delay)  # Allow time for sound to start

def test_play_random_sound(sounds):
    """Test playing random sounds"""
    # Test multiple random sounds
    for _ in range(3):
        sounds.play_random_sound()
        time.sleep(sound_delay)  # Allow time between sounds

def test_play_nonexistent_sound(sounds):
    """Test attempting to play a nonexistent sound"""
    # Should print error message but not crash
    sounds.play("nonexistent_sound.wav")

def test_sound_volume(sounds):
    """Test sound volume setting"""
    # Get the actual volume of the sound
    actual_volume = sounds.sounds["happy1.wav"].get_volume()
    # Use a small tolerance for floating point comparison
    assert abs(actual_volume - sound_volume) < 0.01, f"Expected volume {sound_volume}, got {actual_volume}"
    time.sleep(sound_delay)

def test_available_sounds(sounds):
    """Test that expected sound files are available"""
    # Check for common sound files
    assert "happy1.wav" in sounds.sounds
    
    # Verify all loaded sounds are .wav files
    for sound_name in sounds.sounds.keys():
        assert sound_name.endswith('.wav')

@pytest.mark.parametrize("test_volume", [0.0, 0.3, 0.7, 1.0])
def test_sound_volume_control(test_volume):
    """Test sound volume control for both music and sound effects"""
    # Create new sound instance with test volume
    assets_dir = get_assets_dir()
    sound_player = Sounds(volume=test_volume, sound_directory=assets_dir)
    
    # Test that mixer volume is set correctly
    actual_mixer_volume = pygame.mixer.music.get_volume()
    assert abs(actual_mixer_volume - test_volume) < 0.01, \
        f"Music volume {actual_mixer_volume} does not match expected {test_volume}"
    
    # Test that all loaded sound effects have correct volume
    for sound_name, sound in sound_player.sounds.items():
        actual_volume = sound.get_volume()
        assert abs(actual_volume - test_volume) < 0.01, \
            f"Sound {sound_name} volume {actual_volume} does not match expected {test_volume}"
    
    # Try playing a sound at this volume
    if "happy1.wav" in sound_player.sounds:
        sound_player.play("happy1.wav")
        time.sleep(sound_delay)  # Allow time to start playing
        assert sound_player.is_playing(), f"Sound should be playing at volume {test_volume}"

@pytest.mark.parametrize("invalid_volume", [-0.5, 1.5])
def test_volume_bounds(invalid_volume):
    """Test that volume is properly bounded between 0 and 1"""
    assets_dir = get_assets_dir()
    sound_player = Sounds(volume=invalid_volume, sound_directory=assets_dir)
    
    # Volume should be clamped between 0 and 1
    actual_volume = pygame.mixer.music.get_volume()
    assert 0 <= actual_volume <= 1, f"Volume {actual_volume} should be clamped between 0 and 1"
    
    # Check all sound effects are also properly bounded
    for sound_name, sound in sound_player.sounds.items():
        effect_volume = sound.get_volume()
        assert 0 <= effect_volume <= 1, \
            f"Sound effect {sound_name} volume {effect_volume} should be clamped between 0 and 1"

def test_no_overlap_play(monkeypatch):
    """Test that play() does not start a new sound while one is playing if wait_if_playing=True"""
    assets_dir = get_assets_dir()
    # Mock mixer and sound
    mock_mixer = mock.MagicMock()
    mock_sound = mock.MagicMock()
    mock_mixer.Sound.return_value = mock_sound
    # Simulate busy state for first call, then not busy
    busy_states_iter = iter([True, False])
    mock_mixer.get_busy.side_effect = lambda: next(busy_states_iter, False)
    # Patch os.listdir to return a fake .wav file
    with mock.patch.object(os, "listdir", return_value=["happy1.wav"]):
        sound_player = Sounds(volume=0.5, sound_directory=assets_dir, mixer=mock_mixer)
        result = sound_player.play("happy1.wav", wait_if_playing=True)
        assert result is True
        # Should have waited for busy to become False
        assert mock_mixer.get_busy.call_count >= 1
        mock_sound.play.assert_called_once()

def test_play_without_wait(monkeypatch):
    """Test that play() starts immediately if wait_if_playing=False, even if busy"""
    assets_dir = get_assets_dir()
    mock_mixer = mock.MagicMock()
    mock_sound = mock.MagicMock()
    mock_mixer.Sound.return_value = mock_sound
    mock_mixer.get_busy.return_value = True
    with mock.patch("os.listdir", return_value=["happy1.wav"]):
        sound_player = Sounds(volume=0.5, sound_directory=assets_dir, mixer=mock_mixer)
        result = sound_player.play("happy1.wav", wait_if_playing=False)
        assert result is True
        # Should not wait for busy to become False
        mock_sound.play.assert_called_once() 