#!/usr/bin/env python3
"""
IMU Calibration Test Script
Tests if the BNO055 IMU is properly calibrated.
"""

import numpy as np
import time
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mini_bdx_runtime.raw_imu import Imu


def test_static(imu, duration=3.0, samples=50):
    """Test IMU when robot is still and upright."""
    print("\n" + "="*50)
    print("STATIC TEST - Keep robot STILL and UPRIGHT")
    print("="*50)
    time.sleep(1)
    
    gyro_samples = []
    accel_samples = []
    
    for i in range(samples):
        data = imu.get_data()
        gyro_samples.append(data['gyro'])
        accel_samples.append(data['accelero'])
        time.sleep(duration / samples)
    
    gyro_mean = np.mean(gyro_samples, axis=0)
    gyro_std = np.std(gyro_samples, axis=0)
    accel_mean = np.mean(accel_samples, axis=0)
    accel_std = np.std(accel_samples, axis=0)
    accel_magnitude = np.linalg.norm(accel_mean)
    
    print(f"\nGyro mean:  {np.around(gyro_mean, 4)} rad/s")
    print(f"Gyro std:   {np.around(gyro_std, 4)} rad/s")
    print(f"Accel mean: {np.around(accel_mean, 3)} m/s²")
    print(f"Accel std:  {np.around(accel_std, 3)} m/s²")
    print(f"Accel magnitude: {accel_magnitude:.3f} m/s² (expected: 9.81)")
    
    # Check results
    issues = []
    
    if np.any(np.abs(gyro_mean) > 0.1):
        issues.append(f"⚠️  Gyro drift detected: {gyro_mean}")
    else:
        print("✅ Gyro bias: OK (< 0.1 rad/s)")
    
    if np.any(gyro_std > 0.05):
        issues.append(f"⚠️  Gyro noise high: std={gyro_std}")
    else:
        print("✅ Gyro noise: OK")
    
    if abs(accel_magnitude - 9.81) > 0.5:
        issues.append(f"⚠️  Accel magnitude off: {accel_magnitude:.2f} (expected ~9.81)")
    else:
        print("✅ Accel magnitude: OK")
    
    if abs(accel_mean[2]) < 8.0:
        issues.append(f"⚠️  Z-accel too low when upright: {accel_mean[2]:.2f}")
    else:
        print("✅ Z-axis dominance: OK (upright orientation detected)")
    
    if np.any(np.abs(accel_mean[:2]) > 1.5):
        issues.append(f"⚠️  X/Y accel bias when level: {accel_mean[:2]}")
    else:
        print("✅ X/Y accel bias: OK")
    
    return issues


def test_gyro_response(imu):
    """Test gyro responds to rotation."""
    print("\n" + "="*50)
    print("GYRO RESPONSE TEST")
    print("="*50)
    print("Rotate the robot around each axis when prompted...")
    time.sleep(1)
    
    axes = ['X (roll)', 'Y (pitch)', 'Z (yaw)']
    issues = []
    
    for i, axis in enumerate(axes):
        input(f"\nPress Enter, then ROTATE around {axis} axis...")
        
        max_rate = 0
        for _ in range(30):
            data = imu.get_data()
            rate = abs(data['gyro'][i])
            max_rate = max(max_rate, rate)
            time.sleep(0.05)
        
        if max_rate > 0.5:
            print(f"✅ {axis}: Detected rotation ({max_rate:.2f} rad/s)")
        else:
            issues.append(f"⚠️  {axis}: No rotation detected (max={max_rate:.2f})")
            print(f"⚠️  {axis}: No rotation detected!")
    
    return issues


def check_calibration_file():
    """Check if calibration file exists."""
    print("\n" + "="*50)
    print("CALIBRATION FILE CHECK")
    print("="*50)
    
    calib_path = "imu_calib_data.pkl"
    if os.path.exists(calib_path):
        import pickle
        data = pickle.load(open(calib_path, 'rb'))
        print(f"✅ Calibration file found: {calib_path}")
        print(f"   Accel offsets: {data.get('offsets_accelerometer', 'N/A')}")
        print(f"   Gyro offsets:  {data.get('offsets_gyroscope', 'N/A')}")
        print(f"   Mag offsets:   {data.get('offsets_magnetometer', 'N/A')}")
        return []
    else:
        print(f"⚠️  No calibration file found at {calib_path}")
        print("   Run: python calibrate_imu.py")
        return ["No calibration file - IMU is uncalibrated"]


def main():
    print("="*50)
    print("   IMU CALIBRATION TEST")
    print("="*50)
    
    all_issues = []
    
    # Check calibration file
    all_issues.extend(check_calibration_file())
    
    # Initialize IMU
    print("\nInitializing IMU...")
    try:
        imu = Imu(
            sampling_freq=50,
            calibrate=False,
            upside_down=False,  # Adjust if your IMU is mounted upside down
            enable_spike_filter=True,
        )
    except Exception as e:
        print(f"❌ Failed to initialize IMU: {e}")
        return
    
    time.sleep(0.5)
    
    # Run tests
    all_issues.extend(test_static(imu))
    
    run_gyro_test = input("\nRun gyro response test? (y/n): ").lower().strip() == 'y'
    if run_gyro_test:
        all_issues.extend(test_gyro_response(imu))
    
    # Summary
    print("\n" + "="*50)
    print("   SUMMARY")
    print("="*50)
    
    if not all_issues:
        print("✅ All tests passed! IMU appears properly calibrated.")
    else:
        print(f"⚠️  Found {len(all_issues)} issue(s):")
        for issue in all_issues:
            print(f"   {issue}")
        print("\nConsider running: python calibrate_imu.py")


if __name__ == "__main__":
    main()
