# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Jorge Marco Blanco
# Distributed under the GNU GPL version 3; see LICENSE.

import os
import numpy as np
from aeon.datasets import load_classification

# Standard UCR 112 dataset list
UCR_112 = [
    'ACSF1', 'Adiac', 'ArrowHead', 'BME', 'Beef', 'BeetleFly',
    'BirdChicken', 'CBF', 'Car', 'Chinatown', 'ChlorineConcentration',
    'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY',
    'CricketZ', 'Crop', 'DiatomSizeReduction',
    'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
    'DistalPhalanxTW', 'ECG200', 'ECG5000', 'ECGFiveDays',
    'EOGHorizontalSignal', 'EOGVerticalSignal', 'Earthquakes',
    'ElectricDevices', 'EthanolLevel', 'FaceAll', 'FaceFour',
    'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB',
    'FreezerRegularTrain', 'FreezerSmallTrain', 'GunPoint',
    'GunPointAgeSpan', 'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung', 'Ham', 'HandOutlines', 'Haptics',
    'Herring', 'HouseTwenty', 'InlineSkate', 'InsectEPGRegularTrain',
    'InsectEPGSmallTrain', 'InsectWingbeatSound', 'ItalyPowerDemand',
    'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat',
    'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup',
    'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
    'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 'MoteStrain',
    'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
    'OSULeaf', 'OliveOil', 'PhalangesOutlinesCorrect', 'Phoneme',
    'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'Plane',
    'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
    'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2',
    'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShapeletSim',
    'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace',
    'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2',
    'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols',
    'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2',
    'Trace', 'TwoLeadECG', 'TwoPatterns', 'UMD',
    'UWaveGestureLibraryAll', 'UWaveGestureLibraryX',
    'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine',
    'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'
]

class DataManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def _resolve_dataset_list(self, requested_datasets):
        """Internal helper to parse the config input into a strict list of strings."""
        if isinstance(requested_datasets, str):
            requested_datasets = [requested_datasets]
            
        # Check if "all" was requested (case-insensitive)
        if any(d.lower() == "all" for d in requested_datasets):
            return UCR_112
        
        return requested_datasets

    def yield_datasets(self, requested_datasets):
        """
        Generator that resolves the dataset list, handles loading, 
        catches errors safely, and yields the valid arrays.
        """
        dataset_names = self._resolve_dataset_list(requested_datasets)
        
        for dataset_name in dataset_names:
            try:
                # Load or download via aeon
                X_train, y_train = load_classification(
                    dataset_name, split="train", extract_path=self.data_dir
                )
                X_test, y_test = load_classification(
                    dataset_name, split="test", extract_path=self.data_dir
                )
                
                # Ensure strictly 3D arrays: (n_instances, n_channels, n_timepoints)
                if X_train.ndim == 2:
                    X_train = X_train[:, np.newaxis, :]
                    X_test = X_test[:, np.newaxis, :]
                    
                # Yield the successful payload
                yield dataset_name, X_train, y_train, X_test, y_test
                
            except Exception as e:
                # Catch failures internally so the experiment loop never breaks
                print(f"  [!] Failed to load {dataset_name}: {e}")
                continue