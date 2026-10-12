# SVM Classification for EEG Visual Evoked Potentials

## Dataset: 4 Classes (Apple, Car, Flower, Human Face)
## Method: Stratified 5-Fold Cross-Validation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")
