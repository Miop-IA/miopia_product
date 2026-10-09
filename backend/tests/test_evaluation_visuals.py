import os
import tempfile
import pytest
import pandas as pd
from unittest import mock
import numpy as np

import sys
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from train.train import gerar_dados_sinteticos_para_teste
from scripts.evaluate_model import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_pr_roc_curves,
    plot_calibration_curve
)

def test_visualizations_generated_successfully():
    """
    Testa se as 4 funções básicas geram as imagens e não quebram.
    """
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.3, 0.7, 0.85, 0.15, 0.95])
    
    meta_info = "Model: test-v1 | Pipeline: v1.0"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        p1 = os.path.join(temp_dir, "01_class_dist.png")
        plot_class_distribution(y_true, p1, meta_info)
        assert os.path.exists(p1)
        
        p2 = os.path.join(temp_dir, "02_cm.png")
        plot_confusion_matrix(y_true, y_pred, p2, meta_info)
        assert os.path.exists(p2)
        
        p3 = os.path.join(temp_dir, "03_roc.png")
        plot_pr_roc_curves(y_true, y_prob, p3, meta_info)
        assert os.path.exists(p3)
        
        p4 = os.path.join(temp_dir, "04_calib.png")
        plot_calibration_curve(y_true, y_prob, p4, meta_info)
        assert os.path.exists(p4)

def test_evaluation_script_fails_gracefully_on_missing_data():
    """
    Verifica que a rotina principal (via argparse no sys.argv) 
    rejeita faltas de colunas vitais imediatamente.
    """
    from scripts.evaluate_model import main
    import argparse

    with tempfile.TemporaryDirectory() as temp_dir:
        df_invalid = pd.DataFrame({"texto_cru": ["Olá mundo"]}) # Missing 'target'
        csv_path = os.path.join(temp_dir, "dataset.csv")
        df_invalid.to_csv(csv_path, index=False)
        
        model_path = os.path.join(temp_dir, "fake_model.joblib")
        with open(model_path, "w") as f:
            f.write("fake") # Não vamos carregar, vai quebrar no CSV antes
            
        test_args = ["evaluate_model.py", "--model_path", model_path, "--dataset_path", csv_path, "--output_dir", temp_dir]
        
        with mock.patch("sys.argv", test_args):
            with pytest.raises(ValueError, match="O dataset deve conter as colunas 'target' e 'texto_cru'."):
                main()
