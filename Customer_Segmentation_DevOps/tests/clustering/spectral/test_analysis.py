import logging
import sys
import os
from src.clustering.spectral.analysis import show_silscores

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', '..', '..')
sys.path.append(src_dir)



def test_show_silscores(capsys, caplog):
    """
    Test for show_silscores function.
    """
    caplog.set_level(logging.INFO)

    silscores = {'method1': 0.5, 'method2': 0.7}
    show_silscores(silscores)

    captured = capsys.readouterr()

    assert "method1" in captured.out
    assert "method2" in captured.out
    assert "0.5" in captured.out
    assert "0.7" in captured.out

    assert 'Displaying silhouette scores' in caplog.text
    assert 'Silhouette scores displayed' in caplog.text
