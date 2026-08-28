import subprocess


def test_dvc_pipeline():
    result = subprocess.run(["dvc", "status"], capture_output=True)
    assert result.returncode == 0
