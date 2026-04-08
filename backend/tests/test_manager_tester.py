from manager import manager_tester


def test_run_once_from_labs():
    result = manager_tester.run_once(labs={"glucose": 150.0})
    assert result["status"] == "ok"
    assert result["diagnosis"]["findings"]


def test_parse_labs_from_json():
    labs = manager_tester.parse_labs('{"glucose": 120}', None)
    assert labs["glucose"] == 120


def test_parse_labs_from_file(tmp_path):
    file_path = tmp_path / "labs.json"
    file_path.write_text('{"hemoglobin": 11.2}')
    labs = manager_tester.parse_labs(None, str(file_path))
    assert labs["hemoglobin"] == 11.2
