import pytest

from jssp_openenv.models import parse_job_ids


def test_parse_job_ids():
    assert parse_job_ids("1,2,3") == [1, 2, 3]
    assert parse_job_ids("3,2,1") == [3, 2, 1]
    assert parse_job_ids("") == []
    assert parse_job_ids(",") == []
    assert parse_job_ids("0,") == [0]

    with pytest.raises(ValueError):
        parse_job_ids("1,2,3,a")

    with pytest.raises(ValueError):
        parse_job_ids("0.1")
