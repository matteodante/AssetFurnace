import os

import pytest

from image_to_3dlab import gpu_lock


def test_the_lock_names_this_process_while_held_and_is_gone_after(tmp_path):
    path = tmp_path / "gpu.lock"
    with gpu_lock.gpu("materials", path=path):
        assert path.read_text(encoding="utf-8") == f"{os.getpid()} materials"
    assert not path.exists()


def test_the_lock_is_released_when_the_job_fails(tmp_path):
    path = tmp_path / "gpu.lock"
    with pytest.raises(RuntimeError), gpu_lock.gpu("materials", path=path):
        raise RuntimeError("out of memory")
    assert not path.exists()


def test_a_live_holder_is_waited_for_then_reported_busy(tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text("4242 reference-concept", encoding="utf-8")
    naps = []
    clock = iter(range(0, 100, 10))
    with pytest.raises(gpu_lock.GpuBusy, match="reference-concept"),             gpu_lock.gpu("materials", path=path, wait=25, poll=10, alive=lambda pid: True,
                         sleep=naps.append, clock=lambda: next(clock)):
        pass
    assert naps == [10, 10]
    assert path.read_text(encoding="utf-8") == "4242 reference-concept"


def test_the_lock_of_a_dead_process_is_taken_over(tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text("4242 reference-concept", encoding="utf-8")
    with gpu_lock.gpu("materials", path=path, alive=lambda pid: False):
        assert path.read_text(encoding="utf-8").endswith(" materials")


def test_someone_elses_lock_is_never_removed_on_the_way_out(tmp_path):
    # Another session overwrote the lock while this one computed: leave theirs alone.
    path = tmp_path / "gpu.lock"
    with gpu_lock.gpu("materials", path=path):
        path.write_text("777 other", encoding="utf-8")
    assert path.read_text(encoding="utf-8") == "777 other"


def test_an_unreadable_lock_is_read_as_held_by_nobody_known(tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text("garbage", encoding="utf-8")
    assert gpu_lock.holder(path) == (None, "garbage")


def test_this_process_is_alive():
    assert gpu_lock.pid_alive(os.getpid())
