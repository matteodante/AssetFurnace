import os
from pathlib import Path

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


# ---- a lock nobody can read, a reused pid, two takers at once ------------------------

def _aged(path, seconds):
    old = path.stat().st_mtime - seconds
    os.utime(path, (old, old))


@pytest.mark.parametrize("content", ["", "garbage"])
def test_an_unreadable_lock_left_for_a_minute_is_taken_over(tmp_path, content):
    # 2026-10-17 audit: a writer that died between creating the file and writing its pid
    # left an empty lock, and every GPU job then waited an hour for nobody
    path = tmp_path / "gpu.lock"
    path.write_text(content, encoding="utf-8")
    _aged(path, gpu_lock.UNREADABLE_AFTER + 5)
    naps = []
    with gpu_lock.gpu("materials", path=path, sleep=naps.append):
        assert path.read_text(encoding="utf-8") == f"{os.getpid()} materials"
    assert naps == []


def test_a_fresh_empty_lock_is_someone_still_writing_it(tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text("", encoding="utf-8")
    clock = iter(range(0, 100, 10))
    with pytest.raises(gpu_lock.GpuBusy), gpu_lock.gpu(
            "materials", path=path, wait=15, poll=10, sleep=lambda s: None, clock=lambda: next(clock)):
        pass
    assert path.exists()


def test_a_live_pid_that_started_after_the_lock_was_written_is_a_reused_number(tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text("4242 reference-concept", encoding="utf-8")
    written = path.stat().st_mtime
    with gpu_lock.gpu("materials", path=path, alive=lambda pid: True, started=lambda pid: written + 600,
                      sleep=lambda s: None):
        assert path.read_text(encoding="utf-8").endswith(" materials")


def test_a_live_pid_older_than_its_lock_is_still_the_holder(tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text("4242 reference-concept", encoding="utf-8")
    written = path.stat().st_mtime
    clock = iter(range(0, 100, 10))
    with pytest.raises(gpu_lock.GpuBusy), gpu_lock.gpu(
            "materials", path=path, wait=15, poll=10, alive=lambda pid: True,
            started=lambda pid: written - 600, sleep=lambda s: None, clock=lambda: next(clock)):
        pass


def test_the_lock_never_exists_without_its_holder_written_in_it(tmp_path, monkeypatch):
    path = tmp_path / "gpu.lock"
    seen = []
    real_link = os.link

    def watching(src, dst, *a, **k):
        real_link(src, dst, *a, **k)
        seen.append(Path(dst).read_text(encoding="utf-8"))
    monkeypatch.setattr(gpu_lock.os, "link", watching)
    with gpu_lock.gpu("materials", path=path):
        pass
    assert seen == [f"{os.getpid()} materials"]
    assert list(tmp_path.iterdir()) == []      # no temporary file left behind


def test_a_stale_lock_taken_over_by_someone_else_meanwhile_is_given_back(tmp_path, monkeypatch):
    # two jobs see the same dead lock; the first takes it over before the second removes
    # it, and the second must not delete the first one's fresh lock
    path = tmp_path / "gpu.lock"
    path.write_text("4242 dead", encoding="utf-8")
    real_rename = os.rename

    def overtaken(src, dst):
        if str(src) == str(path):
            path.write_text("777 first", encoding="utf-8")   # the first job got there first
        real_rename(src, dst)
    monkeypatch.setattr(gpu_lock.os, "rename", overtaken)
    clock = iter(range(0, 100, 10))
    with pytest.raises(gpu_lock.GpuBusy, match="777 first"), gpu_lock.gpu(
            "second", path=path, wait=15, poll=10, alive=lambda pid: pid == 777,
            sleep=lambda s: None, clock=lambda: next(clock)):
        pass
    assert path.read_text(encoding="utf-8") == "777 first"


def test_of_many_takers_at_once_exactly_one_holds_the_lock(tmp_path):
    import threading

    path = tmp_path / "gpu.lock"
    held, inside, barrier = [], [], threading.Barrier(8)

    def take(n):
        barrier.wait()
        with gpu_lock.gpu(f"t{n}", path=path, sleep=lambda s: None, poll=0):
            inside.append(n)
            held.append(len(inside))
            inside.remove(n)

    threads = [threading.Thread(target=take, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(held) == 8 and max(held) == 1
    assert not path.exists()


def test_this_process_started_before_now():
    import time

    started = gpu_lock.process_started(os.getpid())
    if started is not None:
        assert started <= time.time()
