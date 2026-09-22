import json
import time

import pytest

from tts_cache import LocalStorage, SqliteIndex, TTSCache
from tts_cache.cli import main, parse_duration


@pytest.fixture
def populated(tmp_path, monkeypatch):
    """A cache on disk with three clips, and the flags the CLI needs to find it."""
    cache = TTSCache(
        storage=LocalStorage(tmp_path / "audio"),
        index=SqliteIndex(tmp_path / "index.db"),
        provider="elevenlabs",
        voice_id="rachel",
        background_writes=False,
    )
    cache.get_or_generate_sync("one", generator_fn=lambda: b"a" * 1000)
    cache.get_or_generate_sync("two", generator_fn=lambda: b"b" * 2000)
    cache.get_or_generate_sync("three", generator_fn=lambda: b"c" * 3000, voice_id="adam")
    cache.get_sync("one")  # one recorded play
    cache.close()

    monkeypatch.setenv("TTS_CACHE_STORAGE", f"local:{tmp_path / 'audio'}")
    monkeypatch.setenv("TTS_CACHE_INDEX", f"sqlite:{tmp_path / 'index.db'}")
    return tmp_path


def run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr()


def test_stats_reports_what_is_cached(populated, capsys):
    code, out = run(capsys, "--json", "stats")

    assert code == 0
    result = json.loads(out.out)
    assert result["clips"] == 3
    assert result["bytes"] == 6000
    assert result["served_from_cache"] == 1
    assert result["voices"]["elevenlabs/rachel"]["clips"] == 2


def test_stats_prints_a_readable_summary(populated, capsys):
    code, out = run(capsys, "stats")

    assert code == 0
    assert "3 clips" in out.out
    assert "elevenlabs/rachel" in out.out


def test_ls_lists_entries_newest_use_first(populated, capsys):
    code, out = run(capsys, "ls")

    assert code == 0
    assert out.out.count("\n") >= 4
    assert "one" in out.out


def test_ls_filters_by_voice(populated, capsys):
    _code, out = run(capsys, "--json", "ls", "--voice-id", "adam")

    entries = json.loads(out.out)["entries"]
    assert [e["text"] for e in entries] == ["three"]


def test_prune_evicts_down_to_the_target(populated, capsys):
    code, out = run(capsys, "--json", "prune", "--target-size-mb", str(3000 / (1024 * 1024)))

    result = json.loads(out.out)
    assert code == 0
    assert result["evicted"] >= 1
    assert result["bytes"] <= 3000


def test_purge_removes_everything_older_than_a_duration(populated, capsys):
    code, out = run(capsys, "--json", "purge", "--older-than", "0s")

    assert code == 0
    assert json.loads(out.out)["purged"] == 3


def test_purge_without_a_ttl_explains_itself(populated, capsys):
    code, out = run(capsys, "purge", "--expired")

    assert code == 1
    assert "TTS_CACHE_TTL" in out.err


def test_purge_expired_reads_the_ttl_from_the_environment(populated, capsys, monkeypatch):
    monkeypatch.setenv("TTS_CACHE_TTL", "1h")
    code, out = run(capsys, "--json", "purge", "--expired")

    assert code == 0
    assert json.loads(out.out)["purged"] == 0, "nothing is an hour old yet"


def test_delete_by_voice_and_by_key(populated, capsys):
    _code, out = run(capsys, "--json", "delete", "--voice-id", "adam")
    assert json.loads(out.out)["deleted"] == 1

    key = json.loads(run(capsys, "--json", "ls")[1].out)["entries"][0]["key"]
    _code, out = run(capsys, "--json", "delete", "--key", key)
    assert json.loads(out.out)["deleted"] == 1

    assert json.loads(run(capsys, "--json", "stats")[1].out)["clips"] == 1


def test_delete_needs_something_to_match_on(populated, capsys):
    code, out = run(capsys, "delete")
    assert code == 1
    assert "--key" in out.err


def test_flush_needs_confirming(populated, capsys):
    code, _out = run(capsys, "flush")
    assert code == 1

    code, _out = run(capsys, "flush", "--all")
    assert code == 0
    assert json.loads(run(capsys, "--json", "stats")[1].out)["clips"] == 0


def test_deleting_a_clip_removes_its_audio_too(populated, capsys):
    run(capsys, "flush", "--all")
    assert not list((populated / "audio").rglob("*.mp3"))


def test_an_unknown_storage_is_refused(capsys, monkeypatch):
    monkeypatch.setenv("TTS_CACHE_STORAGE", "dropbox:/clips")
    code, out = run(capsys, "stats")

    assert code == 2
    assert "unknown storage" in out.err


@pytest.mark.parametrize(
    ("text", "seconds"), [("900", 900), ("30m", 1800), ("14d", 1209600), ("2w", 1209600)]
)
def test_durations_are_read_the_way_a_cron_job_writes_them(text, seconds):
    assert parse_duration(text) == seconds


def test_a_nonsense_duration_is_refused():
    with pytest.raises(Exception, match="not a duration"):
        parse_duration("soon")


def test_timestamps_survive_the_round_trip(populated, capsys):
    entries = json.loads(run(capsys, "--json", "ls")[1].out)["entries"]
    assert all(time.time() - e["created_at"] < 60 for e in entries)
