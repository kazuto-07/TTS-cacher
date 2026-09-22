import pytest

from tts_cache import AudioSpec, cache_key, key_payload, normalize_text


def spec(text="Hello there.", **kwargs):
    base = {"provider": "elevenlabs", "voice_id": "rachel", "model": "turbo-v2"}
    base.update(kwargs)
    return AudioSpec(text=text, **base)


def test_whitespace_is_the_only_thing_normalisation_removes():
    assert normalize_text("  Hello   there.\n") == "Hello there."
    # Case and punctuation survive, because both change how a sentence is read.
    assert normalize_text("HELLO, there!") == "HELLO, there!"


def test_padding_and_case_are_not_the_same_key():
    assert cache_key(spec(" Hello there. ")) == cache_key(spec("Hello there."))
    assert cache_key(spec("hello there.")) != cache_key(spec("Hello there."))


def test_a_key_is_sixty_four_hex_characters():
    key = cache_key(spec())
    assert len(key) == 64
    assert set(key) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "openai"},
        {"voice_id": "adam"},
        {"model": "multilingual-v2"},
        {"audio_format": "wav"},
        {"settings": {"rate": 1}},
    ],
)
def test_every_part_of_the_voice_changes_the_key(change):
    assert cache_key(spec(**change)) != cache_key(spec())


def test_settings_order_does_not_change_the_key():
    a = spec(settings={"rate": 1, "pitch": -2, "style": "calm"})
    b = spec(settings={"style": "calm", "pitch": -2, "rate": 1})
    assert cache_key(a) == cache_key(b)


def test_a_whole_float_and_its_integer_are_one_entry():
    assert cache_key(spec(settings={"rate": 1.0})) == cache_key(spec(settings={"rate": 1}))
    assert cache_key(spec(settings={"rate": 1.5})) != cache_key(spec(settings={"rate": 1}))


def test_the_provider_name_is_case_insensitive_but_the_voice_is_not():
    assert cache_key(spec(provider="ElevenLabs")) == cache_key(spec(provider="elevenlabs"))
    assert cache_key(spec(voice_id="Rachel")) != cache_key(spec(voice_id="rachel"))


def test_the_payload_explains_a_surprising_miss():
    payload = key_payload(spec("  Hello   there. "))
    assert payload["text"] == "Hello there."
    assert payload["provider"] == "elevenlabs"
    assert payload["settings"] == {}


def test_a_custom_normaliser_is_honoured():
    key = cache_key(spec("Hello There."), normalizer=str.lower)
    assert key == cache_key(spec("hello there."), normalizer=str.lower)
