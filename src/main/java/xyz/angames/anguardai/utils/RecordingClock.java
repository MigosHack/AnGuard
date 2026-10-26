package xyz.angames.anguardai.utils;

/** Monotonic wall-time countdown, independent of server tick rate. Zero duration means untimed. */
public final class RecordingClock {
    private final long started;
    private final long duration;

    public RecordingClock(long startedNanos, long seconds) {
        if (seconds < 0 || seconds > RecordingDuration.MAX_SECONDS) throw new IllegalArgumentException("Invalid duration");
        started = startedNanos;
        duration = seconds * 1_000_000_000L;
    }

    public boolean timed() { return duration > 0; }
    private long remainingNanos(long now) { return Math.max(0, duration - Math.max(0, now - started)); }
    public long remainingSeconds(long now) { return (remainingNanos(now) + 999_999_999L) / 1_000_000_000L; }
    public boolean expired(long now) { return timed() && remainingNanos(now) == 0; }
    public float progress(long now) { return timed() ? (float) ((double) remainingNanos(now) / duration) : 1f; }
}
