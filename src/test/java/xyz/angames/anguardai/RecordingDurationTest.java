package xyz.angames.anguardai;

import org.junit.jupiter.api.Test;
import xyz.angames.anguardai.utils.RecordingClock;
import xyz.angames.anguardai.utils.RecordingDuration;
import static org.junit.jupiter.api.Assertions.*;

class RecordingDurationTest {
    @Test void acceptsMinutesClockAndUnits() {
        assertEquals(900, RecordingDuration.parse("15"));
        assertEquals(900, RecordingDuration.parse("15:00"));
        assertEquals(30, RecordingDuration.parse("0:30"));
        assertEquals(30, RecordingDuration.parse("30s"));
        assertEquals(900, RecordingDuration.parse("15m"));
        assertEquals(3600, RecordingDuration.parse("1H"));
        assertEquals(86400, RecordingDuration.parse("24h"));
        assertEquals("15:00", RecordingDuration.format(900));
    }
    @Test void rejectsZeroMalformedAndOverflowingTimes() {
        for (String value : new String[]{"", "0", "0:00", "-1", "15:60", "15:0", "1.5", "25h", "999999999999999999999", "99999999999999999h"}) {
            assertThrows(IllegalArgumentException.class, () -> RecordingDuration.parse(value), value);
        }
    }
    @Test void usesMonotonicElapsedTimeAndCeilingForRemainingSeconds() {
        RecordingClock timer = new RecordingClock(-1000, 900);
        assertEquals(900, timer.remainingSeconds(-1000));
        assertEquals(900, timer.remainingSeconds(-999));
        assertEquals(1, timer.remainingSeconds(900_000_000_000L - 1001));
        assertTrue(timer.expired(900_000_000_000L - 1000));
        assertEquals(.5f, timer.progress(450_000_000_000L - 1000));
        assertEquals(0, timer.progress(Long.MAX_VALUE / 2));
        RecordingClock untimed = new RecordingClock(0, 0);
        assertFalse(untimed.timed());
        assertFalse(untimed.expired(1_000_000_000_000L));
    }
}
