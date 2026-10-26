package xyz.angames.anguardai.utils;

import java.util.Locale;

public final class RecordingDuration {
    public static final long MAX_SECONDS = 24 * 60 * 60;
    private RecordingDuration() {}

    /** Bare numbers are minutes; also accepts MM:SS and explicit s/m/h suffixes. */
    public static long parse(String input) {
        String value = input.toLowerCase(Locale.ROOT);
        try {
            long seconds;
            if (value.matches("[0-9]+:[0-5][0-9]")) {
                String[] parts = value.split(":");
                seconds = Math.addExact(Math.multiplyExact(Long.parseLong(parts[0]), 60), Long.parseLong(parts[1]));
            } else if (value.matches("[0-9]+[smh]?")) {
                char last = value.charAt(value.length() - 1);
                boolean suffix = Character.isLetter(last);
                long number = Long.parseLong(suffix ? value.substring(0, value.length() - 1) : value);
                long multiplier = last == 's' ? 1 : last == 'h' ? 3600 : 60;
                seconds = Math.multiplyExact(number, multiplier);
            } else {
                throw new IllegalArgumentException("Invalid recording duration");
            }
            if (seconds <= 0 || seconds > MAX_SECONDS) throw new IllegalArgumentException("Duration out of bounds");
            return seconds;
        } catch (ArithmeticException error) {
            throw new IllegalArgumentException("Duration out of bounds", error);
        }
    }

    public static String format(long seconds) {
        return String.format(Locale.ROOT, "%02d:%02d", Math.max(0, seconds) / 60, Math.max(0, seconds) % 60);
    }
}
