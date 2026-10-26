package xyz.angames.anguardai.data;

import java.util.UUID;

/** Label-first CSV, with grouping metadata retained for leakage-free training. */
public final class TrainingCsv {
    public static final String[] COLUMNS = {
            "delta_yaw",
            "delta_pitch",
            "accel_yaw",
            "accel_pitch",
            "jerk_pitch",
            "jerk_yaw",
            "gcd_error_yaw",
            "gcd_error_pitch",
            "abs_delta_yaw",
            "abs_delta_pitch",
            "rotation_magnitude",
            "abs_accel_yaw",
            "abs_accel_pitch",
            "acceleration_magnitude",
            "abs_jerk_yaw",
            "abs_jerk_pitch",
            "jerk_magnitude",
            "yaw_direction_fraction",
            "pitch_direction_fraction",
            "yaw_direction_reversal",
            "pitch_direction_reversal",
            "signed_yaw_change",
            "signed_pitch_change",
            "rotation_magnitude_change",
            "delta_yaw_rolling20_mean",
            "delta_yaw_rolling20_std",
            "delta_yaw_rolling20_min",
            "delta_yaw_rolling20_max",
            "delta_pitch_rolling20_mean",
            "delta_pitch_rolling20_std",
            "delta_pitch_rolling20_min",
            "delta_pitch_rolling20_max",
            "accel_yaw_rolling20_mean",
            "accel_yaw_rolling20_std",
            "accel_yaw_rolling20_min",
            "accel_yaw_rolling20_max",
            "accel_pitch_rolling20_mean",
            "accel_pitch_rolling20_std",
            "accel_pitch_rolling20_min",
            "accel_pitch_rolling20_max",
            "gcd_error_yaw_rolling20_mean",
            "gcd_error_yaw_rolling20_std",
            "gcd_error_yaw_rolling20_min",
            "gcd_error_yaw_rolling20_max",
            "gcd_error_pitch_rolling20_mean",
            "gcd_error_pitch_rolling20_std",
            "gcd_error_pitch_rolling20_min",
            "gcd_error_pitch_rolling20_max"
    };
    private TrainingCsv() {}

    public static String header() {
        return "label," + String.join(",", COLUMNS)
                + ",player_id,session_id,window_id,frame_index,timestamp_ms,schema";
    }

    public static String row(int label, float[] values, UUID player, UUID session,
                             UUID window, int frame, long timestamp) {
        if ((label != 0 && label != 1) || values.length != MovementFeatures.COUNT
                || frame < 0 || frame >= MovementFeatures.WINDOW) {
            throw new IllegalArgumentException("Invalid labeled movement sample");
        }
        StringBuilder line = new StringBuilder(768).append(label);
        for (float value : values) {
            if (!Float.isFinite(value)) throw new IllegalArgumentException("Non-finite CSV feature");
            // Float.toString is locale-independent, uses a decimal point and round-trips float32.
            line.append(',').append(Float.toString(value));
        }
        return line.append(',').append(player).append(',').append(session).append(',').append(window)
                .append(',').append(frame).append(',').append(timestamp).append(',')
                .append(MovementFeatures.SCHEMA).toString();
    }
}
