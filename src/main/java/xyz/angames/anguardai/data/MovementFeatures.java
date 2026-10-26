package xyz.angames.anguardai.data;

import java.util.List;

/** Causal features recomputed inside each window, so old eight-column recordings can be replayed. */
public final class MovementFeatures {
    public static final int SCHEMA = 2;
    public static final int COUNT = 48;
    public static final int WINDOW = 40;
    private static final int[] ROLLING_COLUMNS = {0, 1, 2, 3, 6, 7};

    private MovementFeatures() {}

    public static float[][] expand(List<TickDataPOJO> ticks) {
        double[][] base = new double[ticks.size()][8];
        for (int i = 0; i < ticks.size(); i++) {
            TickDataPOJO t = ticks.get(i);
            base[i] = new double[] {t.deltaYaw, t.deltaPitch, t.accelYaw, t.accelPitch,
                    t.jerkPitch, t.jerkYaw, t.gcdErrorYaw, t.gcdErrorPitch};
        }
        return expand(base);
    }

    public static float[][] expand(double[][] base) {
        if (base.length != WINDOW) throw new IllegalArgumentException("Expected 40 movement frames");
        for (double[] row : base) {
            if (row.length != 8) throw new IllegalArgumentException("Expected eight base columns");
            for (double value : row) {
                if (!Double.isFinite(value) || Math.abs(value) > 1e6) {
                    throw new IllegalArgumentException("Invalid movement value");
                }
            }
        }
        float[][] result = new float[WINDOW][COUNT];
        for (int i = 0; i < WINDOW; i++) {
            double[] r = base[i];
            double previousYaw = i == 0 ? r[0] : base[i - 1][0];
            double previousPitch = i == 0 ? r[1] : base[i - 1][1];
            double magnitude = Math.hypot(r[0], r[1]);
            double[] instant = {Math.abs(r[0]), Math.abs(r[1]), magnitude,
                    Math.abs(r[2]), Math.abs(r[3]), Math.hypot(r[2], r[3]),
                    Math.abs(r[5]), Math.abs(r[4]), Math.hypot(r[4], r[5]),
                    magnitude == 0 ? 0 : r[0] / magnitude, magnitude == 0 ? 0 : r[1] / magnitude,
                    r[0] * previousYaw < 0 ? 1 : 0, r[1] * previousPitch < 0 ? 1 : 0,
                    r[0] - previousYaw, r[1] - previousPitch,
                    magnitude - Math.hypot(previousYaw, previousPitch)};
            for (int j = 0; j < 8; j++) result[i][j] = (float) r[j];
            for (int j = 0; j < instant.length; j++) result[i][8 + j] = (float) instant[j];
            int start = Math.max(0, i - 19);
            int count = i - start + 1;
            int offset = 24;
            for (int column : ROLLING_COLUMNS) {
                double sum = 0, min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
                for (int k = start; k <= i; k++) {
                    double v = base[k][column];
                    sum += v;
                    min = Math.min(min, v);
                    max = Math.max(max, v);
                }
                double mean = sum / count, variance = 0;
                for (int k = start; k <= i; k++) {
                    double difference = base[k][column] - mean;
                    variance += difference * difference;
                }
                result[i][offset++] = (float) mean;
                result[i][offset++] = (float) Math.sqrt(variance / count);
                result[i][offset++] = (float) min;
                result[i][offset++] = (float) max;
            }
        }
        return result;
    }
}
