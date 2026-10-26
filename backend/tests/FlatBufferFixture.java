import com.google.flatbuffers.FlatBufferBuilder;
import xyz.angames.anguardai.flatbuffers.TickData;
import xyz.angames.anguardai.flatbuffers.TickDataSequence;
import xyz.angames.anguardai.data.MovementFeatures;
import xyz.angames.anguardai.data.TrainingCsv;
import java.util.UUID;
import java.util.Locale;
import java.util.Arrays;
import java.nio.file.Files;
import java.nio.file.Path;

public class FlatBufferFixture {
    public static void main(String[] args) throws Exception {
        FlatBufferBuilder builder = new FlatBufferBuilder(4096);
        int[] ticks = new int[40];
        double[][] base = new double[40][8];
        for (int i = 0; i < 40; i++) {
            base[i] = new double[] {i / 10f, -i / 10f, .2f, -.3f, .4f, -.5f, 0f, .001f};
            ticks[i] = TickData.createTickData(builder, i / 10f, -i / 10f,
                    .2f, -.3f, .4f, -.5f, 0f, .001f);
        }
        int vector = TickDataSequence.createTicksVector(builder, ticks);
        builder.finish(TickDataSequence.createTickDataSequence(builder, vector));
        Files.write(Path.of(args[0]), builder.sizedByteArray());
        for (int i = 0; i < 40; i++) {
            base[i] = new double[] {i % 3 == 0 ? 0f : (float) (Math.sin(i * 1.7) * 179),
                    (float) (Math.cos(i * .7) * 85), (float) Math.sin(i),
                    (float) Math.cos(i), (float) Math.sin(i * .3), (float) Math.cos(i * .3),
                    i % 2 == 0 ? 0f : .001f, .002f};
        }
        float[][] expanded = MovementFeatures.expand(base);
        Files.writeString(Path.of(args[0] + ".features.json"), Arrays.deepToString(expanded));
        builder.clear();
        for (int i = 0; i < 40; i++) {
            int derived = TickData.createDerivedVector(builder, Arrays.copyOfRange(expanded[i], 8, 48));
            TickData.startTickData(builder);
            TickData.addDerived(builder, derived);
            TickData.addDeltaYaw(builder, expanded[i][0]);
            TickData.addDeltaPitch(builder, expanded[i][1]);
            TickData.addAccelYaw(builder, expanded[i][2]);
            TickData.addAccelPitch(builder, expanded[i][3]);
            TickData.addJerkPitch(builder, expanded[i][4]);
            TickData.addJerkYaw(builder, expanded[i][5]);
            TickData.addGcdErrorYaw(builder, expanded[i][6]);
            TickData.addGcdErrorPitch(builder, expanded[i][7]);
            ticks[i] = TickData.endTickData(builder);
        }
        vector = TickDataSequence.createTicksVector(builder, ticks);
        TickDataSequence.startTickDataSequence(builder);
        TickDataSequence.addTicks(builder, vector);
        TickDataSequence.addSchemaVersion(builder, MovementFeatures.SCHEMA);
        builder.finish(TickDataSequence.endTickDataSequence(builder));
        Files.write(Path.of(args[0] + ".v2"), builder.sizedByteArray());
        Locale.setDefault(Locale.GERMANY);
        StringBuilder csv = new StringBuilder(TrainingCsv.header()).append('\n');
        UUID player = UUID.fromString("00000000-0000-0000-0000-000000000001");
        UUID session = UUID.fromString("00000000-0000-0000-0000-000000000002");
        UUID window = UUID.fromString("00000000-0000-0000-0000-000000000003");
        for (int i = 0; i < 40; i++) {
            csv.append(TrainingCsv.row(1, expanded[i], player, session, window, i, 1000)).append('\n');
        }
        Files.writeString(Path.of(args[0] + ".csv"), csv);
    }
}
