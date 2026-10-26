import com.google.flatbuffers.FlatBufferBuilder;
import xyz.angames.anguardai.flatbuffers.TickData;
import xyz.angames.anguardai.flatbuffers.TickDataSequence;
import java.nio.file.Files;
import java.nio.file.Path;

public class FlatBufferFixture {
    public static void main(String[] args) throws Exception {
        FlatBufferBuilder builder = new FlatBufferBuilder(4096);
        int[] ticks = new int[40];
        for (int i = 0; i < 40; i++) {
            ticks[i] = TickData.createTickData(builder, i / 10f, -i / 10f,
                    .2f, -.3f, .4f, -.5f, 0f, .001f);
        }
        int vector = TickDataSequence.createTicksVector(builder, ticks);
        builder.finish(TickDataSequence.createTickDataSequence(builder, vector));
        Files.write(Path.of(args[0]), builder.sizedByteArray());
    }
}
