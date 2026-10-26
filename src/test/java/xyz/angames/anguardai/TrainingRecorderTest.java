package xyz.angames.anguardai;

import com.google.gson.JsonParser;
import org.bukkit.configuration.file.YamlConfiguration;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import xyz.angames.anguardai.data.TickDataPOJO;
import xyz.angames.anguardai.managers.TrainingRecorder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.UUID;
import java.util.logging.Logger;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class TrainingRecorderTest {
    @TempDir Path directory;
    @Test void hitCountsAndWrittenWindowsMatchPersistedJsonAndCsv() throws Exception {
        SilentAC plugin = mock(SilentAC.class);
        when(plugin.getDataFolder()).thenReturn(directory.toFile());
        when(plugin.getConfig()).thenReturn(new YamlConfiguration());
        when(plugin.getLogger()).thenReturn(Logger.getAnonymousLogger());
        UUID player = UUID.randomUUID();
        try (TrainingRecorder recorder = new TrainingRecorder(plugin)) {
            String session = recorder.start(player, 0);
            recorder.hit(player); recorder.hit(player);
            var frames = new ArrayList<TickDataPOJO>();
            for (int i = 0; i < 40; i++) frames.add(new TickDataPOJO(i / 10f, -i / 10f, .1f, .2f, .3f, .4f, 0, .001f));
            recorder.record(player, session, frames);
            long deadline = System.nanoTime() + 5_000_000_000L;
            while (recorder.statistics(player).windows() == 0 && System.nanoTime() < deadline) Thread.sleep(10);
            assertEquals(2, recorder.statistics(player).hits());
            assertEquals(1, recorder.statistics(player).windows());
        }
        try (var files = Files.list(directory.resolve("training"))) {
            for (Path file : files.toList()) {
                if (file.toString().endsWith(".jsonl")) {
                    var record = JsonParser.parseString(Files.readString(file)).getAsJsonObject();
                    assertEquals(2, record.get("hits").getAsInt());
                    assertEquals(40, record.getAsJsonArray("frames").size());
                    assertEquals(48, record.getAsJsonArray("frames").get(0).getAsJsonArray().size());
                } else if (file.toString().endsWith(".csv")) {
                    var lines = Files.readAllLines(file);
                    assertEquals(41, lines.size());
                    assertEquals(55, lines.get(1).split(",").length);
                }
            }
        }
    }
}
