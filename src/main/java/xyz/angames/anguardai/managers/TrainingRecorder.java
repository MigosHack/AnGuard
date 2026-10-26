package xyz.angames.anguardai.managers;

import com.google.gson.Gson;
import xyz.angames.anguardai.Anguardai;
import xyz.angames.anguardai.data.TickDataPOJO;
import xyz.angames.anguardai.data.MovementFeatures;
import xyz.angames.anguardai.data.TrainingCsv;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/** Explicit administrator labels only; never label using the model's predictions. */
public final class TrainingRecorder implements AutoCloseable {
    private record Session(String id, int label) {}
    private final Map<UUID, Session> sessions = new ConcurrentHashMap<>();
    private record RecordedWindow(String json, float[][] values, int label, UUID player,
                                  UUID session, UUID id, long timestamp) {}
    private final ArrayBlockingQueue<RecordedWindow> queue = new ArrayBlockingQueue<>(1024);
    private final AtomicLong dropped = new AtomicLong();
    private final Thread writer;
    private final Anguardai plugin;
    private final Gson gson = new Gson();
    private volatile boolean running = true;
    private volatile boolean healthy = true;

    public TrainingRecorder(Anguardai plugin) throws IOException {
        this.plugin = plugin;
        Path directory = plugin.getDataFolder().toPath().resolve("training");
        Files.createDirectories(directory);
        String basename = "windows-" + UUID.randomUUID();
        Path file = directory.resolve(basename + ".jsonl");
        BufferedWriter output = Files.newBufferedWriter(file, StandardOpenOption.CREATE_NEW);
        final BufferedWriter csv;
        try {
            csv = plugin.getConfig().getBoolean("training.csv-enabled", true)
                    ? Files.newBufferedWriter(directory.resolve(basename + ".csv"), StandardOpenOption.CREATE_NEW) : null;
            if (csv != null) {
                csv.write(TrainingCsv.header());
                csv.newLine();
                csv.flush();
            }
        } catch (IOException error) {
            output.close();
            throw error;
        }
        writer = new Thread(() -> {
            try (output; csv) {
                while (running || !queue.isEmpty()) {
                    RecordedWindow window = queue.poll(250, java.util.concurrent.TimeUnit.MILLISECONDS);
                    if (window != null) {
                        output.write(window.json());
                        output.newLine();
                        output.flush();
                        if (csv != null) {
                            for (int i = 0; i < window.values().length; i++) {
                                csv.write(TrainingCsv.row(window.label(), window.values()[i], window.player(),
                                        window.session(), window.id(), i, window.timestamp()));
                                csv.newLine();
                            }
                            csv.flush();
                        }
                    }
                }
            } catch (IOException | InterruptedException error) {
                healthy = false;
                plugin.getLogger().severe("Training recorder stopped: " + error.getMessage());
            }
        }, "anguard-training-writer");
        writer.setDaemon(true);
        writer.start();
    }

    public String start(UUID player, int label) {
        if (!healthy || !running) throw new IllegalStateException("Training recorder unavailable");
        Session session = new Session(UUID.randomUUID().toString(), label);
        sessions.put(player, session);
        return session.id();
    }

    public void stop(UUID player) { sessions.remove(player); }

    public boolean isRecording(UUID player) { return sessions.containsKey(player); }
    public String sessionId(UUID player) {
        Session session = sessions.get(player);
        return session == null ? null : session.id();
    }

    public void record(UUID player, String expectedSession, List<TickDataPOJO> frames) {
        Session session = sessions.get(player);
        if (session == null || !session.id().equals(expectedSession) || !healthy || !running) return;
        float[][] values = MovementFeatures.expand(frames);
        UUID windowId = UUID.randomUUID();
        long timestamp = System.currentTimeMillis();
        String line = gson.toJson(Map.of("schema", MovementFeatures.SCHEMA, "player_id", player.toString(),
                "session_id", session.id(), "label", session.label(),
                "timestamp_ms", timestamp, "window_id", windowId.toString(), "frames", values));
        if (!queue.offer(new RecordedWindow(line, values, session.label(), player,
                UUID.fromString(session.id()), windowId, timestamp)) && dropped.incrementAndGet() % 100 == 1) {
            plugin.getLogger().warning("Training queue full; dropped windows: " + dropped.get());
        }
    }

    @Override
    public void close() {
        sessions.clear();
        running = false;
        try {
            writer.join(5000);
            if (writer.isAlive()) plugin.getLogger().warning("Training queue did not drain before shutdown.");
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
        }
    }
}
