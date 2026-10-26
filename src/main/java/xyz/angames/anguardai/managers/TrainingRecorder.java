package xyz.angames.anguardai.managers;

import com.google.gson.Gson;
import xyz.angames.anguardai.Anguardai;
import xyz.angames.anguardai.data.TickDataPOJO;
import xyz.angames.anguardai.data.MovementFeatures;

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
    private final ArrayBlockingQueue<String> queue = new ArrayBlockingQueue<>(1024);
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
        Path file = directory.resolve("windows-" + UUID.randomUUID() + ".jsonl");
        BufferedWriter output = Files.newBufferedWriter(file, StandardOpenOption.CREATE_NEW);
        writer = new Thread(() -> {
            try (output) {
                while (running || !queue.isEmpty()) {
                    String line = queue.poll(250, java.util.concurrent.TimeUnit.MILLISECONDS);
                    if (line != null) {
                        output.write(line);
                        output.newLine();
                        output.flush();
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
        String line = gson.toJson(Map.of("schema", MovementFeatures.SCHEMA, "player_id", player.toString(),
                "session_id", session.id(), "label", session.label(),
                "timestamp_ms", System.currentTimeMillis(), "frames", values));
        if (!queue.offer(line) && dropped.incrementAndGet() % 100 == 1) {
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
