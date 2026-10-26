package xyz.angames.anguardai.checks.impl;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import org.bukkit.Bukkit;
import org.bukkit.entity.Player;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.player.PlayerMoveEvent;
import xyz.angames.anguardai.SilentAC;
import xyz.angames.anguardai.checks.Check;
import xyz.angames.anguardai.data.PlayerProfile;
import xyz.angames.anguardai.data.TickDataPOJO;
import xyz.angames.anguardai.data.MovementFeatures;
import xyz.angames.anguardai.flatbuffers.TickData;
import xyz.angames.anguardai.flatbuffers.TickDataSequence;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class AimCheck extends Check {

    private final int sequence;
    private final int step;
    private static final ThreadLocal<FlatBufferBuilder> BUILDER = ThreadLocal.withInitial(() -> new FlatBufferBuilder(4096));
    private static final Gson GSON = new Gson();

    public AimCheck(Player player, PlayerProfile profile, SilentAC anguardai) {
        super(player, profile, anguardai);
        this.sequence = PlayerProfile.HISTORY_SIZE;
        this.step = Math.max(1, plugin.getConfig().getInt("ai.step", 10));
        if (plugin.getConfig().getInt("ai.sequence", 40) != sequence) {
            plugin.getLogger().warning("Training schema v2 requires ai.sequence=40; using 40.");
        }
    }

    @Override
    public void onDamage(EntityDamageByEntityEvent event) {
        if (event.getDamager().equals(this.player)) {
            if (!plugin.getConfig().getBoolean("ai.observe-only", true)
                    && !plugin.getTrainingRecorder().isRecording(player.getUniqueId())
                    && plugin.getConfig().getBoolean("ai.damage-reduction.enabled", false)
                    && System.currentTimeMillis() - profile.lastPredictionTime < 5000
                    && this.profile.damageMultiplier < 1.0) {
                event.setDamage(event.getDamage() * this.profile.damageMultiplier);
            }
            this.profile.isAttacking = true;
            this.profile.ticksSinceAttack = 0;
            this.profile.ticksSinceLastCombatAction = 0;
        }
        if (event.getEntity().equals(this.player)) {
            this.profile.ticksSinceLastCombatAction = 0;
        }
    }

    @Override
    public void onMove(PlayerMoveEvent event) {}

    public void handlePacketMovement(float toYaw, float toPitch, boolean onGround) {
        if (!Float.isFinite(toYaw) || !Float.isFinite(toPitch)) return;
        if (System.currentTimeMillis() - this.profile.lastTeleportTime < 2000) {
            synchronized (this.profile) {
                this.profile.movementHistory.clear();
                this.profile.ticksStep = 0;
                this.profile.aimProcessor.reset();
                resetProfile(toYaw, toPitch);
            }
            return;
        }
        if (this.profile.lastYaw == toYaw && this.profile.lastPitch == toPitch) return;


        synchronized (this.profile) {
            float deltaYaw = (toYaw - this.profile.lastYaw + 540.0F) % 360.0F - 180.0F;
            float deltaPitch = toPitch - this.profile.lastPitch;
            this.profile.aimProcessor.process(deltaYaw, deltaPitch);
            TickDataPOJO tick = new TickDataPOJO(
                    deltaYaw, deltaPitch,
                    this.profile.aimProcessor.accelYaw,
                    this.profile.aimProcessor.accelPitch,
                    this.profile.aimProcessor.jerkYaw,
                    this.profile.aimProcessor.jerkPitch,
                    this.profile.aimProcessor.getGcdErrorYaw(deltaYaw),
                    this.profile.aimProcessor.getGcdErrorPitch(deltaPitch)
            );


            this.profile.addMovementFrame(tick);
            this.profile.ticksStep++;
            while (this.profile.movementHistory.size() > sequence) {
                this.profile.movementHistory.removeFirst();
            }

            this.profile.lastYaw = toYaw;
            this.profile.lastPitch = toPitch;
            this.updateProfile(toYaw, toPitch, deltaYaw, deltaPitch);
            if (this.profile.movementHistory.size() == sequence && this.profile.ticksStep >= step) {
                sendData();
                this.profile.ticksStep = 0;
            }
        }
    }

    private void sendData() {
        List<TickDataPOJO> history = new ArrayList<>(this.profile.movementHistory);
        String recordingSession = plugin.getTrainingRecorder().sessionId(player.getUniqueId());
        long generation = profile.recordingGeneration.get();
        // Bukkit APIs and labels are read on the server thread. Recording is independent of inference.
        Bukkit.getScheduler().runTask(plugin, () -> {
            if (plugin.getRecordingManager().checkExpiry(player.getUniqueId())) return;
            if (!player.isOnline() || plugin.getPlayerProfileManager().getProfile(player) != profile
                    || profile.recordingGeneration.get() != generation) return;
            if (player.getPing() > 200 || Bukkit.getTPS()[0] < 18.0
                    || System.currentTimeMillis() - profile.lastTeleportTime < 2000) {
                profile.damageMultiplier = 1.0;
                profile.buffer = 0;
                return;
            }
            plugin.getTrainingRecorder().record(player.getUniqueId(), recordingSession, history);
            if ((player.hasPermission("silentac.bypass") || player.hasPermission("anguard.bypass")) || plugin.getTrainingRecorder().isRecording(player.getUniqueId())) return;
            if (!profile.predictionPending.compareAndSet(false, true)) return;
            long requestedAt = System.currentTimeMillis();
            try {
                byte[] binaryData = serialize(history);
                plugin.getAiServer().sendRequest(binaryData)
                        .thenAccept(response -> onResponse(response, generation, requestedAt))
                        .whenComplete((ignored, error) -> profile.predictionPending.set(false));
            } catch (Exception e) {
                profile.predictionPending.set(false);
                plugin.getLogger().warning("Could not submit AI window: " + e.getMessage());
            }
        });
    }

    private void onResponse(String response, long generation, long requestedAt) {
        if (response == null) {
            clearInference(generation);
            return;
        }
        try {
            JsonObject json = GSON.fromJson(response, JsonObject.class);
            if (!json.has("confidence") || !json.has("ready") || !json.get("ready").getAsBoolean()
                    || !json.has("threshold")) {
                clearInference(generation);
                return;
            }

            double probability = json.get("confidence").getAsDouble() / 100.0;
            double threshold = json.get("threshold").getAsDouble() / 100.0;
            if (!Double.isFinite(probability) || probability < 0 || probability > 1
                    || !Double.isFinite(threshold) || threshold < 0 || threshold > 1) {
                clearInference(generation);
                return;
            }

            Bukkit.getScheduler().runTask(plugin, () -> {
                if (!player.isOnline() || plugin.getPlayerProfileManager().getProfile(player) != profile
                        || profile.recordingGeneration.get() != generation) return;
                this.profile.suspicion = probability * 100.0;
                if (plugin.getConfig().getBoolean("ai.observe-only", true)
                        || plugin.getTrainingRecorder().isRecording(player.getUniqueId())
                        || profile.lastTeleportTime > requestedAt
                        || System.currentTimeMillis() - requestedAt > 5000
                        || player.getPing() > 200 || Bukkit.getTPS()[0] < 18.0
                        || System.currentTimeMillis() - profile.lastTeleportTime < 2000) {
                    profile.damageMultiplier = 1.0;
                    profile.buffer = 0;
                    return;
                }
                profile.lastPredictionTime = System.currentTimeMillis();

                double cheatProb = threshold;

                if (probability > cheatProb) {
                    this.profile.buffer += (probability - cheatProb) * plugin.getConfig().getDouble("ai.buffer.multiplier", 100.0);
                } else {
                    this.profile.buffer = Math.max(0, this.profile.buffer - plugin.getConfig().getDouble("ai.buffer.decrease", 0.25));
                }

                double drProb = plugin.getConfig().getDouble("ai.damage-reduction.suspicion", 0.9);
                if (plugin.getConfig().getBoolean("ai.damage-reduction.enabled", false)
                        && drProb >= 0 && drProb < 1 && probability > threshold && probability >= drProb) {
                    double ratio = (probability - drProb) / (1.0 - drProb);
                    this.profile.damageMultiplier = 1.0 - Math.min(1.0, ratio);
                } else {
                    this.profile.damageMultiplier = 1.0;
                }

                if (this.profile.buffer > plugin.getConfig().getDouble("ai.buffer.flag", 50.0)) {
                    this.profile.violationLevel++;
                    this.profile.buffer = plugin.getConfig().getDouble("ai.buffer.reset-on-flag", 25.0);
                    if (plugin.getConfig().getBoolean("autoban.enable")) {
                        String cmd = plugin.getConfig().getString("autoban.command").replace("{player}", player.getName());
                        Bukkit.dispatchCommand(Bukkit.getConsoleSender(), cmd);
                    }
                }
            });
        } catch (Exception ignored) {
            clearInference(generation);
        }
    }

    private void clearInference(long generation) {
        Bukkit.getScheduler().runTask(plugin, () -> {
            if (plugin.getPlayerProfileManager().getProfile(player) == profile
                    && profile.recordingGeneration.get() == generation) {
                profile.buffer = 0;
                profile.damageMultiplier = 1.0;
            }
        });
    }


    private void resetProfile(float yaw, float pitch) {
        this.profile.lastYaw = yaw;
        this.profile.lastPitch = pitch;
        this.profile.lastDeltaYaw = 0;
        this.profile.lastDeltaPitch = 0;
        this.profile.lastAccelYaw = 0;
        this.profile.lastAccelPitch = 0;
        this.profile.aimProcessor.currentYawAccel = 0;
        this.profile.aimProcessor.currentPitchAccel = 0;
        this.profile.aimProcessor.jerkYaw = 0;
        this.profile.aimProcessor.jerkPitch = 0;
    }

    private void updateProfile(float currentYaw, float currentPitch, float deltaYaw, float deltaPitch) {
        float accelYaw = deltaYaw - this.profile.lastDeltaYaw;
        float accelPitch = deltaPitch - this.profile.lastDeltaPitch;

        this.profile.lastYaw = currentYaw;
        this.profile.lastPitch = currentPitch;
        this.profile.lastDeltaYaw = deltaYaw;
        this.profile.lastDeltaPitch = deltaPitch;
        this.profile.lastAccelYaw = accelYaw;
        this.profile.lastAccelPitch = accelPitch;
    }

    private byte[] serialize(List<TickDataPOJO> ticks) {
        FlatBufferBuilder builder = BUILDER.get();
        builder.clear();
        int[] tickOffsets = new int[ticks.size()];
        float[][] features = MovementFeatures.expand(ticks);

        for (int i = 0; i < ticks.size(); i++) {
            TickDataPOJO t = ticks.get(i);

            int derived = TickData.createDerivedVector(builder, java.util.Arrays.copyOfRange(features[i], 8, MovementFeatures.COUNT));
            TickData.startTickData(builder);
            TickData.addDerived(builder, derived);
            TickData.addDeltaYaw(builder, t.deltaYaw);
            TickData.addDeltaPitch(builder, t.deltaPitch);
            TickData.addAccelYaw(builder, t.accelYaw);
            TickData.addAccelPitch(builder, t.accelPitch);
            TickData.addJerkYaw(builder, t.jerkYaw);
            TickData.addJerkPitch(builder, t.jerkPitch);
            TickData.addGcdErrorYaw(builder, t.gcdErrorYaw);
            TickData.addGcdErrorPitch(builder, t.gcdErrorPitch);
            tickOffsets[i] = TickData.endTickData(builder);
        }

        int ticksVector = TickDataSequence.createTicksVector(builder, tickOffsets);
        TickDataSequence.startTickDataSequence(builder);
        TickDataSequence.addTicks(builder, ticksVector);
        TickDataSequence.addSchemaVersion(builder, MovementFeatures.SCHEMA);
        int endOffset = TickDataSequence.endTickDataSequence(builder);
        builder.finish(endOffset);
        ByteBuffer buf = builder.dataBuffer();
        byte[] arr = new byte[buf.remaining()];
        buf.get(arr);
        return arr;
    }
}
