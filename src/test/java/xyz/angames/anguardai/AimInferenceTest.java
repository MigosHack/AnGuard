package xyz.angames.anguardai;

import org.bukkit.Bukkit;
import org.bukkit.configuration.file.YamlConfiguration;
import org.bukkit.entity.Player;
import org.bukkit.scheduler.BukkitScheduler;
import org.junit.jupiter.api.Test;
import xyz.angames.anguardai.checks.impl.AimCheck;
import xyz.angames.anguardai.client.AIServer;
import xyz.angames.anguardai.data.PlayerProfile;
import xyz.angames.anguardai.managers.*;
import xyz.angames.anguardai.utils.InferenceDiagnostics;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.function.IntConsumer;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class AimInferenceTest {
    @Test void observeModeStillReceivesScoreAndHttpFailuresStayDistinctFromZero() {
        for (int code : new int[]{200, 401}) {
            SilentAC plugin = mock(SilentAC.class);
            YamlConfiguration config = new YamlConfiguration();
            config.set("ai.observe-only", true);
            when(plugin.getConfig()).thenReturn(config);
            Player player = mock(Player.class);
            when(player.getUniqueId()).thenReturn(UUID.randomUUID());
            when(player.isOnline()).thenReturn(true);
            PlayerProfile profile = new PlayerProfile(0, 0);
            PlayerProfileManager profiles = mock(PlayerProfileManager.class);
            when(plugin.getPlayerProfileManager()).thenReturn(profiles);
            when(profiles.getProfile(player)).thenReturn(profile);
            when(plugin.getRecordingManager()).thenReturn(mock(RecordingManager.class));
            when(plugin.getTrainingRecorder()).thenReturn(mock(TrainingRecorder.class));
            AIServer api = mock(AIServer.class);
            when(plugin.getAiServer()).thenReturn(api);
            when(api.sendRequest(any(byte[].class), any(IntConsumer.class))).thenAnswer(call -> {
                call.getArgument(1, IntConsumer.class).accept(code);
                return CompletableFuture.completedFuture(code == 200 ? "{\"ready\":true,\"confidence\":97.5,\"threshold\":91}" : null);
            });
            BukkitScheduler scheduler = mock(BukkitScheduler.class);
            when(scheduler.runTask(eq(plugin), any(Runnable.class))).thenAnswer(call -> { call.getArgument(1, Runnable.class).run(); return null; });
            try (var bukkit = mockStatic(Bukkit.class)) {
                bukkit.when(Bukkit::getScheduler).thenReturn(scheduler);
                bukkit.when(Bukkit::getTPS).thenReturn(new double[]{20,20,20});
                AimCheck check = new AimCheck(player, profile, plugin);
                for (int i = 1; i < 40; i++) check.handlePacketMovement(i, 0, true);
                assertEquals(0, profile.predictionRequests);
                assertEquals("—", InferenceDiagnostics.score(profile));
                check.handlePacketMovement(40, 0, true);
                assertEquals(40, profile.windowSamples);
                assertEquals(1, profile.predictionRequests);
                assertEquals(code, profile.lastApiStatus);
                assertEquals(0, profile.buffer);
                assertEquals(0, profile.violationLevel);
                if (code == 200) {
                    assertEquals(1, profile.predictionResponses);
                    assertEquals(97.5, profile.suspicion);
                    assertTrue(profile.lastPredictionTime > 0);
                    assertFalse(profile.inferenceError);
                } else {
                    assertEquals(0, profile.predictionResponses);
                    assertTrue(profile.inferenceError);
                    assertEquals("—", InferenceDiagnostics.score(profile));
                }
            }
        }
    }
}
