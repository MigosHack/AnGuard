package xyz.angames.anguardai;

import net.kyori.adventure.bossbar.BossBar;
import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.serializer.plain.PlainTextComponentSerializer;
import org.bukkit.Bukkit;
import org.bukkit.configuration.file.YamlConfiguration;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.junit.jupiter.api.*;
import org.mockito.ArgumentCaptor;
import org.mockito.MockedStatic;
import xyz.angames.anguardai.data.PlayerProfile;
import xyz.angames.anguardai.managers.PlayerProfileManager;
import xyz.angames.anguardai.managers.RecordingManager;
import xyz.angames.anguardai.managers.TrainingRecorder;

import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class RecordingManagerTest {
    SilentAC plugin;
    TrainingRecorder recorder;
    Player target, owner;
    RecordingManager manager;
    MockedStatic<Bukkit> bukkit;
    AtomicLong time;
    PlayerProfile profile;

    @BeforeEach void setup() {
        plugin = mock(SilentAC.class);
        recorder = mock(TrainingRecorder.class);
        when(plugin.getTrainingRecorder()).thenReturn(recorder);
        when(plugin.getConfig()).thenReturn(YamlConfiguration.loadConfiguration(new InputStreamReader(
                getClass().getResourceAsStream("/config.yml"), StandardCharsets.UTF_8)));
        target = player("Steve"); owner = player("Admin");
        bukkit = mockStatic(Bukkit.class);
        bukkit.when(() -> Bukkit.getPlayer(target.getUniqueId())).thenReturn(target);
        bukkit.when(() -> Bukkit.getPlayer(owner.getUniqueId())).thenReturn(owner);
        PlayerProfileManager profiles = mock(PlayerProfileManager.class);
        profile = new PlayerProfile(0, 0);
        when(plugin.getPlayerProfileManager()).thenReturn(profiles);
        when(profiles.getProfile(target)).thenReturn(profile);
        when(recorder.statistics(any())).thenReturn(new TrainingRecorder.Statistics(12, 5));
        time = new AtomicLong(0);
        manager = new RecordingManager(plugin, time::get);
    }
    private Player player(String name) {
        Player p = mock(Player.class);
        when(p.getName()).thenReturn(name);
        when(p.getUniqueId()).thenReturn(UUID.randomUUID());
        when(p.isOnline()).thenReturn(true);
        return p;
    }
    @AfterEach void cleanup() { bukkit.close(); }

    @Test void whiteProgressBossbarCountsDownAndExpiresOnce() {
        manager.start(owner, target, 0, 900);
        ArgumentCaptor<BossBar> capture = ArgumentCaptor.forClass(BossBar.class);
        verify(target).showBossBar(capture.capture());
        BossBar bar = capture.getValue();
        assertEquals(BossBar.Color.WHITE, bar.color());
        assertEquals(BossBar.Overlay.PROGRESS, bar.overlay());
        assertEquals("Осталось: 15:00", PlainTextComponentSerializer.plainText().serialize(bar.name()));
        verify(owner).showBossBar(bar);
        time.set(450_000_000_000L); manager.tick();
        assertEquals(.5f, bar.progress());
        assertEquals("Осталось: 07:30", PlainTextComponentSerializer.plainText().serialize(bar.name()));
        clearInvocations(recorder);
        time.set(900_000_000_000L); manager.tick(); manager.tick();
        verify(recorder, times(1)).stop(target.getUniqueId());
        verify(target).hideBossBar(bar);
        verify(owner).hideBossBar(bar);
        assertEquals(2, profile.recordingGeneration.get());
    }
    @Test void untimedRecordingHasNoBossbarAndStillCountsHits() {
        manager.start(owner, target, 1, 0);
        verify(target, never()).showBossBar(any());
        EntityDamageByEntityEvent event = mock(EntityDamageByEntityEvent.class);
        when(event.getDamager()).thenReturn(target);
        when(event.getEntity()).thenReturn(mock(Entity.class));
        manager.onHit(event);
        verify(recorder).hit(target.getUniqueId());
        ArgumentCaptor<Component> capture = ArgumentCaptor.forClass(Component.class);
        verify(target).sendActionBar(capture.capture());
        String text = PlainTextComponentSerializer.plainText().serialize(capture.getValue());
        assertTrue(text.contains("Удары: 12"));
        assertTrue(text.contains("Окна: 5"));
        assertTrue(text.contains("Сэмплы: 200"));
        assertTrue(manager.hasRecentActionBar(owner.getUniqueId()));
        when(event.isCancelled()).thenReturn(true); manager.onHit(event);
        verify(recorder, times(1)).hit(target.getUniqueId());
    }
    @Test void replacingStoppingDisconnectingAndShutdownRemoveBars() {
        manager.start(owner, target, 0, 900);
        ArgumentCaptor<BossBar> first = ArgumentCaptor.forClass(BossBar.class);
        verify(target).showBossBar(first.capture());
        manager.start(owner, target, 1, 60);
        verify(target).hideBossBar(first.getValue());
        when(target.isOnline()).thenReturn(false); manager.tick();
        verify(target, times(2)).hideBossBar(any());
        when(target.isOnline()).thenReturn(true);
        manager.start(owner, target, 0, 60); manager.shutdown();
        verify(target, times(3)).hideBossBar(any());
        assertFalse(manager.stop(target.getUniqueId(), false, ""));
    }
    @Test void lateHitCannotExtendExpiredRecording() {
        manager.start(owner, target, 0, 1);
        EntityDamageByEntityEvent event = mock(EntityDamageByEntityEvent.class);
        when(event.getDamager()).thenReturn(target);
        when(event.getEntity()).thenReturn(mock(Entity.class));
        time.set(1_000_000_000L); manager.onHit(event);
        verify(recorder, never()).hit(any());
    }
}
