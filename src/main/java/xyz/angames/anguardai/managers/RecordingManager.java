package xyz.angames.anguardai.managers;

import net.kyori.adventure.bossbar.BossBar;
import net.kyori.adventure.text.Component;
import org.bukkit.Bukkit;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import xyz.angames.anguardai.SilentAC;
import xyz.angames.anguardai.data.PlayerProfile;
import xyz.angames.anguardai.utils.RecordingClock;
import xyz.angames.anguardai.utils.RecordingDuration;
import xyz.angames.anguardai.utils.Theme;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.function.LongSupplier;

/** Main-thread lifecycle for timed/untimed labeled recordings and their UI. */
public final class RecordingManager implements Listener {
    private record Active(Player target, CommandSender owner, String label, RecordingClock clock,
                          BossBar bar, Set<UUID> viewers) {}
    private final SilentAC plugin;
    private final LongSupplier clock;
    private final Map<UUID, Active> active = new HashMap<>();
    private final Map<UUID, Long> actionBars = new HashMap<>();

    public RecordingManager(SilentAC plugin) { this(plugin, System::nanoTime); }
    public RecordingManager(SilentAC plugin, LongSupplier clock) { this.plugin = plugin; this.clock = clock; }

    public void start(CommandSender owner, Player target, int label, long seconds) {
        if (label != 0 && label != 1) throw new IllegalArgumentException("Invalid recording label");
        RecordingClock timer = new RecordingClock(clock.getAsLong(), seconds);
        stop(target.getUniqueId(), false, "");
        plugin.getTrainingRecorder().start(target.getUniqueId(), label);
        resetWindow(target);
        Set<UUID> viewers = new HashSet<>();
        viewers.add(target.getUniqueId());
        if (owner instanceof Player player) viewers.add(player.getUniqueId());
        BossBar bar = timer.timed() ? BossBar.bossBar(Theme.text(Theme.raw(plugin.getConfig(), "recording_bossbar",
                "time", RecordingDuration.format(seconds), "player", target.getName())), 1f,
                BossBar.Color.WHITE, BossBar.Overlay.PROGRESS) : null;
        Active session = new Active(target, owner, label == 1 ? "cheat" : "legit", timer, bar, viewers);
        active.put(target.getUniqueId(), session);
        if (bar != null) viewers.forEach(id -> { Player p = Bukkit.getPlayer(id); if (p != null) p.showBossBar(bar); });
        Component message = Theme.message(plugin.getConfig(), "recording_started", "player", target.getName(),
                "label", session.label(), "time", timer.timed() ? RecordingDuration.format(seconds) : "без таймера");
        owner.sendMessage(message);
        if (!owner.equals(target)) target.sendMessage(message);
    }

    public boolean willNotify(UUID player, CommandSender sender) {
        Active session = active.get(player);
        return session != null && (session.owner().equals(sender) || session.target().equals(sender));
    }

    public boolean stop(UUID player, boolean notify, String reason) {
        Active session = active.remove(player);
        boolean recording = plugin.getTrainingRecorder().isRecording(player);
        TrainingRecorder.Statistics stats = plugin.getTrainingRecorder().statistics(player);
        plugin.getTrainingRecorder().stop(player);
        if (session == null) return recording;
        resetWindow(session.target());
        for (UUID viewer : session.viewers()) {
            Player p = Bukkit.getPlayer(viewer);
            if (p != null) {
                if (session.bar() != null) p.hideBossBar(session.bar());
                if (actionBars.remove(viewer) != null) p.sendActionBar(Component.empty());
            }
        }
        if (notify) {
            Component message = Theme.message(plugin.getConfig(), "recording_stopped", "player", session.target().getName(),
                    "hits", Long.toString(stats.hits()), "windows", Long.toString(stats.windows()), "reason", reason);
            session.owner().sendMessage(message);
            if (!session.owner().equals(session.target()) && session.target().isOnline()) session.target().sendMessage(message);
        }
        return true;
    }

    public boolean checkExpiry(UUID player) {
        Active session = active.get(player);
        if (session != null && session.clock().expired(clock.getAsLong())) {
            stop(player, true, "время вышло");
            return true;
        }
        return false;
    }

    public void tick() {
        long now = clock.getAsLong();
        actionBars.entrySet().removeIf(entry -> entry.getValue() <= now);
        for (Active session : new ArrayList<>(active.values())) {
            if (!session.target().isOnline()) {
                stop(session.target().getUniqueId(), false, "");
            } else if (!checkExpiry(session.target().getUniqueId()) && session.bar() != null) {
                session.bar().name(Theme.text(Theme.raw(plugin.getConfig(), "recording_bossbar",
                        "time", RecordingDuration.format(session.clock().remainingSeconds(now)),
                        "player", session.target().getName())));
                session.bar().progress(session.clock().progress(now));
            }
        }
    }

    public boolean hasRecentActionBar(UUID viewer) { return actionBars.getOrDefault(viewer, Long.MIN_VALUE) > clock.getAsLong(); }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onHit(EntityDamageByEntityEvent event) {
        if (event.isCancelled() || !(event.getDamager() instanceof Player player) || event.getEntity().equals(player)) return;
        UUID id = player.getUniqueId();
        if (checkExpiry(id)) return;
        Active session = active.get(id);
        if (session == null) return;
        plugin.getTrainingRecorder().hit(id);
        TrainingRecorder.Statistics stats = plugin.getTrainingRecorder().statistics(id);
        Component bar = Theme.text(Theme.raw(plugin.getConfig(), "recording_actionbar", "player", player.getName(),
                "hits", Long.toString(stats.hits()), "windows", Long.toString(stats.windows()),
                "samples", Long.toString(stats.windows() * 40), "label", session.label()));
        for (UUID viewer : session.viewers()) {
            Player p = Bukkit.getPlayer(viewer);
            if (p != null) {
                actionBars.put(viewer, clock.getAsLong() + 2_000_000_000L);
                p.sendActionBar(bar);
            }
        }
    }

    private void resetWindow(Player player) {
        PlayerProfile profile = plugin.getPlayerProfileManager().getProfile(player);
        if (profile == null) return;
        synchronized (profile) {
            profile.recordingGeneration.incrementAndGet();
            profile.movementHistory.clear();
            profile.ticksStep = 0;
            profile.buffer = 0;
            profile.damageMultiplier = 1.0;
        }
    }

    public void shutdown() {
        for (UUID id : new ArrayList<>(active.keySet())) stop(id, false, "");
        actionBars.clear();
    }
}
