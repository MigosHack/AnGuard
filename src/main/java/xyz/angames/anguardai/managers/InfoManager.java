package xyz.angames.anguardai.managers;

import net.kyori.adventure.text.Component;
import org.bukkit.entity.Player;
import org.bukkit.Bukkit;
import org.bukkit.scheduler.BukkitRunnable;
import org.bukkit.scheduler.BukkitTask;
import xyz.angames.anguardai.SilentAC;
import xyz.angames.anguardai.data.PlayerProfile;
import xyz.angames.anguardai.utils.Theme;
import xyz.angames.anguardai.utils.InferenceDiagnostics;

import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

public class InfoManager {
    private final SilentAC plugin;
    private final PlayerProfileManager profiles;
    private final Map<UUID, BukkitTask> tasks = new ConcurrentHashMap<>();

    public InfoManager(SilentAC plugin, PlayerProfileManager profiles) { this.plugin = plugin; this.profiles = profiles; }

    public void toggleInfoDisplay(Player sender, Player target) {
        UUID viewer = sender.getUniqueId();
        BukkitTask previous = tasks.remove(viewer);
        if (previous != null) {
            previous.cancel();
            if (!plugin.getRecordingManager().hasRecentActionBar(viewer)) sender.sendActionBar(Component.empty());
            sender.sendMessage(Theme.message(plugin.getConfig(), "tracker_disabled", "player", target.getName()));
            return;
        }
        BukkitTask task = new BukkitRunnable() {
            @Override
            public void run() {
                if (!sender.isOnline() || !target.isOnline()) {
                    cancel();
                    tasks.remove(viewer);
                    return;
                }
                if (plugin.getRecordingManager().hasRecentActionBar(viewer)) return;
                PlayerProfile profile = profiles.getProfile(target);
                if (profile == null) return;
                String sensitivity = profile.aimProcessor.sensitivityX > 0
                        ? String.format(Locale.ROOT, "%.0f%%", profile.aimProcessor.sensitivityX * 100) : "…";
                sender.sendActionBar(Theme.text(String.format(Locale.ROOT,
                        "<gradient:#FFFFFF:#AAAAAA>%s</gradient> <dark_gray>│</dark_gray> "
                        + "<gray>Пинг: <white>%d</white> <dark_gray>│</dark_gray> "
                        + "<gray>VL: <white>%d</white> <dark_gray>│</dark_gray> "
                        + "<gray>Буфер: <white>%.1f</white> <dark_gray>│</dark_gray> "
                        + "<gray>Оценка: <gradient:#FFFFFF:#B0B0B0>%s</gradient> <dark_gray>│</dark_gray> "
                        + "<gray>Sens: <white>%s</white> <dark_gray>│</dark_gray> <gray>Урон: <white>%.2fx</white> <dark_gray>│</dark_gray> <gray>%s",
                        Theme.escape(target.getName()), target.getPing(), profile.violationLevel, profile.buffer,
                        InferenceDiagnostics.score(profile), sensitivity, profile.damageMultiplier,
                        Theme.escape(InferenceDiagnostics.status(profile, plugin.getConfig().getBoolean("checks.aim.enabled", true),
                                plugin.getTrainingRecorder().isRecording(target.getUniqueId()), InferenceDiagnostics.bypassed(target),
                                target.getPing(), Bukkit.getTPS()[0], System.currentTimeMillis())))));
            }
        }.runTaskTimer(plugin, 0L, 2L);
        tasks.put(viewer, task);
        sender.sendMessage(Theme.message(plugin.getConfig(), "tracker_enabled", "player", target.getName()));
    }

    public void shutdown() {
        tasks.values().forEach(BukkitTask::cancel);
        tasks.clear();
    }
}
