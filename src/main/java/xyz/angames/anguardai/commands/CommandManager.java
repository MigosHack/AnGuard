package xyz.angames.anguardai.commands;

import org.bukkit.Bukkit;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.command.TabExecutor;
import org.bukkit.entity.Player;
import xyz.angames.anguardai.SilentAC;
import xyz.angames.anguardai.data.PlayerProfile;
import xyz.angames.anguardai.managers.InfoManager;
import xyz.angames.anguardai.managers.PlayerProfileManager;
import xyz.angames.anguardai.utils.RecordingDuration;
import xyz.angames.anguardai.utils.Theme;

import java.util.List;
import java.util.Locale;

public class CommandManager implements TabExecutor {
    private final SilentAC plugin;
    private final InfoManager infoManager;
    private final PlayerProfileManager profileManager;

    public CommandManager(SilentAC plugin, InfoManager infoManager, PlayerProfileManager profileManager) {
        this.plugin = plugin;
        this.infoManager = infoManager;
        this.profileManager = profileManager;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String alias, String[] args) {
        if (!allowed(sender)) return message(sender, "no_permission");
        if (args.length == 0 || args[0].equalsIgnoreCase("help")) return help(sender);
        if (args[0].equalsIgnoreCase("recording")) return recording(sender, args);
        if (args.length < 2) return message(sender, "invalid_usage");
        Player target = Bukkit.getPlayerExact(args[1]);
        if (target == null) return message(sender, "player_not_found", "player", args[1]);
        switch (args[0].toLowerCase(Locale.ROOT)) {
            case "record":
                if (args.length != 3 || !isLabel(args[2])) return message(sender, "recording_usage");
                plugin.getRecordingManager().start(sender, target, args[2].equalsIgnoreCase("cheat") ? 1 : 0, 0);
                return true;
            case "stop":
                return stop(sender, target);
            case "info":
                if (!(sender instanceof Player player)) return message(sender, "player_only");
                infoManager.toggleInfoDisplay(player, target);
                return true;
            case "profile":
                displayProfile(sender, target);
                return true;
            default:
                return help(sender);
        }
    }

    private boolean recording(CommandSender sender, String[] args) {
        if (args.length < 3 || args.length > 5) return message(sender, "recording_usage");
        String action = args[1].toLowerCase(Locale.ROOT);
        if (!action.equals("start") && !action.equals("stop")) return message(sender, "recording_usage");
        if (action.equals("stop") && args.length != 3) return message(sender, "recording_usage");
        Player target = Bukkit.getPlayerExact(args[2]);
        if (target == null) return message(sender, "player_not_found", "player", args[2]);
        if (action.equals("stop")) return stop(sender, target);
        int label = plugin.getConfig().getString("recording.default-label", "legit").equalsIgnoreCase("cheat") ? 1 : 0;
        long seconds = 0;
        if (args.length >= 4) {
            if (isLabel(args[3])) {
                if (args.length != 4) return message(sender, "recording_usage");
                label = args[3].equalsIgnoreCase("cheat") ? 1 : 0;
            } else {
                try { seconds = RecordingDuration.parse(args[3]); }
                catch (IllegalArgumentException error) { return message(sender, "invalid_time"); }
            }
        }
        if (args.length == 5) {
            if (!isLabel(args[4])) return message(sender, "recording_usage");
            label = args[4].equalsIgnoreCase("cheat") ? 1 : 0;
        }
        plugin.getRecordingManager().start(sender, target, label, seconds);
        return true;
    }

    private boolean stop(CommandSender sender, Player target) {
        boolean alreadyNotified = plugin.getRecordingManager().willNotify(target.getUniqueId(), sender);
        if (!plugin.getRecordingManager().stop(target.getUniqueId(), true, "остановлено вручную")) {
            return message(sender, "recording_not_active", "player", target.getName());
        }
        return alreadyNotified || message(sender, "recording_stop_confirmed", "player", target.getName());
    }

    private static boolean isLabel(String value) { return value.equalsIgnoreCase("legit") || value.equalsIgnoreCase("cheat"); }
    private static boolean allowed(CommandSender sender) { return sender.hasPermission("silentac.admin") || sender.hasPermission("anguard.admin"); }
    private boolean message(CommandSender sender, String key, String... replacements) {
        sender.sendMessage(Theme.message(plugin.getConfig(), key, replacements));
        return true;
    }

    private void displayProfile(CommandSender sender, Player target) {
        PlayerProfile profile = profileManager.getProfile(target);
        if (profile == null) return;
        message(sender, "profile_header");
        for (String line : plugin.getConfig().getStringList("messages.profile_lines")) {
            sender.sendMessage(Theme.text(line.replace("{player}", Theme.escape(target.getName()))
                    .replace("{ping}", Integer.toString(target.getPing()))
                    .replace("{client}", Theme.escape(target.getClientBrandName() == null ? "Unknown" : target.getClientBrandName()))
                    .replace("{version}", Integer.toString(target.getProtocolVersion()))
                    .replace("{suspicion}", String.format(Locale.ROOT, "%.2f", profile.suspicion))
                    .replace("{buffer}", String.format(Locale.ROOT, "%.2f", profile.buffer))));
        }
        message(sender, "profile_footer");
    }

    private boolean help(CommandSender sender) {
        message(sender, "help_header");
        for (String line : plugin.getConfig().getStringList("messages.help_lines")) sender.sendMessage(Theme.text(line));
        return true;
    }

    @Override
    public List<String> onTabComplete(CommandSender sender, Command command, String alias, String[] args) {
        if (!allowed(sender) || args.length == 0) return List.of();
        List<String> choices = List.of();
        if (args.length == 1) choices = List.of("recording", "info", "profile", "help");
        else if (args[0].equalsIgnoreCase("recording")) {
            if (args.length == 2) choices = List.of("start", "stop");
            else if (args.length == 3) choices = Bukkit.getOnlinePlayers().stream().map(Player::getName).toList();
            else if (args[1].equalsIgnoreCase("start") && args.length == 4) choices = List.of("15:00", "5m", "30s", "legit", "cheat");
            else if (args[1].equalsIgnoreCase("start") && args.length == 5 && !isLabel(args[3])) choices = List.of("legit", "cheat");
        } else if (args.length == 2) choices = Bukkit.getOnlinePlayers().stream().map(Player::getName).toList();
        String prefix = args[args.length - 1].toLowerCase(Locale.ROOT);
        return choices.stream().filter(value -> value.toLowerCase(Locale.ROOT).startsWith(prefix)).toList();
    }
}
