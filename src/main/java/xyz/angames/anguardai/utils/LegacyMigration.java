package xyz.angames.anguardai.utils;

import org.bukkit.configuration.file.FileConfiguration;
import org.bukkit.configuration.file.YamlConfiguration;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

public final class LegacyMigration {
    public enum FolderResult { MOVED, BOTH_EXIST, NONE }
    private LegacyMigration() {}

    public static FolderResult prepareFolder(Path current) throws IOException {
        Path legacy = current.resolveSibling("AnGuard");
        FolderResult result = FolderResult.NONE;
        if (!Files.exists(current) && Files.isDirectory(legacy)) {
            Files.move(legacy, current);
            result = FolderResult.MOVED;
        } else if (Files.exists(current) && Files.isDirectory(legacy)) {
            result = FolderResult.BOTH_EXIST;
        }
        Files.createDirectories(current);
        return result;
    }

    public static boolean applyTheme(FileConfiguration config, Path directory, InputStream resource) throws IOException {
        if (config.contains("appearance.version", true) && config.getInt("appearance.version") >= 1) return false;
        if (resource == null) throw new IllegalArgumentException("Missing default configuration");
        Path backup = directory.resolve("config-before-silentac.yml");
        if (!Files.exists(backup)) Files.copy(directory.resolve("config.yml"), backup, StandardCopyOption.COPY_ATTRIBUTES);
        var defaults = YamlConfiguration.loadConfiguration(new InputStreamReader(resource, StandardCharsets.UTF_8));
        var messages = defaults.getConfigurationSection("messages");
        if (messages == null) throw new IllegalArgumentException("Missing default messages");
        config.set("messages", null);
        messages.getValues(false).forEach((key, value) -> config.set("messages." + key, value));
        String command = config.getString("autoban.command");
        if (command != null) config.set("autoban.command", command.replace("AnGuard", "SilentAC"));
        config.set("appearance.version", 1);
        return true;
    }
}
