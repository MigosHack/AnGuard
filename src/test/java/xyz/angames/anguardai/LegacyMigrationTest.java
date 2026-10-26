package xyz.angames.anguardai;

import org.bukkit.configuration.file.YamlConfiguration;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import xyz.angames.anguardai.utils.LegacyMigration;
import java.nio.file.Files;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;

class LegacyMigrationTest {
    @TempDir Path directory;
    @Test void movesLegacyDirectoryWithRecordingsAndNeverMergesExistingDirectories() throws Exception {
        Path legacy = directory.resolve("AnGuard");
        Files.createDirectories(legacy.resolve("training"));
        Files.writeString(legacy.resolve("training/keep.jsonl"), "preserved data");
        Files.writeString(legacy.resolve("config.yml"), "preserved settings");
        Path current = directory.resolve("SilentAC");
        assertEquals(LegacyMigration.FolderResult.MOVED, LegacyMigration.prepareFolder(current));
        assertFalse(Files.exists(legacy));
        assertEquals("preserved data", Files.readString(current.resolve("training/keep.jsonl")));
        Files.createDirectories(legacy);
        Files.writeString(legacy.resolve("config.yml"), "different settings");
        assertEquals(LegacyMigration.FolderResult.BOTH_EXIST, LegacyMigration.prepareFolder(current));
        assertEquals("preserved settings", Files.readString(current.resolve("config.yml")));
        assertEquals("different settings", Files.readString(legacy.resolve("config.yml")));
    }
    @Test void backsUpThemePreservingSafetySettingsAndDoesNotOverwriteLaterCustomizations() throws Exception {
        String old = "vds:\n  port: 9000\nai:\n  observe-only: false\nautoban:\n  enable: false\n  command: 'kick {player} [AnGuard]'\nmessages:\n  prefix: '<red>AnGuard'\n";
        Files.writeString(directory.resolve("config.yml"), old);
        var config = YamlConfiguration.loadConfiguration(directory.resolve("config.yml").toFile());
        // Simulate JavaPlugin adding new resource defaults to an old configuration.
        var defaults = YamlConfiguration.loadConfiguration(new java.io.InputStreamReader(getClass().getResourceAsStream("/config.yml")));
        config.setDefaults(defaults);
        try (var resource = getClass().getResourceAsStream("/config.yml")) {
            assertTrue(LegacyMigration.applyTheme(config, directory, resource));
        }
        assertEquals(old, Files.readString(directory.resolve("config-before-silentac.yml")));
        assertEquals(9000, config.getInt("vds.port"));
        assertFalse(config.getBoolean("ai.observe-only"));
        assertFalse(config.getBoolean("autoban.enable"));
        assertTrue(config.getString("messages.prefix").contains("SilentAC"));
        assertFalse(config.getString("messages.prefix").contains("<red>"));
        config.set("messages.prefix", "custom gray theme");
        try (var resource = getClass().getResourceAsStream("/config.yml")) {
            assertFalse(LegacyMigration.applyTheme(config, directory, resource));
        }
        assertEquals("custom gray theme", config.getString("messages.prefix"));
        assertEquals(old, Files.readString(directory.resolve("config-before-silentac.yml")));
    }
}
