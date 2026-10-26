package xyz.angames.anguardai.utils;

import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.minimessage.MiniMessage;
import org.bukkit.configuration.file.FileConfiguration;

public final class Theme {
    private static final MiniMessage MM = MiniMessage.miniMessage();
    private Theme() {}
    public static Component text(String input) { return MM.deserialize(input); }
    public static String escape(String input) { return MM.escapeTags(input); }
    public static Component message(FileConfiguration config, String key, String... replacements) {
        return text(config.getString("messages.prefix", "") + raw(config, key, replacements));
    }
    public static String raw(FileConfiguration config, String key, String... replacements) {
        String value = config.getString("messages." + key, "<gray>SilentAC: " + key);
        for (int i = 0; i + 1 < replacements.length; i += 2) {
            value = value.replace("{" + replacements[i] + "}", escape(replacements[i + 1]));
        }
        return value;
    }
}
