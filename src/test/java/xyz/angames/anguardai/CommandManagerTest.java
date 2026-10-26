package xyz.angames.anguardai;

import org.bukkit.Bukkit;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.configuration.file.YamlConfiguration;
import org.bukkit.entity.Player;
import org.junit.jupiter.api.Test;
import xyz.angames.anguardai.commands.CommandManager;
import xyz.angames.anguardai.managers.*;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import static org.mockito.Mockito.*;

class CommandManagerTest {
    @Test void routesNewSyntaxLabelsAndLegacyCommandsWithoutAcceptingBadTime() {
        SilentAC plugin = mock(SilentAC.class);
        var config = YamlConfiguration.loadConfiguration(new InputStreamReader(getClass().getResourceAsStream("/config.yml"), StandardCharsets.UTF_8));
        when(plugin.getConfig()).thenReturn(config);
        RecordingManager recording = mock(RecordingManager.class);
        when(plugin.getRecordingManager()).thenReturn(recording);
        CommandSender admin = mock(CommandSender.class);
        when(admin.hasPermission("silentac.admin")).thenReturn(true);
        Player steve = mock(Player.class);
        when(steve.getName()).thenReturn("Steve");
        var commands = new CommandManager(plugin, mock(InfoManager.class), mock(PlayerProfileManager.class));
        try (var bukkit = mockStatic(Bukkit.class)) {
            bukkit.when(() -> Bukkit.getPlayerExact("Steve")).thenReturn(steve);
            Command command = mock(Command.class);
            commands.onCommand(admin, command, "anguard", new String[]{"recording", "start", "Steve", "15:00"});
            verify(recording).start(admin, steve, 0, 900);
            commands.onCommand(admin, command, "silentac", new String[]{"recording", "start", "Steve", "5m", "cheat"});
            verify(recording).start(admin, steve, 1, 300);
            commands.onCommand(admin, command, "anguard", new String[]{"record", "Steve", "cheat"});
            verify(recording).start(admin, steve, 1, 0);
            clearInvocations(recording);
            commands.onCommand(admin, command, "anguard", new String[]{"recording", "start", "Steve", "oops"});
            commands.onCommand(admin, command, "anguard", new String[]{"recording", "start", "Steve", "0"});
            commands.onCommand(admin, command, "anguard", new String[]{"recording", "start", "Steve", "15", "unknown"});
            verifyNoInteractions(recording);
            commands.onCommand(admin, command, "anguard", new String[]{"recording", "stop", "Steve"});
            verify(recording).stop(steve.getUniqueId(), true, "остановлено вручную");
        }
    }
}
