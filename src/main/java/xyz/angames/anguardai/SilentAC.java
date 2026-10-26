package xyz.angames.anguardai;

import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitRunnable;
import xyz.angames.anguardai.client.AIServer;
import xyz.angames.anguardai.commands.CommandManager;
import xyz.angames.anguardai.listeners.PlayerListener;
import xyz.angames.anguardai.managers.*;


public class SilentAC extends JavaPlugin {
    private static SilentAC instance;
    private InfoManager infoManager;
    private PlayerProfileManager playerProfileManager;
    private PacketManager packetManager;
    private AIServer aiServer;
    private TrainingRecorder trainingRecorder;
    private RecordingManager recordingManager;

    @Override
    public void onEnable() {
        instance = this;
        try {
            migrateDataFolder();
            saveDefaultConfig();
            migrateTheme();
            getLogger().info("SilentAC: starting...");
            trainingRecorder = new TrainingRecorder(this);
            playerProfileManager = new PlayerProfileManager(this);
            recordingManager = new RecordingManager(this);
            infoManager = new InfoManager(this, playerProfileManager);
            aiServer = new AIServer(this);
            packetManager = new PacketManager(this, playerProfileManager);
            if (getCommand("silentac") != null) {
                CommandManager commands = new CommandManager(this, infoManager, playerProfileManager);
                getCommand("silentac").setExecutor(commands);
                getCommand("silentac").setTabCompleter(commands);
            }
            getServer().getPluginManager().registerEvents(new PlayerListener(playerProfileManager), this);
            getServer().getPluginManager().registerEvents(recordingManager, this);
            getServer().getOnlinePlayers().forEach(playerProfileManager::createProfile);
            new BukkitRunnable() {
                @Override
                public void run() {
                    recordingManager.tick();
                    playerProfileManager.tickAllProfiles();
                }
            }.runTaskTimer(this, 0L, 1L);
            getLogger().info("SilentAC: enabled.");
        } catch (Exception error) {
            getLogger().severe("SilentAC could not start: " + error.getMessage());
            error.printStackTrace();
            getServer().getPluginManager().disablePlugin(this);
        }
    }

    private void migrateDataFolder() throws Exception {
        switch (xyz.angames.anguardai.utils.LegacyMigration.prepareFolder(getDataFolder().toPath())) {
            case MOVED -> getLogger().info("Moved existing AnGuard data to SilentAC; recordings and configuration retained.");
            case BOTH_EXIST -> getLogger().warning("Both SilentAC and AnGuard data folders exist; using SilentAC without merging or deleting legacy files.");
            case NONE -> { }
        }
    }

    private void migrateTheme() throws Exception {
        try (var resource = getResource("config.yml")) {
            if (xyz.angames.anguardai.utils.LegacyMigration.applyTheme(getConfig(), getDataFolder().toPath(), resource)) {
                saveConfig();
                getLogger().info("Applied SilentAC white/gray theme. Previous configuration retained in config-before-silentac.yml.");
            }
        }
    }

    @Override
    public void onDisable() {
        if (recordingManager != null) recordingManager.shutdown();
        if (trainingRecorder != null) trainingRecorder.close();
        if (infoManager != null) infoManager.shutdown();
        getLogger().info("SilentAC: disabled.");
    }

    public static SilentAC getInstance() { return instance; }
    public static JavaPlugin getSpigotLoader() { return instance; }
    public PlayerProfileManager getPlayerProfileManager() { return playerProfileManager; }
    public InfoManager getInfoManager() { return infoManager; }
    public PacketManager getPacketManager() { return packetManager; }
    public AIServer getAiServer() { return aiServer; }
    public TrainingRecorder getTrainingRecorder() { return trainingRecorder; }
    public RecordingManager getRecordingManager() { return recordingManager; }
}
