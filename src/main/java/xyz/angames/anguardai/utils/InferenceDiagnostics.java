package xyz.angames.anguardai.utils;

import org.bukkit.entity.Player;
import xyz.angames.anguardai.data.PlayerProfile;
import java.util.Locale;

public final class InferenceDiagnostics {
    private InferenceDiagnostics() {}
    public static boolean bypassed(Player player) {
        return player.hasPermission("silentac.bypass") || player.hasPermission("anguard.bypass");
    }
    public static String score(PlayerProfile profile) {
        if (profile.predictionResponses == 0) return "—";
        return String.format(Locale.ROOT, profile.suspicion > 0 && profile.suspicion < .001 ? "%.2e%%" : "%.3f%%", profile.suspicion);
    }
    public static String status(PlayerProfile profile, boolean enabled, boolean recording, boolean bypass,
                                int ping, double tps, long now) {
        if (!enabled) return "Aim отключён";
        if (recording) return "Запись: оценка не запрашивается";
        if (bypass) return "Bypass: оценка не запрашивается";
        if (ping > 200 || tps < 18) return "Пауза: ping/TPS";
        if (now - profile.lastTeleportTime < 2000) return "Пауза после телепорта";
        if (profile.windowSamples < 40) return "Сбор окна: " + profile.windowSamples + "/40";
        if (profile.predictionPending.get()) return "Ожидание API";
        if (profile.inferenceError) return profile.lastApiStatus > 0 ? "Ошибка API/ответа: HTTP " + profile.lastApiStatus : "Ошибка соединения с API";
        if (profile.predictionResponses == 0) return "Ожидание первого ответа API";
        return now - profile.lastPredictionTime > 15000 ? "Нет свежей оценки" : "API отвечает";
    }
}
