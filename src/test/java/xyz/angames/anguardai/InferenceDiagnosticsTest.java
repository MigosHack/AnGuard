package xyz.angames.anguardai;

import org.junit.jupiter.api.Test;
import xyz.angames.anguardai.data.PlayerProfile;
import xyz.angames.anguardai.utils.InferenceDiagnostics;
import static org.junit.jupiter.api.Assertions.*;

class InferenceDiagnosticsTest {
    @Test void unknownScoreIsDistinctFromActualZeroAndSmallScoresRemainVisible() {
        PlayerProfile p = new PlayerProfile(0, 0);
        assertEquals("—", InferenceDiagnostics.score(p));
        p.predictionResponses = 1;
        assertEquals("0.000%", InferenceDiagnostics.score(p));
        p.suspicion = .00000001;
        assertEquals("1.00e-08%", InferenceDiagnostics.score(p));
        p.suspicion = 51.1234;
        assertEquals("51.123%", InferenceDiagnostics.score(p));
    }
    @Test void explainsWhyNoRequestsAreSentBeforeLookingAtScore() {
        PlayerProfile p = new PlayerProfile(0, 0);
        assertTrue(InferenceDiagnostics.status(p, false, false, false, 50, 20, 10_000).contains("отключён"));
        assertTrue(InferenceDiagnostics.status(p, true, true, false, 50, 20, 10_000).contains("Запись"));
        assertTrue(InferenceDiagnostics.status(p, true, false, true, 50, 20, 10_000).contains("Bypass"));
        assertTrue(InferenceDiagnostics.status(p, true, false, false, 201, 20, 10_000).contains("ping/TPS"));
        assertTrue(InferenceDiagnostics.status(p, true, false, false, 50, 17, 10_000).contains("ping/TPS"));
        assertEquals("Сбор окна: 0/40", InferenceDiagnostics.status(p, true, false, false, 50, 20, 10_000));
    }
    @Test void distinguishesPendingErrorsSuccessfulAndStaleResponses() {
        PlayerProfile p = new PlayerProfile(0, 0);
        p.windowSamples = 40;
        p.predictionPending.set(true);
        assertEquals("Ожидание API", InferenceDiagnostics.status(p, true, false, false, 50, 20, 10_000));
        p.predictionPending.set(false); p.inferenceError = true; p.lastApiStatus = 401;
        assertTrue(InferenceDiagnostics.status(p, true, false, false, 50, 20, 10_000).contains("401"));
        p.lastApiStatus = -1;
        assertTrue(InferenceDiagnostics.status(p, true, false, false, 50, 20, 10_000).contains("соединения"));
        p.inferenceError = false; p.predictionResponses = 1; p.lastPredictionTime = 10_000;
        assertEquals("API отвечает", InferenceDiagnostics.status(p, true, false, false, 50, 20, 10_000));
        assertEquals("Нет свежей оценки", InferenceDiagnostics.status(p, true, false, false, 50, 20, 30_000));
    }
}
