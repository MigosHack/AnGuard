package xyz.angames.anguardai;

import org.junit.jupiter.api.Test;
import xyz.angames.anguardai.client.ApiEndpoint;
import static org.junit.jupiter.api.Assertions.*;

class ApiEndpointTest {
    @Test void supportsHttpsBaseUrlAndKeepsLegacyLoopback() {
        assertEquals("https://ac.example.com/predict_binary", ApiEndpoint.resolve("https://ac.example.com/", "127.0.0.1", 8000).toString());
        assertEquals("https://ac.example.com:8443/api/predict_binary", ApiEndpoint.resolve("https://ac.example.com:8443/api/", null, 0).toString());
        assertEquals("http://127.0.0.1:8000/predict_binary", ApiEndpoint.resolve("", "127.0.0.1", 8000).toString());
        assertEquals("http://[::1]:8000/predict_binary", ApiEndpoint.resolve(null, "::1", 8000).toString());
    }
    @Test void rejectsRemotePlaintextAndAmbiguousAddresses() {
        for (String url : new String[]{"http://ac.example.com", "http://203.0.113.1:8000", "ftp://ac.example.com", "https://user:pass@ac.example.com", "https://ac.example.com?key=bad", "https://ac.example.com/#fragment", "ac.example.com", "https://ac.example.com:0", "https://ac.example.com:99999"}) {
            assertThrows(IllegalArgumentException.class, () -> ApiEndpoint.resolve(url, "127.0.0.1", 8000), url);
        }
        assertThrows(IllegalArgumentException.class, () -> ApiEndpoint.resolve("", "203.0.113.1", 8000));
    }
}
