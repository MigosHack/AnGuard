package xyz.angames.anguardai.client;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.Locale;

public final class ApiEndpoint {
    private ApiEndpoint() {}

    public static URI resolve(String configuredUrl, String legacyHost, int legacyPort) {
        try {
            URI base = configuredUrl != null && !configuredUrl.isBlank() ? new URI(configuredUrl.trim())
                    : new URI("http", null, legacyHost, legacyPort, null, null, null);
            String scheme = base.getScheme() == null ? "" : base.getScheme().toLowerCase(Locale.ROOT);
            String host = base.getHost();
            if (host == null || base.getUserInfo() != null || base.getQuery() != null || base.getFragment() != null
                    || base.getPort() == 0 || base.getPort() > 65535) {
                throw new IllegalArgumentException("Invalid vds.url: use an API base URL without credentials, query or fragment");
            }
            boolean loopback = host.equalsIgnoreCase("localhost") || host.equals("127.0.0.1")
                    || host.equals("::1") || host.equals("[::1]");
            if (!scheme.equals("https") && !(scheme.equals("http") && loopback)) {
                throw new IllegalArgumentException("Remote API requires an HTTPS vds.url; HTTP is supported only on loopback");
            }
            String path = base.getPath() == null ? "" : base.getPath().replaceAll("/+$", "");
            return new URI(scheme, null, host, base.getPort(), path + "/predict_binary", null, null);
        } catch (URISyntaxException error) {
            throw new IllegalArgumentException("Invalid API address in vds.url / vds.ip", error);
        }
    }
}
