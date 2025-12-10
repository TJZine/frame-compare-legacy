"""Security tests for slowpics webhook URL validation."""

from __future__ import annotations

from src.frame_compare.slowpics import _is_safe_webhook_url


class TestWebhookUrlValidation:
    """Tests for webhook URL security validation."""

    def test_accepts_valid_https_url(self) -> None:
        assert _is_safe_webhook_url("https://discord.com/api/webhooks/12345/abcdef") is True

    def test_accepts_valid_https_with_port(self) -> None:
        assert _is_safe_webhook_url("https://example.com:8443/webhook") is True

    def test_rejects_http_url(self) -> None:
        assert _is_safe_webhook_url("http://example.com/webhook") is False

    def test_rejects_localhost(self) -> None:
        assert _is_safe_webhook_url("https://localhost/webhook") is False
        assert _is_safe_webhook_url("https://localhost:8080/webhook") is False

    def test_rejects_loopback_ipv4(self) -> None:
        assert _is_safe_webhook_url("https://127.0.0.1/webhook") is False
        assert _is_safe_webhook_url("https://127.0.0.1:443/webhook") is False

    def test_rejects_loopback_ipv6(self) -> None:
        assert _is_safe_webhook_url("https://[::1]/webhook") is False

    def test_rejects_zero_address(self) -> None:
        assert _is_safe_webhook_url("https://0.0.0.0/webhook") is False

    def test_rejects_private_ip_rfc1918_class_a(self) -> None:
        assert _is_safe_webhook_url("https://10.0.0.1/webhook") is False
        assert _is_safe_webhook_url("https://10.255.255.255/webhook") is False

    def test_rejects_private_ip_rfc1918_class_b(self) -> None:
        assert _is_safe_webhook_url("https://172.16.0.1/webhook") is False
        assert _is_safe_webhook_url("https://172.31.255.255/webhook") is False

    def test_rejects_private_ip_rfc1918_class_c(self) -> None:
        assert _is_safe_webhook_url("https://192.168.0.1/webhook") is False
        assert _is_safe_webhook_url("https://192.168.255.255/webhook") is False

    def test_rejects_link_local(self) -> None:
        assert _is_safe_webhook_url("https://169.254.169.254/webhook") is False

    def test_rejects_empty_url(self) -> None:
        assert _is_safe_webhook_url("") is False

    def test_rejects_malformed_url(self) -> None:
        assert _is_safe_webhook_url("not-a-url") is False

    def test_rejects_missing_scheme(self) -> None:
        assert _is_safe_webhook_url("example.com/webhook") is False

    def test_accepts_public_domain(self) -> None:
        assert _is_safe_webhook_url("https://hooks.slack.com/services/T00000/B00000/XXXX") is True

    def test_accepts_public_ip(self) -> None:
        # Example public IP (Google DNS)
        assert _is_safe_webhook_url("https://8.8.8.8/webhook") is True
