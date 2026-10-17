from glm_review.client import ClientConfig, GLMClient


def test_payload_contains_reasoning_configuration():
    client = GLMClient(ClientConfig(model="glm-test"))
    payload = client.build_payload(
        [{"role": "user", "content": "test"}],
        n=8,
        max_tokens=123,
        reasoning_effort="high",
    )
    assert payload["model"] == "glm-test"
    assert payload["n"] == 8
    assert payload["max_tokens"] == 123
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "high",
    }


def test_auth_header_is_optional():
    assert "Authorization" not in GLMClient(ClientConfig())._headers()
    headers = GLMClient(ClientConfig(api_key="secret"))._headers()
    assert headers["Authorization"] == "Bearer secret"
