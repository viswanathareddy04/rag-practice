from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_status():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data

def test_api_get_cards():
    response = client.get("/api/cards")
    assert response.status_code == 200
    cards = response.json()
    assert isinstance(cards, list)
    assert len(cards) >= 1
    td_card = next((c for c in cards if "td_aeroplan_vip" in c["card_id"]), None)
    assert td_card is not None
    assert td_card["is_indexed"] is True

def test_api_query_stream():
    payload = {
        "card_id": "td_aeroplan_vip",
        "query": "What is the flight delay coverage limit?"
    }
    with client.stream("POST", "/api/query", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = []
        for line in response.iter_lines():
            if line:
                events.append(line)
        assert any("event: context" in e or "event: token" in e for e in events)
