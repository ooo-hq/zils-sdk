"""Exercise clients against the existing gateway and its local runtime fixture."""

from contextlib import contextmanager
from types import SimpleNamespace

from tests.test_api import FixtureStore, GatewayTest
from tests.test_decision_http import server
from zils_platform import decision_http

QUESTIONS = {
    "eligible": {"type": "noul", "instructions": "Is a refund requested?"},
    "team": {"type": "choice", "criteria": {"billing": None, "support": None}},
    "urgency": {"type": "score", "criteria": [{"level": "low"}, ["high"]]},
}


@contextmanager
def api():
    fixture = GatewayTest()
    fixture.setUp()
    fixture.billable_tokens = 73
    store = FixtureStore()
    calls = []
    try:
        with server(decision_http, fixture.runtime) as port:
            gateway = fixture.metered_gateway(store, port)

            def dispatch(*args):
                calls.append(args)
                return gateway.dispatch(*args)

            with server(decision_http, dispatch) as gateway_port:
                yield SimpleNamespace(
                    url=f"http://127.0.0.1:{gateway_port}",
                    store=store,
                    calls=calls,
                )
    finally:
        fixture.tearDown()
