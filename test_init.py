from dialog_manager import RaspDialogManager
from tgalice.testing.testing_utils import make_context


def test_init():
    dm = RaspDialogManager('data/world.pkl')
    ctx = make_context(new_session=True)
    resp = dm.respond(ctx)
    assert resp.text.startswith('Привет! Назовите')
