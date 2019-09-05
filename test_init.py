import tgalice
from dialog_manager import RaspDialogManager
# from tgalice.testing.testing_utils import make_context


def make_context(text='', prev_response=None, new_session=False):
    if prev_response is not None:
        user_object = prev_response.updated_user_object
    else:
        user_object = {}
    if new_session:
        metadata = {'new_session': True}
    else:
        metadata = {}
    return tgalice.dialog.Context(user_object=user_object, metadata=metadata, message_text=text)


def test_init():
    dm = RaspDialogManager('data/world.pkl')
    ctx = make_context(new_session=True)
    resp = dm.respond(ctx)
    assert resp.text.startswith('Привет! Назовите')
