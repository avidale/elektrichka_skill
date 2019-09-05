import logging
import tgalice

from dialog_manager import RaspDialogManager

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    mongo_db = tgalice.storage.database_utils.get_mongo_or_mock()
    connector = tgalice.dialog_connector.DialogConnector(
        dialog_manager=RaspDialogManager(world_filename='data/world.pkl'),
        storage=tgalice.session_storage.MongoBasedStorage(database=mongo_db, collection_name='sessions'),
        log_storage=tgalice.storage.message_logging.MongoMessageLogger(
            database=mongo_db, collection_name='message_logs', detect_pings=True
        )
    )
    server = tgalice.flask_server.FlaskServer(connector=connector)
    server.parse_args_and_run()
