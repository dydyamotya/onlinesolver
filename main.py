from gui import main
import argparse
import pathlib
import datetime
import logging

if __name__ == '__main__':
    # Main App
    parser = argparse.ArgumentParser()
    parser.add_argument("--modbus", action="store_true")
    parser.add_argument("--isclass", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    cwd = pathlib.Path().cwd()
    logs_folder = cwd / "logs"
    config_file = cwd / "config.conf"
    pictures_folder = cwd / "pictures"
    logs_folder.mkdir(exist_ok=True)
    logging_file_name = (logs_folder / datetime.datetime.now().strftime("%y%m%d_%H%M%S")).with_suffix(".log")
    logging.basicConfig(filename=logging_file_name.as_posix(),
                        filemode='a',
                        level=logging.DEBUG if args.debug else logging.INFO,
                        datefmt="%y%m%d_%H:%M:%S",
                        format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
    logger.addHandler(stream_handler)

    main(cwd, config_file, modbus_support=args.modbus, is_class=args.isclass, debug=args.debug, pictures_folder=pictures_folder)
