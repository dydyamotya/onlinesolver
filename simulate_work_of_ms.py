import argparse
import pathlib

import time
import itertools
import shutil
import logging

logger = logging.getLogger(__name__)


def get_number_of_files_in_work_dir(work_dir: pathlib.Path):
    return len(tuple(work_dir.iterdir()))

def simulate(files_dir, work_dir):
    work_dir = pathlib.Path(work_dir)
    files_dir = pathlib.Path(files_dir)
    files_iterator = itertools.cycle(files_dir.iterdir())
    prev_number_of_files = get_number_of_files_in_work_dir(work_dir)
    while True:
        try:
            current_number_of_files = get_number_of_files_in_work_dir(work_dir)
            logger.debug(f"{current_number_of_files} {prev_number_of_files}")
            if current_number_of_files < prev_number_of_files or current_number_of_files == 0:
                file = next(files_iterator)
                shutil.copy(file, work_dir / file.name)
                logger.info(f"Copied {file} to {work_dir / file.name}")

            prev_number_of_files = get_number_of_files_in_work_dir(work_dir)
            time.sleep(1)
        except KeyboardInterrupt:
            break
        except:
            raise




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_simulation_files", action="store")
    parser.add_argument("path_to_pseudowork_directory", action="store")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    simulate(args.path_to_simulation_files, args.path_to_pseudowork_directory)