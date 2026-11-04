"""Module for getting feature matrices."""

# ==
# author: uSerratos
# date: Nov 2024
# ==

import os
import sys
from argparse import ArgumentParser, Namespace

# Only import utils if it hasn't been imported already
if 'utils' not in sys.modules:
    if os.environ.get('RUNNING_AS_SUBPROCESS') != 'true':
        os.system("cls||clear")
    from utils import *


ASK_OPTIONS: List[str] = ["coda", "openpose", "template"]

@dataclass(slots=True, kw_only=True)
class Args:
    """Arguments object

    Attributes:
    ------
        data: str = None
        sample_rate: int = None
        output: str = None
        pattern: list = None
        stickfigure_mode: str = None

    """

    data: str = None
    files_list: bool = False
    sample_rate: int = None
    output_dir: str = None
    pattern: bool = False
    stickfigure_mode: bool = False
    curated_list: str = None


def fetch_args() -> dict:
    if len(sys.argv) > 1:
        parser = ArgumentParser()

        parser.usage = "This code does feature matrix batch processing"

        parser.add_argument( "-d", "--data", help="Provide directory where study is located", type=str, nargs="?", required=True,)
        parser.add_argument( "-l", "--files_list", help="Input is csv list of files to process", type=str, nargs="?", required=False, default=None,)
        parser.add_argument( "-sr", "--sample_rate", help="Data collection sample rate", type=int, nargs="?", default=100, required=True,)
        parser.add_argument( "-od", "--output_dir", help="Output directory", nargs="?", type=str, required=True, default=None,)
        parser.add_argument( "-p", "--pattern", help="Data search pattern", type=str, nargs="?", required=False, default="/*",)
        parser.add_argument( "-sm", "--stickfigure_mode", help="Choose stickfigure mode from stickfigure template", type=str, nargs="?", required=True,)
        parser.add_argument( "-c", "--curated_list", help="Curated list of files to process as a csv file", type=str, nargs="?", required=False, default=None,)

        args = parser.parse_args()

    else:
        ad: dict = dict.fromkeys(
            {
                "data",
                "files_list",
                "output_dir",
                "stickfigure_mode",
                "sample_rate",
                "pattern",
            }
        )
        ad.update({"data": os.path.normpath(input("\n Study directory: \n => "))})
        ad.update({"files_list": ask_question_options(question="If you have a target file list, select file_list, otherwise select path", options=["path", "files_list"],)})
        ad.update({"output_dir": os.path.normpath(input("\n Output directory: \n => "))})
        ad.update({"sample_rate": int(input("\n Data collection sample rate: \n => "))})
        ad.update({"stickfigure_mode": ask_question_options(question="Which stickfigure should be used", options=["coda", "openpose", "template"],)})

        if ad["files_list"] in "files_list":
            ad.update(
            {"pattern": input("\n Data file search pattern: \n => ")}
            )
            ad.update(
                    {
                    "files_list": os.path.normpath(
                        input("\n Provide target files list: \n => ")
                    )
                }
            )
        else:
            ad.update({"files_list": None})
            ad.update({"pattern": "/*"})

        args: Args = Args(
            data=ad["data"],
            files_list=ad["files_list"],
            output_dir=ad["output_dir"],
            pattern=ad["pattern"],
            stickfigure_mode=ad["stickfigure_mode"],
            sample_rate=ad["sample_rate"],
        )

    message = (
        f"\n FEATURE MATRIX BATCH PROCESSING PARAMETERS: \n"
        + f"\n  [study directory] => {args.data}"
        + f"\n  [files list] => {args.files_list}"
        + f"\n  [output directory] => {args.output_dir}"
        + f"\n  [data sample rate] => {args.sample_rate}"
        + f"\n  [stickfigure mode] => {args.stickfigure_mode}"
        + f"\n  [File search pattern] => {args.pattern}"
    )

    print_color(message, color=bold_color.GREEN)

    message = (
        f"|{__file__}| Input directory: {args.data}"
    )
    logger.info(message)

    return args


def list_and_create(
    main_path: str,
    output: str,
    search_pattern: str | list = None,
) -> FilesList:
    files_list: FilesList = list_directory_items(
        file_or_directory=main_path,
        recursive=True,
        search_pattern=search_pattern,
    )

    files_list_new: list = [
        os.path.join(output, os.path.relpath(i, main_path))
        for i in files_list["unique_parents"]
    ]

    for f in files_list_new:
        try:
            if f != main_path:
                os.makedirs(f, exist_ok=True)
        except Exception as e:
            logger.critical(
                f"|{__file__}| Falied to create {f} directory. Reason {e}"
            )

    return files_list


@tictoc
def main():
    # === FETCH INPUTS === #
    args = fetch_args()

    # === PARAMETERS INIT === #
    output_path = os.path.join(args.output_dir, "feature_matrix")
    output_path = create_directory(directory=output_path)
    search_pattern = [args.pattern, "T", ".csv"]

    if args.curated_list:
        curated_list = pd.read_csv(args.curated_list)
        curated_list = list(curated_list.groupby(["ppid", "session", "trial"]).groups.keys())

    if not args.files_list:
        files_list = list_and_create(
            main_path=args.data,
            output=output_path,
            search_pattern=search_pattern,
        )
    else:
        files_list_ = list_and_create(
            main_path=args.data,
            output=output_path,
            search_pattern=search_pattern,
        )["items"]

        target_files_list_ = pd.read_csv(args.files_list, header=None)

        new_target_files_list_ = []
        for i, e in enumerate(target_files_list_.values):
            path = list(Path(e[0]).parts[-3:])
            p2 = "_" + path[-1].split("_")[-1]
            p1 = f"{path[0]}\{path[1]}"
            new_target_files_list_.append([p1, p2])

        target_files_list = []
        for i in files_list_:
            for j in new_target_files_list_:
                if (j[0] in i) and (j[1] in i):
                    target_files_list.append(i)

        unique_parents = natsorted(
            {os.path.dirname(os.path.normpath(i)) for i in target_files_list}
        )
        prefix = os.path.commonpath(unique_parents)

        files_list: FilesList = {
            "items": target_files_list,
            "mode": None,
            "items_abs": None,
            "unique_parents": unique_parents,
        }

    # === BATCH PROCESSING === #
    for i in tqdm(files_list["items"], desc="Processing files", unit="file", leave=False):
        logger.info(f"|{__file__}| Processing: {i}")

        if args.files_list:
            output_name = os.path.join(
                output_path, os.path.relpath(i, prefix)
            )
        else:
            output_name = os.path.join(
                output_path, os.path.relpath(i, args.data)
            )
        
        output_directory = os.path.dirname(output_name)

        stickfigure = StickFigure(
            body=skf[args.stickfigure_mode]["body"],
            data_mapping=skf[args.stickfigure_mode]["data_mapping"],
        )

        current_file = os.path.basename(i)

        path_parts = Path(i).parts
        trial_number = re.search(r'T\d{3}', current_file).group()
        session_number = re.search(r'S\d{3}', i).group()
        ppid = path_parts[path_parts.index(session_number) - 1]
        current_ppid_session_trial = (int(ppid), int(session_number[-3:]), int(trial_number[-3:]))

        if args.curated_list:
            if current_ppid_session_trial not in curated_list:
                print_color(f"|{__file__}| Skipping {i} because it is not in the curated list", bold_color.YELLOW)
                continue

        file_name = f"feature_matrix_{trial_number}"

        try:
            feature_matrix(
                file=i,
                stickfigure=stickfigure,
                data_rate=args.sample_rate,
                save=True,
                file_name=file_name,
                output_directory=output_directory,
            )
        except Exception as e:
            message = (
                f"|feature_matrix_batch_processing.py| [feature matrix]"
                + f"{i} failed, reason {e}"
            )
            logger.critical(message)

    print_color("\nFeature matrix completed successfully!".capitalize(), bold_color.BLUE)
    print_color(f"\nThe data can be found in: ".capitalize() + f"{output_path}", bold_color.BLUE)


if __name__ == "__main__":
    print_welcome_message()
    main()
