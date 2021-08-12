import pandas as pd
import click
import click_pathlib
import shutil


@click.command()
@click.argument("results_dir", type=click_pathlib.Path(exists=True, dir_okay=True, resolve_path=True))
@click.argument("output_dir", required=False, default=None, type=click_pathlib.Path(exists=False, dir_okay=True, resolve_path=True))
@click.option("-n", "--move-np", is_flag=True, help="Moves the .npy and .npz files together with the gathered results.")
@click.option("-d", "--delete", is_flag=True, help="Delete files after gathering.")
@click.option("-q", "--quiet", is_flag=True, help="Do not print anything.")
def main(results_dir, output_dir, move_np, delete, quiet):
    if output_dir is None:
        output_dir = results_dir
    name_gathered_file = "gathered_results.csv"

    files = [x for x in results_dir.glob('**/*') if x.is_file()]
    lst_df = []
    lst_other_files = []
    real_count = 0
    nb_other_files = 0
    nb_total_files = 0
    set_extension_other_files = set()
    for csv_filename in files:
        if csv_filename.name == name_gathered_file:
            continue
        nb_total_files += 1

        if csv_filename.suffix == ".csv":
            df = pd.read_csv(csv_filename)
            lst_df.append(df)
            real_count += 1
        else:
            nb_other_files += 1
            lst_other_files.append(csv_filename)
            set_extension_other_files.add(csv_filename.suffix)

    if real_count < 2:
        raise ValueError(f"{real_count} csv file found. Should be at least 2. Nothing done.")

    total_df = pd.concat(lst_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    if len(total_df) > 0:
        total_df.to_csv(output_dir / name_gathered_file, index=False)
    else:
        raise ValueError("No csv file found in the directory or they were empty")

    nb_other_files_moved = 0
    set_extensions_other_files = set()
    if move_np:
        for not_csv_file in lst_other_files:
            if not_csv_file.suffix in [".npy", ".npz"]:
                set_extensions_other_files.add(not_csv_file.suffix)
                shutil.copy(not_csv_file, output_dir / not_csv_file.name)
                nb_other_files_moved += 1

    nb_files_deleted = 0
    set_extensions_deleted = set()

    if delete and len(total_df) > 0:
        for csv_filename in files:
            if csv_filename.suffix == ".csv" and csv_filename.name != name_gathered_file:
                set_extensions_deleted.add(csv_filename.suffix)
                csv_filename.unlink()
                nb_files_deleted += 1

    print(f"Total nb. files: {nb_total_files}")
    print(f"Nb. CSV files gathered: {len(total_df)}")
    print(f"Nb. Other files: {nb_other_files} (extensions: {set_extension_other_files})")
    print(f"Nb. Other files moved: {nb_other_files_moved} (extensions: {set_extensions_other_files})")
    print(f"Nb. Other files deleted: {nb_files_deleted} (extensions: {set_extensions_deleted})")


if __name__ == "__main__":
    main()