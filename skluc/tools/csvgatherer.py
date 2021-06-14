import pandas as pd
import click
import click_pathlib


@click.command()
@click.argument("results_dir", type=click_pathlib.Path(exists=True, dir_okay=True, resolve_path=True))
@click.argument("output_dir", required=False, default=None, type=click_pathlib.Path(exists=False, dir_okay=True, resolve_path=True))
@click.option("-c", "--count", is_flag=True, help="Print the number of lines in the resulting csv.")
@click.option("-d", "--delete", is_flag=True, help="Delete files after gathering.")
def main(results_dir, output_dir, count, delete):
    if output_dir is None:
        output_dir = results_dir
    name_gathered_file = "gathered_results.csv"

    files = [x for x in results_dir.glob('**/*') if x.is_file()]
    lst_df = []
    lst_other_files = []
    real_count = 0
    for csv_filename in files:
        if csv_filename.name == name_gathered_file:
            continue

        if csv_filename.suffix == ".csv":
            df = pd.read_csv(csv_filename)
            lst_df.append(df)
            real_count += 1
        else:
            lst_other_files.append(csv_filename)

    if real_count < 2:
        raise ValueError(f"{real_count} csv file found. Should be at least 2. Nothing done.")

    total_df = pd.concat(lst_df)
    if count:
        print(len(total_df))
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(total_df) > 0:
        total_df.to_csv(output_dir / name_gathered_file, index=False)
    else:
        raise ValueError("No csv file found in the directory or they were empty")

    if delete and len(total_df) > 0:
        for csv_filename in files:
            if csv_filename.suffix == ".csv" and csv_filename.name != name_gathered_file:
                csv_filename.unlink()



if __name__ == "__main__":
    main()