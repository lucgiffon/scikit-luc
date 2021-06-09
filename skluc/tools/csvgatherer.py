import pandas as pd
import click
import click_pathlib


@click.command()
@click.argument("results_dir", type=click_pathlib.Path(exists=True, dir_okay=True, resolve_path=True))
@click.argument("output_dir", type=click_pathlib.Path(exists=False, dir_okay=True, resolve_path=True))
@click.option("-c", "--count", is_flag=True, help="Print the number of lines in the resulting csv.")
def main(results_dir, output_dir, count):

    files = [x for x in results_dir.glob('**/*') if x.is_file()]
    lst_df = []
    lst_other_files = []
    for csv_filename in files:
        if csv_filename.suffix == ".csv":
            df = pd.read_csv(csv_filename)
            lst_df.append(df)
        else:
            lst_other_files.append(csv_filename)

    total_df = pd.concat(lst_df)
    if count:
        print(len(total_df))
    output_dir.mkdir(parents=True, exist_ok=True)
    total_df.to_csv(output_dir / "gathered_results.csv", index=False)


if __name__ == "__main__":
    main()