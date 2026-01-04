#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np


def readin_dataset_csv(files: list[str]):
    """ Load all identified files into a single DataFrame.

    Args:
        files (list[str]): The list of file paths.
    """

    if len(files) < 1:
        raise AttributeError(name="Files")

    first = True

    for file in files:
        if first:
            df = pd.read_csv(file)
        else:
            df.concat(pd.read_csv(file))

    return df


def export_dataset_csv(df: pd.DataFrame, target_file: str) -> bool:
    """ Optionally export a DataFrame to a CSV file. If file is empty, do not export.
    
    Args:
        df (pd.DataFrame): DataFrame to export
        target_file (str): Path to the target CSV file
    """

    # check if the target file is empty
    if target_file is None:
        return False
    
    target_file = target_file.strip()
    if target_file == "":
        return False
    
    # export to CSV with headers
    df.to_csv(target_file, index=False, header=True)
    print(f"Exported {len(df)} rows to {target_file}.")

    return True


def remove_bot_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove 'actor' rows from the DataFrame that are identified as bots.
    Args:
        df (pd.DataFrame): DataFrame to remove rows from.
    Returns:
        pd.DataFrame: The adjusted DataFrame.
    """

    bots_to_remove = [
        '[bot]',
        'deploy_key'
        # 'iiac-at-shell-reader',
        # 'ITSO-siti-cpe-team-frontera-github',
        # 'iiac-at-shell'
    ]

    # remove rows where the 'actor' has '[bot]' in the name or other known bots
    for bot in bots_to_remove:
        df = df[~df['actor'].str.contains(bot)]

    return df


def get_min_med_max_from_dataframe(df: pd.DataFrame) -> tuple[int, int, int]:
    """Get the minimum, median, and maximum values from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to get the values from.

    Returns:
        tuple[int, int, int]: The minimum, median, and maximum values.
    """

    min_value = df.min().min()
    max_value = df.max().max()
    med_value = df.median().median()

    return (min_value, med_value, max_value)


def bin_all_values_in_dataframe(df: pd.DataFrame, bin_start: int, bin_end: int, bin_increment: int) -> list[tuple[int, int]]:
    """Bin all values in a DataFrame into a set of bins.

    Args:
        df (pd.DataFrame): The DataFrame to bin.
        bin_start (int): The starting value of the bins.
        bin_end (int): The ending value of the bins.
        bin_increment (int): The increment value of the bins.

    Returns:
        A list of bins where each item is a list of the bin exclusive end and count.
    """

    bins = np.arange(bin_start, bin_end, bin_increment)
    binned_values = np.digitize(df.values.flatten(), bins)
    binned_counts = np.bincount(binned_values)

    # add the last bin
    bins = np.append(bins, bin_end)

    return [(bins[i], binned_counts[i]) for i in range(len(binned_counts))]


def get_markdown_table_for_list(items: list, title: str, column_number: int):

    # make a copy to make sure we don't overwrite the original
    items = items.copy()

    markdown = f"| {title} |"
    for num in range(1, column_number):
        markdown += " |"
    markdown += "\n"

    markdown += "| --- |"
    for num in range(1, column_number):
        markdown += " --- |"
    markdown += "\n"

    while len(items) % column_number != 0:
        items.append("-")

    for index in range(0, len(items), column_number):
        markdown += f"| {' | '.join(items[index:index+column_number])} |\n"
    return markdown


if __name__ == "__main__":
    print("This file is not mean to be run as a script.")
