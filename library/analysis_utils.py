#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

    
class AnomalyV3Attrs:
    def __init__(self) -> None:
        pass

    BASE_FEATURES_TO_USE: list[str] = [
        'actor', 
        'day', 
        'is_weekend', 
        'codespaces_policy_group_deleted', 
        'codespaces_policy_group_updated', 
        'environment_remove_protection_rule', 
        'environment_update_protection_rule', 
        'git_clone', 
        'hook_create', 
        'integration_installation_create', 
        'ip_allow_list_disable', 
        'ip_allow_list_disable_for_installed_apps', 
        'ip_allow_list_entry_create', 
        'oauth_application_create', 
        'org_add_outside_collaborator', 
        'org_recovery_codes_downloaded', 
        'org_recovery_code_used', 
        'org_recovery_codes_printed', 
        'org_recovery_codes_viewed', 
        'personal_access_token_request_created', 
        'personal_access_token_access_granted', 
        'protected_branch_destroy', 
        'protected_branch_policy_override', 
        'public_key_create', 
        'pull_request_create', 
        'pull_request_merge', 
        'repo_access', 
        'repo_download_zip', 
        'repository_branch_protection_evaluation_disable', 
        'repository_ruleset_destroy', 
        'repository_ruleset_update', 
        'repository_secret_scanning_protection_disable', 
        'secret_scanning_push_protection_bypass', 
        'ssh_certificate_authority_create', 
        'ssh_certificate_requirement_disable', 
        'workflow_run_create', 
        'unique_ips_used', 
        'unique_repos_accessed',
        #'active_write_repos_written', 
        #'active_read_repos_written', 
        'outlier_time_count']
    
    FEATURES_TO_MEAN_EXCLUDE = ["is_weekend"]
    FEATURES_TO_ZSCORE_EXCLUDE = ["is_weekend"]

    BOTS_TO_REMOVE: list[str] = [
                '\[bot\]',
                'deploy_key',
                #'iiac-at-shell-reader',
                #'ITSO-siti-cpe-team-frontera-github',
                #'iiac-at-shell'
            ]


def floyd_marshall_process_chunk(k, dist, n):
    for i in range(n):
        for j in range(n):
            if dist[i, j] > dist[i, k] + dist[k, j]:
                dist[i, j] = dist[i, k] + dist[k, j]

        print(f"Processed chunk {k}.")
    return dist


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


def link_repos(row_number: int, repo_row_name: str, repo_list: list, df: pd.DataFrame) -> list[tuple[str, int]]:
    """Method for performing link checking for a given repository row.

    Args:
        row_number (int): The row number of the repository.
        repo_row_name (str): The name of the repository to check.
        repo_list (list): The list of repositories to check against.
        df (pd.DataFrame): The data frame that contains the user activity data.

    Returns:
        list[tuple[str, int]]: The list of repositories that were linked to the row repository.
    """

    print(f"Evaluating #{row_number}: {repo_row_name}...")

    # create a stack with the list of repos
    stack = list(repo_list)

    # list for storing links
    links = []

    # foreach column in the row
    while len(stack) > 0:

        repo_col_name = stack.pop()

        # if the column matches pow, pass
        if repo_col_name == repo_row_name:
            continue

        df_filtered = df[((df['repository'] == repo_row_name) | (df['repository'] == repo_col_name))]

        repo_int = False
        for actor in df_filtered['actor'].unique():

            # filter df to where actor matches 'actor', repository matches row or column name
            df_filtered_actor = df_filtered[df_filtered['actor'] == actor]

            # if df_filtered contains less than 2 rows, continue
            if len(df_filtered_actor) < 2:
                continue
            else:
                if not repo_int:
                    links.append((repo_col_name, 0))
                links[-1] = (repo_col_name, links[-1][1] + 1)
                repo_int = True

    print(f"{repo_row_name}: {len(links)} repos linked via {sum([l[1] for l in links])} users.")

    return (repo_row_name, links)


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
