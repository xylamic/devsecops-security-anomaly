#!/usr/bin/env python
# coding: utf-8

    
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
