from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pasqal_cloud import AUTH0_CONFIG, PASQAL_ENDPOINTS, SDK, Auth0Conf, Endpoints, TokenProvider

DEFAULT_CLOUD_ENV = "prod"


@lru_cache(maxsize=5)
def _get_client(
    username: Optional[str] = None,
    password: Optional[str] = None,
    project_id: Optional[str] = None,
    environment: Optional[str] = None,
    token_provider: Optional[TokenProvider] = None,
) -> SDK:
    auth0conf: Optional[Auth0Conf] = None
    endpoints: Optional[Endpoints] = None

    username = os.environ.get("PASQAL_CLOUD_USERNAME", "") if username is None else username
    password = os.environ.get("PASQAL_CLOUD_PASSWORD", "") if password is None else password
    project_id = os.environ.get("PASQAL_CLOUD_PROJECT_ID", "") if project_id is None else project_id

    environment = (
        os.environ.get("PASQAL_CLOUD_ENV", DEFAULT_CLOUD_ENV)
        if environment is None
        else environment
    )

    # setup configuration for environments different than production
    if environment == "preprod":
        auth0conf = AUTH0_CONFIG["preprod"]
        endpoints = PASQAL_ENDPOINTS["preprod"]
    elif environment == "dev":
        auth0conf = AUTH0_CONFIG["dev"]
        endpoints = PASQAL_ENDPOINTS["dev"]

    if all([username, password, project_id]) or all([token_provider, project_id]):
        pass
    else:
        raise Exception("You must either provide project_id and log-in details or a token provider")

    return SDK(
        username=username,
        password=password,
        project_id=project_id,
        auth0=auth0conf,
        endpoints=endpoints,
        token_provider=token_provider,
    )


def get_client(credentials: dict) -> SDK:
    return _get_client(
        username=credentials.get("username", None),
        password=credentials.get("password", None),
        project_id=credentials.get("project_id", None),
        environment=credentials.get("environment", DEFAULT_CLOUD_ENV),
        token_provider=credentials.get("token_provider", None),
    )
