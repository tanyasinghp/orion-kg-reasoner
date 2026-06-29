from typing import Annotated
from dataclasses import dataclass
from datetime import datetime

import httpx
import jwt
from jwt.exceptions import InvalidTokenError
from fastapi import Depends, HTTPException, status, Header
from fastapi.security.utils import get_authorization_scheme_param
from pydantic import BaseModel, Field

from config import settings


class OpenIdConfiguration(BaseModel):
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    end_session_endpoint: str
    jwks_uri: str
    introspection_endpoint: str
    revocation_endpoint: str
    response_types_supported: list[str]
    grant_types_supported: list[str]
    subject_types_supported: list[str]
    id_token_signing_alg_values_supported: list[str]
    scopes_supported: list[str]
    token_endpoint_auth_methods_supported: list[str]
    claims_supported: list[str]
    code_challenge_methods_supported: list[str]
    tls_client_certificate_bound_access_tokens: bool
    frontchannel_logout_supported: bool
    frontchannel_logout_session_supported: bool
    backchannel_logout_supported: bool
    backchannel_logout_session_supported: bool
    authorization_response_iss_parameter_supported: bool


@dataclass
class OpenIdProvider:
    oidc_config: OpenIdConfiguration
    jwks_client: jwt.PyJWKClient

    @staticmethod
    def discover(server_url: str) -> "OpenIdProvider":
        with httpx.Client() as client:
            response = client.get(f"{server_url}/.well-known/openid-configuration")
            oidc_config = OpenIdConfiguration(**response.json())
            jwks_client = jwt.PyJWKClient(oidc_config.jwks_uri, cache_keys=True)
            return OpenIdProvider(oidc_config, jwks_client)


_oidc_enabled: bool = bool(settings.api.oidc.server_url and settings.api.oidc.issuer)
_oidc_provider: OpenIdProvider | None = None
if _oidc_enabled:
    _oidc_provider = OpenIdProvider.discover(settings.api.oidc.server_url)


class UserToken(BaseModel):
    subject: str = Field(alias="sub")
    audience: str = Field(alias="aud")
    issuer: str = Field(alias="iss")
    issued_at: datetime = Field(alias="iat")
    expires_at: datetime = Field(alias="exp")


async def validate_token(token: str) -> UserToken:
    if not _oidc_enabled or _oidc_provider is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OIDC authentication is not configured",
        )
    signing_key = _oidc_provider.jwks_client.get_signing_key_from_jwt(token)
    try:
        supported_signing_algorithms = _oidc_provider.oidc_config.id_token_signing_alg_values_supported
        payload = jwt.decode(
            token,
            key=signing_key,
            algorithms=supported_signing_algorithms,
            audience=settings.api.oidc.client_id,
            issuer=settings.api.oidc.issuer,
            options={"strict_aud": True}
        )
        return UserToken(**payload)
    except InvalidTokenError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from err


async def get_current_user(authorization: str = Header(None)) -> UserToken:
    if not _oidc_enabled:
        return UserToken(sub="local", aud="", iss="", iat=datetime.now(), exp=datetime.now())
    scheme, token = get_authorization_scheme_param(authorization)
    if scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await validate_token(token)


UserDependency = Annotated[UserToken, Depends(get_current_user)]
