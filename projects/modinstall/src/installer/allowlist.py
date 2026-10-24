from __future__ import annotations

from urllib.parse import urlparse


def check_url_allowed(url: str, allowed_domains: list[str]) -> bool:
    """Check if the given URL's hostname strictly matches or is a subdomain

    of any of the allowed domains.

    Parameters:
        url (str): The URL to check.
        allowed_domains (list[str]): A list of allowed domain strings.

    Returns:
        bool: True if the URL's hostname is allowed, False otherwise.

    Raises:
        ValueError: If the URL is malformed or missing a hostname.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValueError(f"Malformed URL: {url}") from e

    hostname = parsed.hostname
    if not hostname:
        if parsed.scheme and not parsed.netloc:
            raise ValueError(f"Malformed URL: {url}")
        if not parsed.scheme and not parsed.path:
            raise ValueError(f"Malformed URL: {url}")
        
        if parsed.netloc == "":
            raise ValueError(f"Malformed URL: {url}")
            
        try:
            hostname = parsed.netloc.split(":")[0].lower()
        except Exception as e:
            raise ValueError(f"Malformed URL: {url}") from e

        if not hostname:
            raise ValueError(f"Malformed URL: {url}")
    else:
        hostname = hostname.lower()

    if not allowed_domains:
        return False

    for domain in allowed_domains:
        if not domain:
            continue
        norm_domain = domain.strip().lower()
        if not norm_domain:
            continue

        if hostname == norm_domain:
            return True

        if hostname.endswith("." + norm_domain):
            return True

    return False
