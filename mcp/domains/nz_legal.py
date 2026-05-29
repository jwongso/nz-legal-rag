"""NZ Legal RAG domain configuration."""

import config
from .base import DomainConfig

NZ_LEGAL = DomainConfig(
    name="nz-legal-rag",
    description=(
        "Search NZ Tenancy Tribunal decisions, employment cases, legislation, "
        "and verify claims against live official sources."
    ),
    qdrant_collection=config.QDRANT_COLLECTION,
    source_labels=config.COURTS,
    enabled_tools=[
        "search_cases",
        "get_document",
        "search_legislation",
        "list_sources",
        "verify_live",
    ],
    tool_kwargs={
        # search_legislation: known Act prefixes for the act= parameter
        "act_prefixes": {
            "RTA":      "Residential Tenancies Act 1986",
            "ERA2000":  "Employment Relations Act 2000",
            "PA2020":   "Privacy Act 2020",
            "CCLA2017": "Contract and Commercial Law Act 2017",
            "CA1993":   "Companies Act 1993",
            "CRA1961":  "Crimes Act 1961",
        },
        # verify_live: trusted sources to browse for this domain
        "verify_sources": [
            {
                "id": "nz_legislation",
                "label": "NZ Legislation (legislation.govt.nz)",
                "base_url": "https://www.legislation.govt.nz",
            },
            {
                "id": "nz_tenancy_services",
                "label": "Tenancy Services (tenancy.govt.nz)",
                "base_url": "https://www.tenancy.govt.nz",
            },
        ],
    },
)
