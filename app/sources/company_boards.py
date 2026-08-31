"""Company board slugs for the Greenhouse and Lever sources.

Both APIs are **per company**, not search endpoints, so there is no query to
send — there is only a list of boards to poll. That list is data, so it lives
here rather than being buried in either source.

**Every slug below was verified to return at least one posting on 2026-08-31.**
Slugs drift: a company migrates ATS, renames its board, or moves to a private
one, and the endpoint then 404s. The sources log a 404 and continue rather than
failing the run, and :mod:`scripts.verify_boards` re-checks the whole list.

The Indian-company coverage here is thinner than intended, and the reason is
worth recording: most large Indian tech employers (Razorpay, Swiggy, Zomato,
Flipkart, PhonePe, Meesho, Zerodha, Freshworks, Zoho, Byju's, Unacademy,
Lenskart, Nykaa, Delhivery, Udaan) do **not** expose a public Greenhouse or
Lever board — they run their own careers site or a private ATS. All 45 were
probed and 404'd; see NOT_FOUND below. What survives is India-headquartered or
India-hiring companies that do use these boards, plus remote-friendly global
companies that hire into India.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

#: Verified 2026-08-31. Comment shows the posting count at verification time,
#: as a rough signal of board size — these move constantly.
GREENHOUSE_COMPANIES: tuple[str, ...] = (
    # --- India-headquartered or major India engineering presence -----------
    "groww",          # 5    fintech, Bengaluru
    "postman",        # 63   API tooling, Bengaluru
    "druva",          # 38   data protection, Pune
    "netradyne",      # 24   computer vision, Bengaluru
    "highradius",     # 82   fintech AI, Hyderabad
    "glance",         # 49   consumer internet, Bengaluru
    "zenoti",         # 48   SaaS, Hyderabad
    "hackerrank",     # 30   developer assessment, Bengaluru
    "rubrik",         # 129  data security, large Bengaluru office
    # --- remote-friendly global, hire into India ---------------------------
    "databricks",     # 855
    "stripe",         # 572
    "anthropic",      # 571
    "mongodb",        # 405
    "elastic",        # 342
    "cloudflare",     # 306
    "brex",           # 294
    "samsara",        # 248
    "gitlab",         # 221  fully remote
    "scaleai",        # 217
    "affirm",         # 210
    "coinbase",       # 189  remote-first
    "airbnb",         # 166
    "figma",          # 163
    "reddit",         # 153
    "twilio",         # 142
    "robinhood",      # 128
    "instacart",      # 123
    "asana",          # 122
    "vercel",         # 91
    "discord",        # 51
    "dropbox",        # 41   virtual-first
    "turing",         # 25   remote engineering marketplace
    "wise",           # 21
    "airtable",       # 16
    "springboard",    # 10
)

#: Verified 2026-08-31.
LEVER_COMPANIES: tuple[str, ...] = (
    "binance",        # 272  remote-first
    "gohighlevel",    # 86
    "zeta",           # 20   fintech, Bengaluru
    "fampay",         # 15   fintech, Bengaluru
    "cred",           # 14   fintech, Bengaluru
    "fi",             # 8    neobank, Bengaluru
)

#: Probed and 404 on 2026-08-31. Kept so the next person does not spend an
#: afternoon rediscovering that these boards are not public.
NOT_FOUND: dict[str, tuple[str, ...]] = {
    "greenhouse": (
        "razorpay", "zerodha", "cred", "meesho", "swiggy", "zomato", "flipkart",
        "phonepe", "dream11", "sharechat", "unacademy", "byjus", "browserstack",
        "freshworks", "zoho", "hasura", "chargebee", "innovaccer", "mindtickle",
        "gupshup", "whatfix", "clevertap", "haptik", "yellowai", "darwinbox",
        "icertis", "udaan", "urbancompany", "lenskart", "nykaa", "delhivery",
        "rapido", "notion", "doordash", "plaid", "openai", "canva", "atlassian",
        "hashicorp", "confluent", "grafana", "supabase", "render", "fly",
        "temporal", "sprinklr", "cohesity", "nutanix", "arista", "couchbase",
        "juspay", "setu", "zeta", "tekion", "amagi",
    ),
    "lever": (
        "razorpay", "zepto", "navi", "jupiter", "slice", "khatabook", "bharatpe",
        "spinny", "cars24", "leadsquared", "exotel", "netcore", "matchmove",
        "jar", "zolve", "stashfin", "upstox", "angelone", "opensea", "ramp",
        "attentive", "voiceflow", "huggingface", "perplexityai", "cohere",
        "runwayml", "together", "sardine", "deel", "remote", "match", "revolut",
        "monzo", "checkout", "gocardless", "juspay", "practo", "curefit",
    ),
}


def interleave(
    by_company: "list[tuple[str, list]]", limit: int | None
) -> list:
    """Flatten per-company results round-robin, capped at ``limit``.

    Round-robin, not concatenation: Databricks alone returns 855 postings and
    would fill any cap before a single Indian board was reached. Taking one from
    each board in turn keeps the sample spread across employers, which is what
    a candidate actually wants from a search.
    """
    flat: list = []
    if not by_company:
        return flat
    index = 0
    while True:
        added = False
        for _, jobs in by_company:
            if index < len(jobs):
                flat.append(jobs[index])
                added = True
                if limit is not None and len(flat) >= limit:
                    return flat
        if not added:
            return flat
        index += 1


def fetch_boards_concurrently(
    companies: "tuple[str, ...]",
    fetch_one: "Callable[[str], list]",
    max_concurrency: int = 8,
) -> "list[tuple[str, list]]":
    """Fetch every board concurrently, returning ``(company, jobs)`` in order.

    Forty sequential HTTPS round trips at ~1s each is forty seconds of waiting
    on work that is entirely I/O-bound.

    ``asyncio.gather`` over ``asyncio.to_thread``, rather than an async HTTP
    client: ``JobSource.fetch`` is a synchronous interface and must not change
    (``base.py`` is deliberately untouched), so the blocking per-board call is
    pushed to a worker thread and the gather waits on all of them. A semaphore
    caps in-flight requests so a 40-board list does not open 40 sockets at once
    against two hosts.

    Exceptions are captured, not raised: ``fetch_one`` already swallows HTTP
    errors per board, and ``return_exceptions=True`` covers anything it misses,
    because one dead board must not lose the other thirty-nine.
    """
    import asyncio

    if not companies:
        return []

    async def run() -> "list[tuple[str, list]]":
        semaphore = asyncio.Semaphore(max_concurrency)

        async def one(company: str) -> "tuple[str, list]":
            async with semaphore:
                return company, await asyncio.to_thread(fetch_one, company)

        results = await asyncio.gather(
            *(one(company) for company in companies), return_exceptions=True
        )
        out: list[tuple[str, list]] = []
        for company, result in zip(companies, results):
            if isinstance(result, BaseException):
                logger.warning("board %s failed: %r", company, result)
                out.append((company, []))
            else:
                out.append(result)
        return out

    return asyncio.run(run())
