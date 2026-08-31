"""A starter skill vocabulary with aliases, for dictionary extraction.

Dictionary extraction can only find what it knows, and the `skills` table it
reads from was populated entirely as a side effect of LLM extraction: 225 rows,
**zero aliases**, and a long tail of pre-fix junk. Without aliases the matcher
cannot connect "k8s" to Kubernetes or "JS" to JavaScript, which is most of the
value of having an alias column at all.

This is that vocabulary. It is deliberately a *seed*, not a taxonomy: a few
hundred well-known technologies with their common surface forms, biased toward
what appears in the backend/data/platform postings this project actually
ingests. It is not exhaustive and is not meant to be — the table keeps growing
from resume parses.

Seeding is idempotent: existing rows keep their id (so `job_skills` and
`resume_skills` references survive) and gain the aliases they were missing.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models import Skill
from app.normalize import normalize_text

logger = logging.getLogger(__name__)

#: canonical name -> alias surface forms
SEED_SKILLS: dict[str, tuple[str, ...]] = {
    # --- languages ---------------------------------------------------------
    "Python": ("py", "python3"),
    "JavaScript": ("js", "ecmascript", "es6"),
    "TypeScript": ("ts",),
    "Java": (),
    "Go": ("golang",),
    "Rust": (),
    "C": (),
    "C++": ("cpp", "cplusplus"),
    "C#": ("csharp", "c sharp"),
    "Ruby": (),
    "PHP": (),
    "Scala": (),
    "Kotlin": (),
    "Swift": (),
    "R": (),
    "MATLAB": (),
    "Perl": (),
    "Bash": ("shell scripting", "bash scripting"),
    "PowerShell": (),
    "SQL": (),
    "HTML": ("html5",),
    "CSS": ("css3",),
    "Elixir": (),
    "Haskell": (),
    "Dart": (),
    "Groovy": (),
    "Objective-C": ("objc",),
    # --- backend frameworks ------------------------------------------------
    "FastAPI": (),
    "Django": (),
    "Flask": (),
    "Spring Boot": ("springboot",),
    "Express": ("expressjs", "express.js"),
    "NestJS": ("nest.js",),
    "Rails": ("ruby on rails", "ror"),
    "Laravel": (),
    ".NET": ("dotnet", "asp.net", ".net core"),
    "gRPC": (),
    "GraphQL": (),
    "REST APIs": ("restful", "rest api", "restful api"),
    "Celery": (),
    "RabbitMQ": (),
    # --- frontend ----------------------------------------------------------
    "React": ("reactjs", "react.js"),
    "Angular": ("angularjs",),
    "Vue": ("vuejs", "vue.js"),
    "Next.js": ("nextjs",),
    "Svelte": (),
    "Redux": (),
    "Tailwind CSS": ("tailwind",),
    "Webpack": (),
    "Node.js": ("nodejs",),
    "jQuery": (),
    # --- data stores -------------------------------------------------------
    "PostgreSQL": ("postgres", "psql", "postgresql"),
    "MySQL": (),
    "SQLite": (),
    "MongoDB": ("mongo",),
    "Redis": (),
    "Elasticsearch": ("elastic search", "elk"),
    "Cassandra": (),
    "DynamoDB": (),
    "ClickHouse": (),
    "Snowflake": (),
    "BigQuery": (),
    "Redshift": (),
    "Oracle": (),
    "SQL Server": ("mssql", "microsoft sql server"),
    "Neo4j": (),
    "pgvector": (),
    # --- data / ML ---------------------------------------------------------
    "Kafka": ("apache kafka",),
    "Spark": ("apache spark", "pyspark"),
    "Airflow": ("apache airflow",),
    "dbt": (),
    "Dagster": (),
    "Hadoop": (),
    "Flink": ("apache flink",),
    "pandas": (),
    "NumPy": ("numpy",),
    "scikit-learn": ("sklearn", "scikit learn"),
    "PyTorch": (),
    "TensorFlow": ("tensor flow",),
    "Keras": (),
    "Machine Learning": ("ml",),
    "Deep Learning": (),
    "NLP": ("natural language processing",),
    "Computer Vision": (),
    "LLM": ("llms", "large language models"),
    "RAG": ("retrieval augmented generation",),
    "Tableau": (),
    "Power BI": ("powerbi",),
    "Looker": (),
    # --- infra / cloud -----------------------------------------------------
    "Docker": (),
    "Kubernetes": ("k8s", "kubernetes"),
    "Terraform": (),
    "Ansible": (),
    "AWS": ("amazon web services",),
    "Azure": ("microsoft azure",),
    "GCP": ("google cloud", "google cloud platform"),
    "Linux": (),
    "Nginx": (),
    "Jenkins": (),
    "GitHub Actions": (),
    "GitLab CI": ("gitlab-ci",),
    "CI/CD": ("cicd", "ci cd"),
    "Prometheus": (),
    "Grafana": (),
    "Datadog": (),
    "OpenTelemetry": ("otel",),
    "Helm": (),
    "Serverless": ("aws lambda",),
    "Microservices": ("micro services",),
    "Git": (),
    "Vault": ("hashicorp vault",),
    "Kibana": (),
    "Splunk": (),
    "Istio": (),
    "Envoy": (),
    # --- practices ---------------------------------------------------------
    "Agile": (),
    "Scrum": (),
    "TDD": ("test driven development",),
    "Unit Testing": ("unit tests",),
    "pytest": (),
    "Jest": (),
    "Selenium": (),
    "Playwright": (),
    "Cypress": (),
    "JUnit": (),
    "Postman": (),
    "System Design": ("systems design",),
    "Distributed Systems": (),
    "Data Modelling": ("data modeling",),
    "Performance Tuning": ("performance optimization", "query tuning"),
    "Code Review": ("code reviews",),
    "Technical Writing": (),
    "Incident Response": (),
    "Observability": (),
    "Security": ("appsec", "application security"),
    "OAuth": ("oauth2",),
    "JWT": (),
    "Web Scraping": (),
    "ETL": ("elt",),
    "Data Pipelines": ("data pipeline",),
    "API Design": (),
    "Sales": (),
    "Salesforce": (),
    "SEO": (),
    "Figma": (),
    "Jira": (),
    "Excel": ("microsoft excel", "ms excel"),
}



#: Aliases deliberately NOT used, and why. Audited 2026-08-31 after
#: `monitoring` -> Observability was found matching accountancy roles.
#:
#: The rule: an alias must be an abbreviation ("k8s") or a spelling variant
#: ("postgres", "nodejs") of the canonical term. It must not be an ordinary
#: English word that merely co-occurs with it.
#:
#:   monitoring -> Observability   ordinary word; matched "Accountant II"
#:   shell, sh  -> Bash            ordinary word; "shell company", "she"
#:   spring     -> Spring Boot     the season
#:   rest       -> REST APIs       "the rest of the team"
#:   node       -> Node.js         "node in the graph"
#:   torch      -> PyTorch         ordinary word
#:   cv         -> Computer Vision curriculum vitae — in *job postings*
#:   lambda     -> Serverless      Python keyword, mathematical term
#:   on-call    -> Incident Response  prose in most engineering postings
#:   scraping   -> Web Scraping     ordinary word
#:   unix       -> Linux            a different OS, not an alias
REJECTED_ALIASES: dict[str, str] = {
    "monitoring": "Observability",
    "shell": "Bash",
    "sh": "Bash",
    "spring": "Spring Boot",
    "rest": "REST APIs",
    "node": "Node.js",
    "torch": "PyTorch",
    "cv": "Computer Vision",
    "lambda": "Serverless",
    "on-call": "Incident Response",
    "on call": "Incident Response",
    "scraping": "Web Scraping",
    "unix": "Linux",
}


def seed_skills(session: Session) -> tuple[int, int]:
    """Insert missing skills and backfill aliases. Returns ``(added, updated)``.

    Idempotent, and non-destructive: an existing row keeps its id, so every
    `job_skills` and `resume_skills` foreign key referencing it stays valid.
    Aliases are merged rather than replaced, so anything added by hand survives.
    """
    existing = {
        normalize_text(name): (skill_id, list(aliases or []))
        for skill_id, name, aliases in session.execute(
            select(Skill.id, Skill.name, Skill.aliases)
        )
    }

    added = 0
    updated = 0

    for name, aliases in SEED_SKILLS.items():
        key = normalize_text(name)
        row = existing.get(key)
        if row is None:
            session.execute(
                pg_insert(Skill)
                .values(name=name, aliases=list(aliases))
                .on_conflict_do_nothing(index_elements=[Skill.name])
            )
            added += 1
            continue

        skill_id, current = row
        merged = list(current)
        lowered = {a.casefold() for a in merged}
        for alias in aliases:
            if alias.casefold() not in lowered:
                merged.append(alias)
                lowered.add(alias.casefold())
        if merged != current:
            session.execute(
                Skill.__table__.update()
                .where(Skill.id == skill_id)
                .values(aliases=merged)
            )
            updated += 1

    session.commit()
    logger.info("Seeded skills: %s added, %s alias-updated", added, updated)
    return added, updated
