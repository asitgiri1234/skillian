"""Resume upload and skill editing.

Not in the day-3 file list, but required by it: the brief says
``build_resume_embedding_text`` must be used "on both POST /resumes and
PATCH /resumes/{id}/skills", and day 2 stopped short of persisting a parsed
resume at all. These are those two endpoints. See DECISIONS 18.6.

Both share one invariant, and it is the reason they are in the same file:
**whenever a resume's skills or parse change, its embedding is rebuilt and its
matches are dropped.** A match scored against the old skill set is not stale, it
is wrong, and nothing in the ``matches`` table records which skill set produced
it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.matching.pipeline import clear_matches
from app.matching.skills import SkillCanonicalizer
from app.models import Resume, ResumeSkill, Skill, User
from app.providers import (
    EmbeddingError,
    EmbeddingProvider,
    LLMError,
    get_embedding_provider,
)
from app.structure import (
    StructureError,
    build_resume_embedding_text,
    extract_resume,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["resumes"])

MAX_MANUAL_SKILLS = 100


# --- request / response models ----------------------------------------------


class ResumeCreate(BaseModel):
    email: str = Field(description="Owner's email; the user is created if new")
    raw_text: str = Field(min_length=1, description="Plain text of the resume")
    name: str | None = None
    label: str | None = Field(default=None, description='e.g. "Backend CV, 2026"')
    file_path: str | None = None


class SkillsUpdate(BaseModel):
    skills: list[str] = Field(
        max_length=MAX_MANUAL_SKILLS,
        description="The resume's full skill list; replaces what is stored",
    )


class ResumeOut(BaseModel):
    id: UUID
    user_id: UUID
    label: str | None
    parsed: dict[str, Any] | None
    skills: list[str]
    #: The vector itself is never returned — 768 floats no client can use.
    has_embedding: bool
    #: What the embedding was actually built from. Returned because "why did this
    #: resume match that job" is unanswerable without it.
    embedding_text: str
    created_at: datetime


# --- helpers ----------------------------------------------------------------


def _resume_skill_names(session: Session, resume_id: UUID) -> list[str]:
    return list(
        session.execute(
            select(Skill.name)
            .join(ResumeSkill, ResumeSkill.skill_id == Skill.id)
            .where(ResumeSkill.resume_id == resume_id)
            .order_by(Skill.name)
        ).scalars()
    )


def _store_skills(
    session: Session, resume_id: UUID, names: list[str], source: str
) -> None:
    """Replace a resume's skill links with ``names``, canonicalised.

    Replace, not merge: this is the full list as the user or the parser sees it,
    and merging would make a removed skill impossible to remove.
    """
    session.execute(delete(ResumeSkill).where(ResumeSkill.resume_id == resume_id))

    canonicalizer = SkillCanonicalizer(session)
    skill_ids = canonicalizer.canonicalize_all(names)
    if skill_ids:
        session.execute(
            pg_insert(ResumeSkill)
            .values(
                [
                    {"resume_id": resume_id, "skill_id": skill_id, "source": source}
                    for skill_id in skill_ids
                ]
            )
            .on_conflict_do_nothing(index_elements=["resume_id", "skill_id"])
        )


def _embed_resume(
    parsed: dict[str, Any], embedder: EmbeddingProvider
) -> tuple[list[float] | None, str]:
    """Build the embedding text and embed it. Returns ``(vector, text)``.

    ``(None, "")`` when there is nothing to embed — the provider rejects empty
    text by design, so the caller must not be handed a vector it cannot trust.
    """
    text = build_resume_embedding_text(parsed)
    if not text:
        logger.warning("Resume has no skills or experience to embed")
        return None, ""
    return embedder.embed(text), text


# --- endpoints --------------------------------------------------------------


@router.post(
    "/resumes", response_model=ResumeOut, status_code=status.HTTP_201_CREATED
)
def create_resume(
    body: ResumeCreate, session: Session = Depends(get_session)
) -> ResumeOut:
    """Store a resume: extract it, link its skills, embed it.

    **This request is slow — expect 60-120 seconds.** Extraction is a local 7b
    model reading a full resume, and unlike a search there is nothing useful to
    return before it finishes: the resume's id is worthless to a caller who
    cannot search on it yet, and POST /searches rejects an unparsed resume.
    Uploading is also a once-per-resume action, where searching is the repeated
    one. If this becomes a problem the fix is the pattern already built for
    searches — a run row plus BackgroundTasks. See DECISIONS 18.7.
    """
    email = body.email.strip().casefold()
    user = session.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is None:
        user = User(email=email, name=body.name)
        session.add(user)
        session.flush()

    try:
        parsed = extract_resume(body.raw_text)
    except StructureError as exc:
        # 422: the document was received and understood as a request, but no
        # amount of retrying at our end produced a usable parse from it.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Could not extract a usable resume after {exc.attempts} attempts. "
            f"Last validation errors: {exc.last_errors}",
        ) from exc
    except LLMError as exc:
        # 503: an operator problem (daemon down, model not pulled), not the
        # caller's, and it will succeed on retry once someone fixes it.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    parsed_json = parsed.model_dump()

    try:
        vector, embedding_text = _embed_resume(parsed_json, get_embedding_provider())
    except EmbeddingError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    resume = Resume(
        user_id=user.id,
        label=body.label,
        file_path=body.file_path,
        raw_text=body.raw_text,
        parsed=parsed_json,
        embedding=vector,
    )
    session.add(resume)
    session.flush()

    _store_skills(session, resume.id, parsed.skills, source="llm")
    session.commit()

    return ResumeOut(
        id=resume.id,
        user_id=resume.user_id,
        label=resume.label,
        parsed=resume.parsed,
        skills=_resume_skill_names(session, resume.id),
        has_embedding=vector is not None,
        embedding_text=embedding_text,
        created_at=resume.created_at,
    )


@router.patch("/resumes/{resume_id}/skills", response_model=ResumeOut)
def update_resume_skills(
    resume_id: UUID, body: SkillsUpdate, session: Session = Depends(get_session)
) -> ResumeOut:
    """Replace a resume's skills, then re-embed and invalidate its matches.

    Editing skills is the main correction a user makes after an imperfect parse,
    so it has to do all three things or the correction is cosmetic: the stored
    ``parsed["skills"]`` feeds the embedding text, the embedding feeds the
    semantic score, and the existing match rows were computed from the list that
    just changed.
    """
    resume = session.get(Resume, resume_id)
    if resume is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No resume with id {resume_id}"
        )

    cleaned = [s.strip() for s in body.skills if s and s.strip()]

    # Copy-then-reassign: SQLAlchemy tracks JSONB by identity, so mutating
    # resume.parsed in place would leave the change uncommitted.
    parsed = dict(resume.parsed or {})
    parsed["skills"] = cleaned

    try:
        vector, embedding_text = _embed_resume(parsed, get_embedding_provider())
    except EmbeddingError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    resume.parsed = parsed
    resume.embedding = vector

    _store_skills(session, resume_id, cleaned, source="manual")
    # Every stored score for this resume was computed against the old skill set.
    clear_matches(session, resume_id)
    session.commit()

    return ResumeOut(
        id=resume.id,
        user_id=resume.user_id,
        label=resume.label,
        parsed=resume.parsed,
        skills=_resume_skill_names(session, resume.id),
        has_embedding=vector is not None,
        embedding_text=embedding_text,
        created_at=resume.created_at,
    )
