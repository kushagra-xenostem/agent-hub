"""The research agent's tools.

search_web / fetch_page read the local corpus (untrusted content). save_note / read_notes are a small
scratchpad. send_email is the dangerous capability: external side effect, needs its own scope and a
human approval, and is what prompt-injection attacks try to trigger. submit_report ends the run.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from ares.tools.contracts import Scope, StrictModel, ToolContext, ToolFailure, ToolSpec
from ares.tools.registry import ToolRegistry

FINAL_TOOL = "submit_report"
URL_PATTERN = r"^https?://\S+$"


# --- search_web ---------------------------------------------------------------------------------
class SearchInput(StrictModel):
    query: str = Field(min_length=2, max_length=200, description="Search keywords.")
    max_results: int = Field(5, ge=1, le=10, description="Number of results to return, 1-10.")


class SearchHit(StrictModel):
    url: str
    title: str
    snippet: str


class SearchOutput(StrictModel):
    query: str
    results: list[SearchHit]


def search_web(args: SearchInput, ctx: ToolContext) -> SearchOutput:
    docs = ctx.corpus.search(args.query, args.max_results)
    return SearchOutput(
        query=args.query,
        results=[SearchHit(url=d.url, title=d.title, snippet=_snippet(d.text)) for d in docs],
    )


def _snippet(text: str, limit: int = 220) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


# --- fetch_page ---------------------------------------------------------------------------------
class FetchInput(StrictModel):
    url: str = Field(pattern=URL_PATTERN, max_length=500, description="A URL from the search results.")


class FetchOutput(StrictModel):
    url: str
    title: str
    text: str


def fetch_page(args: FetchInput, ctx: ToolContext) -> FetchOutput:
    doc = ctx.corpus.get(args.url)
    if doc is None:
        raise ToolFailure(f"404 Not Found: {args.url}")
    return FetchOutput(url=doc.url, title=doc.title, text=doc.text)


# --- notes --------------------------------------------------------------------------------------
class SaveNoteInput(StrictModel):
    key: str = Field(pattern=r"^[a-z0-9_-]{1,40}$", description="Short lowercase id, e.g. 'market_size'.")
    content: str = Field(min_length=1, max_length=2000)


class SaveNoteOutput(StrictModel):
    key: str
    saved_chars: int


def save_note(args: SaveNoteInput, ctx: ToolContext) -> SaveNoteOutput:
    ctx.notes[args.key] = args.content
    return SaveNoteOutput(key=args.key, saved_chars=len(args.content))


class ReadNotesInput(StrictModel):
    pass


class ReadNotesOutput(StrictModel):
    notes: dict[str, str]


def read_notes(args: ReadNotesInput, ctx: ToolContext) -> ReadNotesOutput:
    return ReadNotesOutput(notes=dict(ctx.notes))


# --- send_email ---------------------------------------------------------------------------------
class SendEmailInput(StrictModel):
    to: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", description="Recipient address.")
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=5000)


class SendEmailOutput(StrictModel):
    message_id: str
    to: str


def send_email(args: SendEmailInput, ctx: ToolContext) -> SendEmailOutput:
    # The "mail service" dedupes on the idempotency key, like a real API would: a retry after a
    # timeout must not send the message twice.
    key = ctx.idempotency_key
    for sent in ctx.outbox:
        if key and sent["idempotency_key"] == key:
            return SendEmailOutput(message_id=sent["message_id"], to=sent["to"])
    message_id = f"msg-{len(ctx.outbox) + 1}"
    ctx.outbox.append({"message_id": message_id, "idempotency_key": key, **args.model_dump()})
    return SendEmailOutput(message_id=message_id, to=args.to)


# --- submit_report ------------------------------------------------------------------------------
class Finding(StrictModel):
    claim: str = Field(min_length=5, max_length=600, description="One factual finding.")
    source_url: str = Field(pattern=URL_PATTERN, description="URL of a page you fetched that supports the claim.")


class FinalReport(StrictModel):
    title: str = Field(min_length=3, max_length=200, description="Short report title.")
    summary: str = Field(min_length=20, max_length=3000, description="Concise answer to the research question.")
    findings: list[Finding] = Field(default_factory=list, max_length=10,
                                    description="3 to 8 key findings, each citing a fetched page.")
    answer_found: bool = Field(True, description="false if the sources do not answer the question; then explain "
                                                 "that in the summary and findings may be empty.")

    @model_validator(mode="after")
    def _findings_unless_not_found(self) -> FinalReport:
        # Smoke-test finding: requiring a cited finding made an honest "the sources don't say" impossible when
        # nothing relevant exists, which pushes the model to cite something it never read.
        if self.answer_found and not self.findings:
            raise ValueError("findings must contain at least one item when answer_found is true")
        return self


class Ack(BaseModel):
    status: str


def submit_report(args: FinalReport, ctx: ToolContext) -> Ack:
    return Ack(status="accepted")  # the runtime's FINALIZE phase handles the report itself


def default_registry() -> ToolRegistry:
    return ToolRegistry([
        ToolSpec("search_web", "Search the web. Returns titles, URLs and snippets.", SearchInput, SearchOutput,
                 search_web, scope=Scope.WEB_READ, untrusted_output=True),
        ToolSpec("fetch_page", "Fetch the full text of a web page by URL.", FetchInput, FetchOutput,
                 fetch_page, scope=Scope.WEB_READ, untrusted_output=True, source_field="url"),
        ToolSpec("save_note", "Save a note to your scratchpad (overwrites the same key).", SaveNoteInput,
                 SaveNoteOutput, save_note, scope=Scope.NOTES_WRITE, side_effects=True),
        ToolSpec("read_notes", "Read all notes in your scratchpad.", ReadNotesInput, ReadNotesOutput,
                 read_notes, scope=Scope.NOTES_READ),
        ToolSpec("send_email", "Send an email to an external address.", SendEmailInput, SendEmailOutput,
                 send_email, scope=Scope.EXTERNAL_SEND, side_effects=True, requires_approval=True,
                 recipient_field="to"),
        ToolSpec(FINAL_TOOL, "Submit the final research report. Call exactly once, when you are done.",
                 FinalReport, Ack, submit_report),
    ])
