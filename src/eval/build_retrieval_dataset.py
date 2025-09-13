import json
import pathlib
from typing import Dict, List, Iterable


META_PATH = pathlib.Path("rag_index/meta.json")
OUT_PATH = pathlib.Path("kb/retrieval_eval_dataset.json")


def load_meta() -> List[Dict]:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return meta


def index_meta_by_source_and_section(meta: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """Build nested index: source -> section -> list of chunk dicts"""
    by_src: Dict[str, Dict[str, List[Dict]]] = {}
    for item in meta:
        src = item.get("source")
        sec = item.get("section") or "Summary"
        by_src.setdefault(src, {}).setdefault(sec, []).append(item)
    return by_src


def chunks_for_sections(by_src: Dict[str, Dict[str, List[Dict]]], source: str, sections: Iterable[str]) -> List[str]:
    out: List[str] = []
    for sec in sections:
        for it in by_src.get(source, {}).get(sec, []):
            out.append(it.get("id"))
    return out


def source_path(src: str) -> str:
    """Return dataset path for a given source filename, normalized for doc-level eval.

    We keep full kb-relative paths for readability; retrieval_eval normalizes the
    basename so both `kb/<file>.md` and `kb/recommendable_resources/<file>.md`
    will match meta "source" values like `<file>.md`.
    """
    if src.startswith("r_"):
        return f"kb/recommendable_resources/{src}"
    return f"kb/{src}"


def build_items() -> List[Dict]:
    meta = load_meta()
    by_src = index_meta_by_source_and_section(meta)

    # Each question may reference multiple docs (KB and recommendable_resources)
    # with specific sections considered relevant.
    qdef = [
        {
            "id": "ret_001",
            "question": "What is Shakers and what does it offer?",
            "docs": [
                {"source": "overview.md", "sections": ["Summary", "Key Features"]},
            ],
        },
        {
            "id": "ret_002",
            "question": "Outline the payment flow for a milestone on Shakers.",
            "docs": [
                {"source": "payments.md", "sections": ["Payment Flow"]},
                {"source": "r_docs_payments_overview.md", "sections": ["Summary", "Key Points"]},
            ],
        },
        {
            "id": "ret_003",
            "question": "Are refunds possible before a milestone is approved?",
            "docs": [
                {"source": "payments.md", "sections": ["Notes"]},
                {"source": "r_docs_payments_overview.md", "sections": ["Key Points"]},
            ],
        },
        {
            "id": "ret_004",
            "question": "How long do freelancer payouts take after approval?",
            "docs": [
                {"source": "payments.md", "sections": ["Notes"]},
                {"source": "r_docs_payments_overview.md", "sections": ["Summary"]},
            ],
        },
        {
            "id": "ret_005",
            "question": "How are invoices generated on Shakers?",
            "docs": [
                {"source": "invoicing.md", "sections": ["Summary", "Invoices"]},
                {"source": "r_docs_invoices_faq.md", "sections": ["Summary", "Key Points"]},
            ],
        },
        {
            "id": "ret_006",
            "question": "Does Shakers calculate VAT or sales tax on invoices?",
            "docs": [
                {"source": "invoicing.md", "sections": ["Tax"]},
                {"source": "r_docs_invoices_vat_eu.md", "sections": ["Summary", "Key Points"]},
            ],
        },
        {
            "id": "ret_007",
            "question": "What security features are available?",
            "docs": [
                {"source": "security.md", "sections": ["Security Features"]},
            ],
        },
        {
            "id": "ret_008",
            "question": "What are the main roles and their permissions?",
            "docs": [
                {"source": "roles.md", "sections": ["Roles", "Permissions"]},
            ],
        },
        {
            "id": "ret_009",
            "question": "What are the steps to onboard a new company user?",
            "docs": [
                {"source": "onboarding.md", "sections": ["Steps"]},
            ],
        },
        {
            "id": "ret_010",
            "question": "Define a milestone and escrow in Shakers.",
            "docs": [
                {"source": "glossary.md", "sections": ["Summary"]},
                {"source": "r_guide_milestones_setup.md", "sections": ["Summary", "Key Points"]},
            ],
        },
        {
            "id": "ret_011",
            "question": "How does the dispute resolution process work?",
            "docs": [
                {"source": "disputes.md", "sections": ["Process"]},
                {"source": "r_policy_disputes.md", "sections": ["Summary", "Key Points"]},
            ],
        },
        {
            "id": "ret_012",
            "question": "What plans does Shakers offer and how is billing handled?",
            "docs": [
                {"source": "pricing.md", "sections": ["Plans", "Billing"]},
                {"source": "r_docs_budgets_ranges.md", "sections": ["Key Points"]},
            ],
        },
        {
            "id": "ret_013",
            "question": "What criteria does the freelancer matching consider?",
            "docs": [
                {"source": "matching.md", "sections": ["Matching Criteria"]},
            ],
        },
        {
            "id": "ret_014",
            "question": "How are projects structured and how is time tracked?",
            "docs": [
                {"source": "projects.md", "sections": ["Milestones", "Time Tracking"]},
                {"source": "r_guide_milestones_setup.md", "sections": ["Key Points"]},
            ],
        },
        {
            "id": "ret_015",
            "question": "How can I troubleshoot a 'Login failed' issue?",
            "docs": [
                {"source": "faq.md", "sections": ["Common Issues", "Troubleshooting Steps"]},
            ],
        },
        {
            "id": "ret_016",
            "question": "What are best practices around contracts and NDAs?",
            "docs": [
                {"source": "contracts.md", "sections": ["Features", "Best Practices"]},
                {"source": "r_docs_contracts_fixed_price.md", "sections": ["Key Points"]},
                {"source": "r_article_contracts_tnm_vs_fixed.md", "sections": ["Key Points"]},
            ],
        },
        {
            "id": "ret_017",
            "question": "What are the steps in the talent vetting process?",
            "docs": [
                {"source": "vetting.md", "sections": ["Steps", "Result"]},
            ],
        },
        {
            "id": "ret_018",
            "question": "What messaging features are available?",
            "docs": [
                {"source": "messaging.md", "sections": ["Features"]},
            ],
        },
        {
            "id": "ret_019",
            "question": "How do I post a job?",
            "docs": [
                {"source": "matching.md", "sections": ["Posting a Job"]},
                {"source": "r_template_job_description_android.md", "sections": ["Summary", "Key Points"]},
            ],
        },
        {
            "id": "ret_020",
            "question": "Where can I see VAT/tax summaries?",
            "docs": [
                {"source": "invoicing.md", "sections": ["Tax"]},
                {"source": "r_docs_invoices_vat_eu.md", "sections": ["Summary"]},
            ],
        },
        {
            "id": "ret_021",
            "question": "What benefits are included in the Enterprise plan?",
            "docs": [
                {"source": "pricing.md", "sections": ["Plans"]},
            ],
        },
        {
            "id": "ret_022",
            "question": "How do I enable notifications for milestone updates?",
            "docs": [
                {"source": "messaging.md", "sections": ["Tips"]},
            ],
        },
        {
            "id": "ret_023",
            "question": "How do I create a workspace and add members?",
            "docs": [
                {"source": "onboarding.md", "sections": ["Steps"]},
            ],
        },
        {
            "id": "ret_024",
            "question": "Give a high-level overview of RAG architecture basics.",
            "docs": [
                {"source": "r_docs_rag_architecture_basics.md", "sections": ["Summary", "Key Points"]},
            ],
        },
        {
            "id": "ret_028",
            "question": "How do fixed-price contracts differ from time & materials?",
            "docs": [
                {"source": "r_article_contracts_tnm_vs_fixed.md", "sections": ["Key Points"]},
                {"source": "contracts.md", "sections": ["Features"]},
                {"source": "r_docs_contracts_fixed_price.md", "sections": ["Key Points"]},
            ],
        },
        {
            "id": "ret_029",
            "question": "How can I defend against prompt injection attacks?",
            "docs": [
                {"source": "r_article_prompt_injection_defense.md", "sections": ["Summary", "Key Points"]},
                {"source": "security.md", "sections": ["Security Features"]},
            ],
        },
    ]

    items: List[Dict] = []
    for it in qdef:
        doc_entries = it["docs"]
        rel_chunks: List[str] = []
        src_docs: List[str] = []
        for de in doc_entries:
            src = de["source"]
            secs = de.get("sections", [])
            # collect chunk ids for requested sections; fallback to all chunks
            chunk_ids = chunks_for_sections(by_src, src, secs) if secs else []
            if not chunk_ids:
                chunk_ids = [x.get("id") for sec_items in by_src.get(src, {}).values() for x in sec_items]
            rel_chunks.extend(chunk_ids)
            src_docs.append(source_path(src))
        # dedupe and sort
        rel_chunks = sorted(dict.fromkeys(rel_chunks))
        src_docs = sorted(dict.fromkeys(src_docs))

        items.append({
            "id": it["id"],
            "question": it["question"],
            "source_docs": src_docs,
            "relevant_chunks": rel_chunks,
        })

    return items


def main():
    items = build_items()
    OUT_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} items to {OUT_PATH}")


if __name__ == "__main__":
    main()
