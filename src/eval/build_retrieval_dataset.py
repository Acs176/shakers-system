import json
import pathlib
from typing import Dict, List


META_PATH = pathlib.Path("rag_index/meta.json")
OUT_PATH = pathlib.Path("kb/retrieval_eval_dataset.json")


def load_meta() -> List[Dict]:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return meta


def index_meta_by_source_and_section(meta: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    by_src: Dict[str, Dict[str, List[Dict]]] = {}
    for item in meta:
        src = item.get("source")
        sec = item.get("section") or "Summary"
        by_src.setdefault(src, {}).setdefault(sec, []).append(item)
    return by_src


def chunks_for_sections(by_src: Dict[str, Dict[str, List[Dict]]], source: str, sections: List[str]) -> List[str]:
    out: List[str] = []
    for sec in sections:
        for it in by_src.get(source, {}).get(sec, []):
            out.append(it.get("id"))
    return out


def build_items() -> List[Dict]:
    meta = load_meta()
    by_src = index_meta_by_source_and_section(meta)

    qdef = [
        {
            "id": "ret_001",
            "question": "What is Shakers and what does it offer?",
            "doc": "overview.md",
            "sections": ["Summary", "Key Features"],
        },
        {
            "id": "ret_002",
            "question": "Outline the payment flow for a milestone on Shakers.",
            "doc": "payments.md",
            "sections": ["Payment Flow"],
        },
        {
            "id": "ret_003",
            "question": "Are refunds possible before a milestone is approved?",
            "doc": "payments.md",
            "sections": ["Notes"],
        },
        {
            "id": "ret_004",
            "question": "How long do freelancer payouts take after approval?",
            "doc": "payments.md",
            "sections": ["Notes"],
        },
        {
            "id": "ret_005",
            "question": "How are invoices generated on Shakers?",
            "doc": "invoicing.md",
            "sections": ["Summary", "Invoices"],
        },
        {
            "id": "ret_006",
            "question": "Does Shakers calculate VAT or sales tax on invoices?",
            "doc": "invoicing.md",
            "sections": ["Tax"],
        },
        {
            "id": "ret_007",
            "question": "What security features are available?",
            "doc": "security.md",
            "sections": ["Security Features"],
        },
        {
            "id": "ret_008",
            "question": "What are the main roles and their permissions?",
            "doc": "roles.md",
            "sections": ["Roles", "Permissions"],
        },
        {
            "id": "ret_009",
            "question": "What are the steps to onboard a new company user?",
            "doc": "onboarding.md",
            "sections": ["Steps"],
        },
        {
            "id": "ret_010",
            "question": "Define a milestone and escrow in Shakers.",
            "doc": "glossary.md",
            "sections": ["Summary"],
        },
        {
            "id": "ret_011",
            "question": "How does the dispute resolution process work?",
            "doc": "disputes.md",
            "sections": ["Process"],
        },
        {
            "id": "ret_012",
            "question": "What plans does Shakers offer and how is billing handled?",
            "doc": "pricing.md",
            "sections": ["Plans", "Billing"],
        },
        {
            "id": "ret_013",
            "question": "What criteria does the freelancer matching consider?",
            "doc": "matching.md",
            "sections": ["Matching Criteria"],
        },
        {
            "id": "ret_014",
            "question": "How are projects structured and how is time tracked?",
            "doc": "projects.md",
            "sections": ["Milestones", "Time Tracking"],
        },
        {
            "id": "ret_015",
            "question": "How can I troubleshoot a 'Login failed' issue?",
            "doc": "faq.md",
            "sections": ["Common Issues", "Troubleshooting Steps"],
        },
        {
            "id": "ret_016",
            "question": "What are best practices around contracts and NDAs?",
            "doc": "contracts.md",
            "sections": ["Features", "Best Practices"],
        },
        {
            "id": "ret_017",
            "question": "What are the steps in the talent vetting process?",
            "doc": "vetting.md",
            "sections": ["Steps", "Result"],
        },
        {
            "id": "ret_018",
            "question": "What messaging features are available?",
            "doc": "messaging.md",
            "sections": ["Features"],
        },
        {
            "id": "ret_019",
            "question": "How do I post a job?",
            "doc": "matching.md",
            "sections": ["Posting a Job"],
        },
        {
            "id": "ret_020",
            "question": "Where can I see VAT/tax summaries?",
            "doc": "invoicing.md",
            "sections": ["Tax"],
        },
        {
            "id": "ret_021",
            "question": "What benefits are included in the Enterprise plan?",
            "doc": "pricing.md",
            "sections": ["Plans"],
        },
        {
            "id": "ret_022",
            "question": "How do I enable notifications for milestone updates?",
            "doc": "messaging.md",
            "sections": ["Tips"],
        },
        {
            "id": "ret_023",
            "question": "How do I create a workspace and add members?",
            "doc": "onboarding.md",
            "sections": ["Steps"],
        },
    ]

    items: List[Dict] = []
    for it in qdef:
        src = it["doc"]
        secs = it["sections"]
        rel_chunks = chunks_for_sections(by_src, src, secs)
        if not rel_chunks:
            # Fallback: mark all chunks from the doc relevant
            rel_chunks = [x.get("id") for sec_items in by_src.get(src, {}).values() for x in sec_items]
        items.append({
            "id": it["id"],
            "question": it["question"],
            "source_docs": [f"kb/{src}"],
            "relevant_chunks": sorted(rel_chunks),
        })

    return items


def main():
    items = build_items()
    OUT_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} items to {OUT_PATH}")


if __name__ == "__main__":
    main()

