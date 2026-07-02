"""Command-line entrypoint for the evaluation harness."""

from pathlib import Path

from src.eval.harness import (
    default_eval_cases,
    load_eval_cases,
    render_report,
    run_eval_suite,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = project_root / "docs"
    index_dir = project_root / ".rag" / "faiss"
    cases_file = Path(__file__).resolve().with_name("cases.sample.json")

    cases = load_eval_cases(cases_file) or default_eval_cases()
    results = run_eval_suite(docs_dir=docs_dir, index_dir=index_dir, cases=cases)
    print(render_report(results))


if __name__ == "__main__":
    main()
