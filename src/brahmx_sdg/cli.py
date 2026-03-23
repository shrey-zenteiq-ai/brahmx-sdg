"""
BrahmX SDG CLI — Synthetic Data Generation pipeline.

Commands:
  ingest run         — ingest source files into a KB
  gold run           — generate one gold bundle from a task spec
  gold batch-run     — generate gold bundles for all specs in a directory
  silver run         — expand silver data from gold references
  corpus assemble    — assemble JSONL corpus from gold bundles
  corpus export      — list available bundles and stats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="brahmx-sdg",
    help="BrahmX SDG — Synthetic Data Generation Pipeline",
    add_completion=False,
)
console = Console()

# ── Sub-apps ──────────────────────────────────────────────────────────────────

gold_app = typer.Typer(name="gold", help="Gold data generation")
silver_app = typer.Typer(name="silver", help="Silver data expansion")
corpus_app = typer.Typer(name="corpus", help="Corpus assembly")
ingest_app = typer.Typer(name="ingest", help="Source document ingestion")

app.add_typer(ingest_app)
app.add_typer(gold_app)
app.add_typer(silver_app)
app.add_typer(corpus_app)


# ── Ingest ────────────────────────────────────────────────────────────────────

@ingest_app.command("run")
def ingest_run(
    source: Path = typer.Option(..., "--source", "-s", help="Path to source file or directory"),
    kb_dir: Path = typer.Option(Path("data/kb"), "--kb-dir", help="Knowledge base output directory"),
    domain: str = typer.Option("", "--domain", help="Domain tag for all ingested atoms"),
    chunk_size: int = typer.Option(2048, "--chunk-size", help="Characters per chunk (~512 tokens)"),
    license_tag: str = typer.Option("unknown", "--license"),
) -> None:
    """Ingest source files into a local knowledge base."""
    from brahmx_sdg.ingestion.source_ingestion import SourceIngestionPipeline

    pipeline = SourceIngestionPipeline(
        chunk_size=chunk_size,
        domain=domain,
        license_tag=license_tag,
        redistribution_allowed=True,
    )

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task(f"Ingesting {source}...", total=None)
        result = pipeline.run(source_path=source, output_dir=kb_dir)

    console.print(
        f"[green]✓ Ingested[/green] {result['atom_count']} chunks "
        f"from {result['doc_count']} documents → {kb_dir}/chunks/"
    )
    if result["rejected_count"]:
        console.print(f"[yellow]  {result['rejected_count']} files skipped[/yellow]")


# ── Gold ──────────────────────────────────────────────────────────────────────

@gold_app.command("run")
def gold_run(
    spec: Path = typer.Option(..., "--spec", help="Path to task spec JSON"),
    kb_path: Path = typer.Option(Path("data/kb"), "--kb-path", help="Knowledge base directory"),
    output_dir: Path = typer.Option(Path("data/gold"), "--output-dir"),
    routing_config: Path = typer.Option(
        Path("configs/routing/models.yaml"),
        "--routing-config",
        help="Model routing config",
    ),
    no_llm_dean: bool = typer.Option(False, "--no-llm-dean", help="Skip LLM rubric scoring in Dean"),
) -> None:
    """Run the full gold generation pipeline for a single task spec."""
    _check_openai_key()

    from brahmx_sdg.generation.gold_generator import GoldGenerator

    generator = GoldGenerator(routing_config=str(routing_config))
    if no_llm_dean:
        # Patch Dean to skip LLM scoring (faster/cheaper for quick tests)
        from brahmx_sdg.verification.dean import Dean
        generator._dean = Dean(
            routing_config=str(routing_config), use_llm_scoring=False
        )

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task(f"Generating gold bundle for {spec.stem}...", total=None)
        result = generator.run(spec_path=spec, kb_path=kb_path, output_dir=output_dir)

    if result.success:
        console.print(f"[green]✓ Gold bundle:[/green] {result.bundle_id}")
        console.print(f"  Bundle:  {result.output_path}")
        console.print(f"  Slices:  {output_dir}/slices/")
    else:
        console.print(f"[red]✗ Generation blocked:[/red] {result.reason}")
        raise typer.Exit(code=1)


@gold_app.command("batch-run")
def gold_batch_run(
    specs_dir: Path = typer.Option(..., "--specs-dir", help="Directory containing task spec JSON files"),
    kb_path: Path = typer.Option(Path("data/kb"), "--kb-path"),
    output_dir: Path = typer.Option(Path("data/gold"), "--output-dir"),
    routing_config: Path = typer.Option(Path("configs/routing/models.yaml"), "--routing-config"),
    no_llm_dean: bool = typer.Option(False, "--no-llm-dean"),
    stop_on_failure: bool = typer.Option(False, "--stop-on-failure"),
) -> None:
    """Batch-generate gold bundles for all *.json task specs in a directory."""
    _check_openai_key()

    specs = sorted(specs_dir.glob("*.json"))
    if not specs:
        console.print(f"[red]No *.json spec files found in {specs_dir}[/red]")
        raise typer.Exit(code=1)

    from brahmx_sdg.generation.gold_generator import GoldGenerator
    from brahmx_sdg.verification.dean import Dean

    generator = GoldGenerator(routing_config=str(routing_config))
    if no_llm_dean:
        generator._dean = Dean(routing_config=str(routing_config), use_llm_scoring=False)

    success_count = 0
    fail_count = 0

    table = Table(title=f"Batch Gold Generation — {len(specs)} specs")
    table.add_column("Spec", style="cyan")
    table.add_column("Result", justify="center")
    table.add_column("Bundle ID / Reason")

    for spec in specs:
        console.print(f"  Processing [cyan]{spec.stem}[/cyan]...", end="")
        result = generator.run(spec_path=spec, kb_path=kb_path, output_dir=output_dir)
        if result.success:
            success_count += 1
            table.add_row(spec.stem, "[green]PASS[/green]", result.bundle_id)
            console.print(f" [green]✓[/green]")
        else:
            fail_count += 1
            table.add_row(spec.stem, "[red]FAIL[/red]", result.reason[:80])
            console.print(f" [red]✗[/red]")
            if stop_on_failure:
                break

    console.print(table)
    console.print(
        f"\n[green]{success_count} succeeded[/green], "
        f"[red]{fail_count} failed[/red] out of {len(specs)} specs."
    )
    console.print(f"Bundles in: [bold]{output_dir}[/bold]")
    console.print(f"Slices in:  [bold]{output_dir}/slices/[/bold]")

    if fail_count and stop_on_failure:
        raise typer.Exit(code=1)


# ── Silver ────────────────────────────────────────────────────────────────────

@silver_app.command("run")
def silver_run(
    spec: Path = typer.Option(..., "--spec", help="Silver batch spec"),
    gold_ref: Path = typer.Option(..., "--gold-ref", help="Path to gold bundles"),
    output_dir: Path = typer.Option(Path("data/silver"), "--output-dir"),
) -> None:
    """Run silver data expansion pipeline."""
    _check_openai_key()
    from brahmx_sdg.generation.silver_generator import SilverGenerator

    generator = SilverGenerator()
    result = generator.run(spec_path=spec, gold_ref=gold_ref, output_dir=output_dir)
    console.print(
        f"[green]✓[/green] Generated {result['bundle_count']} silver bundles "
        f"({result['rejected_count']} rejected)"
    )


# ── Corpus ────────────────────────────────────────────────────────────────────

@corpus_app.command("assemble")
def corpus_assemble(
    manifest: Path = typer.Option(..., "--manifest", help="Assembly manifest YAML"),
    output: Path = typer.Option(Path("data/corpus"), "--output"),
) -> None:
    """Assemble a JSONL corpus from gold bundles."""
    from brahmx_sdg.corpus.corpus_assembler import CorpusAssembler

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Assembling corpus...", total=None)
        assembler = CorpusAssembler()
        result = assembler.run(manifest_path=manifest, output_path=output)

    console.print(f"[green]✓ Corpus {result.version}[/green] — {result.total_examples} examples")
    console.print(f"  Output:   {result.output_path}")
    console.print(f"  Manifest: {result.manifest_path}")
    if result.slices_by_type:
        table = Table(title="Slices by type")
        table.add_column("Task type")
        table.add_column("Count", justify="right")
        for ttype, cnt in sorted(result.slices_by_type.items()):
            table.add_row(ttype, str(cnt))
        console.print(table)


@corpus_app.command("stats")
def corpus_stats(
    gold_dir: Path = typer.Option(Path("data/gold"), "--gold-dir"),
) -> None:
    """Show stats for generated gold bundles."""
    bundles = sorted(gold_dir.glob("GOLD-*.json"))
    if not bundles:
        console.print(f"[yellow]No GOLD-*.json files found in {gold_dir}[/yellow]")
        return

    table = Table(title=f"Gold Bundles in {gold_dir}")
    table.add_column("Bundle ID", style="cyan")
    table.add_column("Section ID")
    table.add_column("Domain")
    table.add_column("Candidates", justify="right")
    table.add_column("Claims", justify="right")
    table.add_column("Auditor")

    for f in bundles[:50]:  # cap at 50 rows
        try:
            data = json.loads(f.read_text())
            table.add_row(
                data.get("record_id", "?")[:20],
                data.get("task_spec", {}).get("section_id", "?")[:20],
                data.get("task_spec", {}).get("domain", "?")[:16],
                str(len(data.get("candidates", []))),
                str(len(data.get("claim_ledger", {}).get("claims", []))),
                data.get("auditor_report", {}).get("status", "?"),
            )
        except Exception:
            table.add_row(f.stem, "parse error", "", "", "", "")

    console.print(table)
    console.print(f"\nTotal: [bold]{len(bundles)}[/bold] bundles")

    slices_dir = gold_dir / "slices"
    if slices_dir.exists():
        slice_count = len(list(slices_dir.glob("*.json")))
        console.print(f"Slices: [bold]{slice_count}[/bold] files in {slices_dir}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_openai_key() -> None:
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        console.print(
            "[red]ERROR:[/red] OPENAI_API_KEY is not set.\n"
            "Export it before running:\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "Or create a .env file and load it:\n"
            "  source .env"
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
