#!/usr/bin/env python3
"""Interactive Textual TUI for browsing M-IFEval instruction-following evaluation results.

USAGE:
    pip install textual  # Install dependency first
    python mifeval-viewer.py <language>

    Languages: en, ja, es, fr

FEATURES:
- Browse model responses and instruction-following results for a specific language
- View prompts, responses, and which instructions were followed/not followed
- Filter to show only failed prompts with 'f' key
- Vim-style search to filter models (press '/', type query, ESC to clear)
- Color-coded results (green=followed, red=not followed)
- Toggle between strict and loose evaluation modes

KEYBOARD SHORTCUTS:
    q - Quit
    f - Toggle filter to show only failed prompts (where not all instructions were followed)
    / - Search/filter models (vim-style)
    ESC - Clear search filter
    c - Toggle compact mode (truncate long text fields)
    m - Toggle evaluation mode (strict/loose)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rich.console import Group
from rich.table import Table
from rich.text import Text

BASE_DIR = Path(__file__).parent
SCORES_DIR = BASE_DIR / "scores"

# Available languages
LANGUAGES = ["en", "ja", "es", "fr"]

try:  # Defer Textual import so users get a clear message if it's missing.
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical, VerticalScroll
    from textual.reactive import reactive
    from textual.widgets import Button, Footer, Header, Input, ListItem, ListView, Select, Static
except ImportError as exc:  # pragma: no cover - executed only when Textual is absent
    App = None  # type: ignore[assignment]
    ComposeResult = None  # type: ignore[assignment]
    Container = Horizontal = Vertical = VerticalScroll = None  # type: ignore[assignment]
    reactive = None  # type: ignore[assignment]
    Button = Footer = Header = Input = ListItem = ListView = Select = Static = None  # type: ignore[assignment]
    TEXTUAL_IMPORT_ERROR = exc
else:
    TEXTUAL_IMPORT_ERROR = None


def display_model_name(safe_name: str) -> str:
    """Convert safe filename to display name (replace __ with /)"""
    return safe_name.replace("__", "/")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of records"""
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file and return dict"""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def safe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


@dataclass
class InstructionRecord:
    """Single prompt-response with instruction-following results"""
    index: int
    prompt: str
    response: str
    instruction_ids: List[str]
    follow_all_instructions: bool
    follow_instruction_list: List[bool]
    raw: Dict[str, Any]

    @property
    def is_correct(self) -> bool:
        """Returns True if all instructions were followed"""
        return self.follow_all_instructions

    @property
    def is_incorrect(self) -> bool:
        """Returns True if not all instructions were followed"""
        return not self.follow_all_instructions

    @property
    def num_instructions(self) -> int:
        """Total number of instructions"""
        return len(self.instruction_ids)

    @property
    def num_followed(self) -> int:
        """Number of instructions that were followed"""
        return sum(1 for followed in self.follow_instruction_list if followed)

    @property
    def num_not_followed(self) -> int:
        """Number of instructions that were not followed"""
        return sum(1 for followed in self.follow_instruction_list if not followed)

    def get_status_icon(self) -> str:
        """Return visual indicator for correctness"""
        if self.follow_all_instructions:
            return "✅"
        else:
            # Partial success
            if self.num_followed > 0:
                return "⚠️"
            return "❌"


@dataclass
class ModelData:
    safe_name: str
    display_name: str
    language: str
    records: List[InstructionRecord]
    scores: Dict[str, Any]

    @property
    def num_correct(self) -> int:
        return sum(1 for r in self.records if r.is_correct)

    @property
    def num_incorrect(self) -> int:
        return sum(1 for r in self.records if r.is_incorrect)

    @property
    def total_prompts(self) -> int:
        return len(self.records)

    def get_accuracy(self, mode: str = "strict") -> Optional[float]:
        """Calculate prompt-level accuracy for given mode (strict/loose)"""
        if not self.scores:
            return None
        mode_data = self.scores.get(mode, {})
        acc = mode_data.get("prompt_level_accuracy")
        return acc * 100 if acc is not None else None

    def get_instruction_accuracy(self, mode: str = "strict") -> Optional[float]:
        """Calculate instruction-level accuracy for given mode"""
        if not self.scores:
            return None
        mode_data = self.scores.get(mode, {})
        acc = mode_data.get("instruction_level_accuracy")
        return acc * 100 if acc is not None else None


def list_models_in_language(language: str, scores_dir: Path) -> List[str]:
    """List all models that have answers for the specified language"""
    models = set()

    # Pattern: {lang}_input_{model}_answers.jsonl
    for answers_file in scores_dir.glob(f"{language}_input_*_answers.jsonl"):
        # Extract model name from filename
        filename = answers_file.stem
        # Remove "{lang}_input_" prefix and "_answers" suffix
        model_name = filename.replace(f"{language}_input_", "").replace("_answers", "")
        models.add(model_name)

    return sorted(models)


def load_model_data(
    model_safe_name: str,
    language: str,
    scores_dir: Path,
) -> Optional[ModelData]:
    """Load all prompt responses and instruction results for a model in a language"""

    # Load answers: {lang}_input_{model}_answers.jsonl
    answers_path = scores_dir / f"{language}_input_{model_safe_name}_answers.jsonl"
    if not answers_path.exists():
        return None

    answers = load_jsonl(answers_path)
    if not answers:
        return None

    # Load scores: {lang}_input_{model}_scores.json
    scores_path = scores_dir / f"{language}_input_{model_safe_name}_scores.json"
    scores_data = load_json(scores_path)

    # Process records
    records = []
    for idx, record in enumerate(answers):
        instruction_record = InstructionRecord(
            index=idx,
            prompt=record.get("prompt", ""),
            response=record.get("response", ""),
            instruction_ids=record.get("instruction_id_list", []),
            follow_all_instructions=record.get("follow_all_instructions", False),
            follow_instruction_list=record.get("follow_instruction_list", []),
            raw=record,
        )
        records.append(instruction_record)

    if not records:
        return None

    return ModelData(
        safe_name=model_safe_name,
        display_name=display_model_name(model_safe_name),
        language=language,
        records=records,
        scores=scores_data,
    )


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length characters if longer"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [press 'c' to expand]"


def format_instruction_result(instruction_id: str, followed: bool) -> Text:
    """Format instruction result with color coding"""
    if followed:
        return Text(f"✓ {instruction_id}", style="green")
    else:
        return Text(f"✗ {instruction_id}", style="bold red")


def build_record_renderable(record: InstructionRecord, compact: bool = False) -> Group:
    """Build renderable for a single instruction-following record"""
    # Icon and header
    icon = record.get_status_icon()
    icon_style = "green" if icon == "✅" else "bold red" if icon == "❌" else "yellow"

    header = Text()
    header.append(icon + " ", style=icon_style)
    header.append(f"Prompt {record.index + 1}", style="bold cyan")
    header.append(" • ", style="dim")
    if record.follow_all_instructions:
        header.append(f"ALL INSTRUCTIONS FOLLOWED ({record.num_instructions}/{record.num_instructions})", style="bold green")
    else:
        header.append(f"PARTIAL ({record.num_followed}/{record.num_instructions} followed)", style="yellow" if record.num_followed > 0 else "bold red")

    # Main content table
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", width=20, style="bold", no_wrap=True)
    table.add_column(justify="center", width=1, style="dim", no_wrap=True)
    table.add_column(ratio=1, overflow="fold")

    # Prompt
    prompt_text = truncate_text(record.prompt, 400) if compact else record.prompt
    table.add_row("Prompt", "|", Text(prompt_text, style="bold white"))

    # Response
    response_text = truncate_text(record.response, 400) if compact else record.response
    table.add_row("Response", "|", Text(response_text, style="cyan"))

    # Instructions section
    table.add_row("", "", Text(""))  # Spacing
    table.add_row("Instructions", "|", Text(f"{record.num_instructions} total", style="dim"))

    # List each instruction and whether it was followed
    for instruction_id, followed in zip(record.instruction_ids, record.follow_instruction_list):
        result_text = format_instruction_result(instruction_id, followed)
        table.add_row("", "|", result_text)

    return Group(header, table)


def build_records_renderable(records: List[InstructionRecord], compact: bool = False) -> Group:
    """Build renderable for all records"""
    if not records:
        return Group(Text("No records found.", style="dim"))

    renderables: List[Any] = []
    for record in records:
        renderables.append(build_record_renderable(record, compact=compact))
        renderables.append(Text(""))  # Spacing

    return Group(*renderables)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactive viewer for M-IFEval instruction-following evaluation results."
    )
    parser.add_argument(
        "language",
        choices=LANGUAGES,
        help="Language to view (en, ja, es, fr)"
    )
    parser.add_argument("--model", help="Model name to preselect (safe or display name)")
    parser.add_argument("--scores-dir", type=Path, default=SCORES_DIR, help="Path to scores directory")
    parser.add_argument("--mode", choices=["strict", "loose"], default="strict", help="Evaluation mode (default: strict)")
    args = parser.parse_args(argv)

    language = args.language
    scores_dir = args.scores_dir

    # Check for textual
    if App is None or TEXTUAL_IMPORT_ERROR is not None:
        print("This viewer requires the 'textual' package. Install it with `pip install textual`.")
        if TEXTUAL_IMPORT_ERROR is not None:
            print(f"Import error: {TEXTUAL_IMPORT_ERROR}")
        return 1

    # Check if any data exists for this language
    model_names = list_models_in_language(language, scores_dir)
    if not model_names:
        print(f"No models found for language '{language}' in {scores_dir}")
        print(f"Available languages: {', '.join(LANGUAGES)}")
        return 1

    app = MIFEvalViewerApp(
        language=language,
        model_names=model_names,
        scores_dir=scores_dir,
        preselect_model=args.model,
        initial_mode=args.mode,
    )
    app.run()
    return 0


if App is not None:

    class ModelListItem(ListItem):
        def __init__(self, model: ModelData, mode: str = "strict"):
            # Format model name with accuracy if available
            label = model.display_name
            acc = model.get_accuracy(mode)
            if acc is not None:
                label += f" ({acc:.1f}%)"
            super().__init__(Static(label))
            self.model = model

    class MIFEvalViewerApp(App):
        CSS = """
        Screen {
            layout: vertical;
        }
        #body {
            layout: horizontal;
            height: 1fr;
        }
        #sidebar {
            width: 45;
            min-width: 35;
            height: 1fr;
            border: solid $surface-darken-1;
            padding: 1 0;
        }
        #language-info {
            margin: 0 1 1 1;
            color: $text-muted;
        }
        #model-search {
            margin: 0 1 1 1;
            height: auto;
        }
        #model-list {
            height: 1fr;
            margin: 0 1 0 1;
            overflow: auto;
        }
        #main {
            layout: vertical;
            width: 1fr;
            height: 1fr;
            padding: 1;
        }
        #content-summary {
            min-height: 1;
            margin-bottom: 1;
        }
        #details-panel {
            height: 1fr;
            border: solid $surface-darken-1;
            padding: 0 1;
        }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("f", "toggle_incorrect_only", "Toggle failed only"),
            ("c", "toggle_compact", "Toggle compact mode"),
            ("m", "toggle_mode", "Toggle strict/loose mode"),
            ("/", "focus_search", "Search"),
            ("escape", "clear_search", "Clear search"),
        ]

        show_incorrect_only = reactive(False)
        compact_mode = reactive(False)
        eval_mode = reactive("strict")
        search_query = reactive("")

        def __init__(
            self,
            language: str,
            model_names: List[str],
            scores_dir: Path,
            preselect_model: Optional[str] = None,
            initial_mode: str = "strict",
        ) -> None:
            super().__init__()
            self.language = language
            self.model_names = model_names
            self.scores_dir = scores_dir
            self.preselect_model = preselect_model
            self.models: List[ModelData] = []
            self.selected_model: Optional[ModelData] = None
            self._is_mounted = False
            self.eval_mode = initial_mode

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="body"):
                with Vertical(id="sidebar"):
                    yield Static(f"Language: {self.language.upper()}", id="language-info")
                    yield Input(placeholder="Search models (/)", id="model-search")
                    yield ListView(id="model-list")
                with Vertical(id="main"):
                    yield Static("", id="content-summary")
                    with VerticalScroll(id="details-panel"):
                        yield Static("Select a model to view instruction-following results.", id="details-content")
            yield Footer()

        def on_mount(self) -> None:
            self.title = f"M-IFEval • {self.language.upper()} • {self.eval_mode.upper()}"
            self._load_models()
            self._populate_models()
            self._is_mounted = True

        def action_toggle_incorrect_only(self) -> None:
            self.show_incorrect_only = not self.show_incorrect_only

        def action_toggle_compact(self) -> None:
            self.compact_mode = not self.compact_mode

        def action_toggle_mode(self) -> None:
            """Toggle between strict and loose evaluation modes"""
            self.eval_mode = "loose" if self.eval_mode == "strict" else "strict"
            self.title = f"M-IFEval • {self.language.upper()} • {self.eval_mode.upper()}"
            # Refresh model list to show updated accuracies
            self._populate_models()
            # Refresh content with new mode
            self._refresh_content()

        def action_focus_search(self) -> None:
            """Focus the search input"""
            search_input = self.query_one("#model-search", Input)
            search_input.focus()

        def action_clear_search(self) -> None:
            """Clear the search query"""
            search_input = self.query_one("#model-search", Input)
            search_input.value = ""
            # Focus back to model list
            model_list = self.query_one("#model-list", ListView)
            model_list.focus()

        def watch_show_incorrect_only(self, _: bool) -> None:
            self._refresh_content()

        def watch_compact_mode(self, _: bool) -> None:
            self._refresh_content()

        def watch_search_query(self, query: str) -> None:
            """Update model list when search query changes"""
            if self._is_mounted:
                self._populate_models()

        def on_input_changed(self, event: Input.Changed) -> None:
            """Handle search input changes"""
            if event.input.id == "model-search":
                self.search_query = event.value

        def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
            if isinstance(event.item, ModelListItem):
                self._select_model(event.item.model)

        def _load_models(self) -> None:
            """Load all model data"""
            self.models = []
            for model_name in self.model_names:
                model_data = load_model_data(
                    model_name,
                    self.language,
                    self.scores_dir,
                )
                if model_data:
                    self.models.append(model_data)

            # Sort by accuracy (descending) using current mode
            self.models.sort(key=lambda m: m.get_accuracy(self.eval_mode) or 0, reverse=True)

        def _populate_models(self) -> None:
            """Populate model list with optional search filtering"""
            model_list = self.query_one("#model-list", ListView)
            model_list.clear()

            if not self.models:
                model_list.append(ListItem(Static("No models found.")))
                return

            # Filter models based on search query
            filtered_models = self.models
            if self.search_query:
                query_lower = self.search_query.lower()
                filtered_models = [
                    m for m in self.models
                    if query_lower in m.safe_name.lower() or query_lower in m.display_name.lower()
                ]

            if not filtered_models:
                model_list.append(ListItem(Static(f"No models matching '{self.search_query}'")))
                return

            for model in filtered_models:
                model_list.append(ModelListItem(model, mode=self.eval_mode))

            # Preselect model if specified
            if self.preselect_model and not self.search_query:
                index = self._find_model_index(self.preselect_model, filtered_models)
            else:
                index = 0

            index = max(0, min(index, len(filtered_models) - 1))
            if filtered_models:
                model_list.index = index

        def _find_model_index(self, target: Optional[str], models: Optional[List[ModelData]] = None) -> int:
            if not target:
                return 0
            if models is None:
                models = self.models
            target_lower = target.lower()
            for idx, model in enumerate(models):
                if model.safe_name.lower() == target_lower or model.display_name.lower() == target_lower:
                    return idx
            return 0

        def _select_model(self, model: Optional[ModelData]) -> None:
            """Select a model and show its instruction-following results"""
            if model is None:
                self.selected_model = None
                self._refresh_content()
                return

            self.selected_model = model
            self._refresh_content()

        def _refresh_content(self) -> None:
            """Refresh the content panel"""
            summary = self.query_one("#content-summary", Static)
            details = self.query_one("#details-content", Static)

            if not self.selected_model:
                summary.update("Select a model.")
                details.update(Text("Select a model to display instruction-following results."))
                return

            model = self.selected_model
            records = model.records
            if self.show_incorrect_only:
                records = [r for r in records if r.is_incorrect]

            # Update summary
            summary_text = f"{model.display_name}: {len(records)} prompts"
            if self.show_incorrect_only:
                summary_text += " (failed only)"
            if self.compact_mode:
                summary_text += " [COMPACT]"

            # Add accuracy stats for current mode
            prompt_acc = model.get_accuracy(self.eval_mode)
            inst_acc = model.get_instruction_accuracy(self.eval_mode)
            if prompt_acc is not None:
                summary_text += f" • Prompt-Level: {prompt_acc:.1f}%"
            if inst_acc is not None:
                summary_text += f" • Instruction-Level: {inst_acc:.1f}%"
            summary_text += f" [{self.eval_mode.upper()}]"

            summary.update(summary_text)

            # Update details
            if not records:
                details.update(Text("No records match the current filter."))
                return

            details.update(build_records_renderable(records, compact=self.compact_mode))


if __name__ == "__main__":
    sys.exit(main())
