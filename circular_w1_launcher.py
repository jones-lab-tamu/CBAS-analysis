"""Small Tkinter wrapper for the existing Circular W1 analysis and figure scripts."""

from __future__ import annotations

import csv
import os
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from phase_behavior_mutual_information import (
    INPUT_FILENAME_PATTERN,
    discover_input_files,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_SCRIPT = SCRIPT_DIR / "circular_w1_repertoire_analysis.py"
FIGURE_SCRIPT = SCRIPT_DIR / "circular_w1_repertoire_figure.py"
OUTPUT_DIRNAME = "Circular_W1_Repertoire"
METADATA_FILENAME = "cohort_metadata.csv"
FIGURE_OUTPUT_DIRNAME = "Cohort_Figure_Visualization"
MORE_THAN_TWO_GROUPS_MESSAGE = (
    "Circular W1 analysis supports this cohort, but the current cohort figure "
    "requires exactly two groups."
)
CHECKED = "☑"
UNCHECKED = "☐"


def _looks_like_animal_folder(path: Path) -> bool:
    """Use only immediate animal-folder signatures; never recurse for identity."""

    if (path / "FRP_Phase_Output").exists():
        return True
    return any(INPUT_FILENAME_PATTERN.fullmatch(file.name) for file in path.glob("*.csv"))


def scan_cohort(cohort_root: Path) -> list[dict[str, object]]:
    """Detect immediate group/animal folders with a source or phase signature."""

    cohort_root = cohort_root.expanduser().resolve()
    if not cohort_root.is_dir():
        raise FileNotFoundError(f"Cohort folder does not exist: {cohort_root}")

    records: list[dict[str, object]] = []
    group_dirs = sorted(
        (path for path in cohort_root.iterdir() if path.is_dir()),
        key=lambda path: (path.name.casefold(), path.name),
    )
    for group_dir in group_dirs:
        animal_dirs = sorted(
            (path for path in group_dir.iterdir() if path.is_dir()),
            key=lambda path: (path.name.casefold(), path.name),
        )
        for animal_dir in animal_dirs:
            if _looks_like_animal_folder(animal_dir):
                records.append(
                    {
                        "include": True,
                        "group": group_dir.name,
                        "animal": animal_dir.name,
                    }
                )
    return records


def build_metadata_rows(
    records: list[dict[str, object]],
) -> list[dict[str, str | int]]:
    """Validate included rows and assign display order from their visible order."""

    included = [record for record in records if bool(record["include"])]
    if len(included) < 2:
        raise ValueError("At least two animals must be included.")

    normalized_ids: dict[str, list[str]] = {}
    for record in included:
        animal = str(record["animal"])
        normalized_ids.setdefault(animal.casefold(), []).append(animal)
    duplicates = sorted(
        {animal for animal_ids in normalized_ids.values() if len(animal_ids) > 1 for animal in animal_ids},
        key=str.casefold,
    )
    if duplicates:
        raise ValueError(
            "Duplicate animal IDs are included: " + ", ".join(duplicates)
        )

    rows: list[dict[str, str | int]] = []
    for display_order, record in enumerate(included, start=1):
        animal = str(record["animal"])
        group = str(record["group"])
        if not animal.strip():
            raise ValueError("An included animal ID is blank.")
        if not group.strip():
            raise ValueError(f"{animal}: group name is blank.")
        rows.append(
            {
                "animal": animal,
                "group": group,
                "display_order": display_order,
            }
        )
    return rows


def check_inputs(
    cohort_root: Path | None,
    records: list[dict[str, object]],
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Check filesystem structure only; scientific validation stays in the backend."""

    if cohort_root is None or not cohort_root.is_dir():
        return [], ["Select an existing cohort folder first."]

    included = [record for record in records if bool(record["include"])]
    cohort_errors: list[str] = []
    if not included:
        cohort_errors.append("No animals are included or detected.")
    elif len(included) < 2:
        cohort_errors.append("At least two animals must be included to run analysis.")

    id_counts: dict[str, int] = {}
    for record in included:
        animal = str(record["animal"])
        id_counts[animal.casefold()] = id_counts.get(animal.casefold(), 0) + 1

    reports: list[tuple[str, list[str]]] = []
    for record in included:
        group = str(record["group"])
        animal = str(record["animal"])
        label = f"{animal} [{group}]"
        issues: list[str] = []
        if not animal.strip():
            issues.append("animal ID is blank")
        elif id_counts[animal.casefold()] > 1:
            issues.append("duplicate animal ID is included more than once")
        if not group.strip():
            issues.append("group name is blank")
            reports.append((label, issues))
            continue

        animal_dir = cohort_root / group / animal
        if not animal_dir.is_dir():
            issues.append(f"missing animal directory: {animal_dir}")
            reports.append((label, issues))
            continue

        try:
            input_files, missing_indices = discover_input_files(animal_dir)
        except FileNotFoundError:
            issues.append("no matching CBAS source CSV files")
        except ValueError as error:
            issues.append(str(error))
        else:
            source_animals = {
                match.group("animal")
                for _, file_path in input_files
                if (match := INPUT_FILENAME_PATTERN.fullmatch(file_path.name))
                is not None
            }
            if source_animals != {animal}:
                issues.append(
                    "source-file prefix does not match the animal ID "
                    f"(found: {', '.join(sorted(source_animals)) or 'none'})"
                )
            if missing_indices:
                issues.append(
                    "source-file indices are not contiguous; missing "
                    + ", ".join(str(index) for index in missing_indices)
                )

        phase_dir = animal_dir / "FRP_Phase_Output"
        if not phase_dir.is_dir():
            issues.append("missing FRP_Phase_Output directory")
        else:
            for filename in ("frp_phase_summary.csv", "frp_phase_cycles.csv"):
                if not (phase_dir / filename).is_file():
                    issues.append(f"missing FRP_Phase_Output\\{filename}")
        reports.append((label, issues))

    return reports, cohort_errors


def write_metadata(
    cohort_root: Path,
    rows: list[dict[str, str | int]],
) -> Path:
    output_dir = cohort_root / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / METADATA_FILENAME
    with metadata_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["animal", "group", "display_order"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return metadata_path


class CircularW1Launcher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Circular W1 Cohort Analysis")
        self.geometry("940x760")
        self.minsize(760, 620)

        self.cohort_path_var = tk.StringVar()
        self.control_group_var = tk.StringVar()
        self.experimental_group_var = tk.StringVar()
        self.records: list[dict[str, object]] = []
        self._tree_records: dict[str, dict[str, object]] = {}
        self._output_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._busy = False
        self._more_than_two_groups_logged = False
        self._process_kind = ""
        self._analysis_output_dir: Path | None = None
        self._figure_output_dir: Path | None = None

        self._build_window()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._drain_output_queue)
        self.append_log("Choose a cohort folder to begin.")

    @property
    def cohort_root(self) -> Path | None:
        value = self.cohort_path_var.get().strip()
        return Path(value).expanduser().resolve() if value else None

    def _build_window(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)
        outer.rowconfigure(7, weight=1)

        ttk.Label(
            outer,
            text="Circular W1 Cohort Analysis",
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        path_frame = ttk.Frame(outer)
        path_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        path_frame.columnconfigure(1, weight=1)
        ttk.Label(path_frame, text="Cohort folder:").grid(row=0, column=0, padx=(0, 8))
        self.path_entry = ttk.Entry(path_frame, textvariable=self.cohort_path_var)
        self.path_entry.grid(row=0, column=1, sticky="ew")
        self.path_entry.bind("<Return>", lambda _event: self.load_cohort_folder())
        self.browse_button = ttk.Button(
            path_frame,
            text="Browse",
            command=self.browse_cohort_folder,
        )
        self.browse_button.grid(row=0, column=2, padx=(8, 0))

        ttk.Label(outer, text="Detected cohort:").grid(
            row=2, column=0, sticky="w", pady=(2, 4)
        )
        table_frame = ttk.Frame(outer)
        table_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 8))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.cohort_table = ttk.Treeview(
            table_frame,
            columns=("include", "group", "animal"),
            show="headings",
            height=9,
            selectmode="browse",
        )
        self.cohort_table.heading("include", text="Include")
        self.cohort_table.heading("group", text="Group")
        self.cohort_table.heading("animal", text="Animal")
        self.cohort_table.column("include", width=100, anchor="center", stretch=False)
        self.cohort_table.column("group", width=220, anchor="w")
        self.cohort_table.column("animal", width=180, anchor="w")
        self.cohort_table.grid(row=0, column=0, sticky="nsew")
        self.cohort_table.bind("<Button-1>", self._on_table_click)
        table_scroll = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.cohort_table.yview,
        )
        table_scroll.grid(row=0, column=1, sticky="ns")
        self.cohort_table.configure(yscrollcommand=table_scroll.set)
        ttk.Label(
            outer,
            text=(
                "Click a checkbox to include or exclude an animal. Group and Animal "
                "are read-only folder names; correct the cohort folder structure "
                "if either label is wrong."
            ),
        ).grid(row=4, column=0, sticky="w", pady=(0, 8))

        controls = ttk.Frame(outer)
        controls.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self.check_button = ttk.Button(
            controls,
            text="Check Inputs",
            command=self.on_check_inputs,
        )
        self.check_button.pack(side="left", padx=(0, 12))
        ttk.Label(controls, text="Control group:").pack(side="left", padx=(0, 5))
        self.control_combo = ttk.Combobox(
            controls,
            textvariable=self.control_group_var,
            state="readonly",
            width=18,
        )
        self.control_combo.pack(side="left", padx=(0, 14))
        ttk.Label(controls, text="Experimental group:").pack(side="left", padx=(0, 5))
        self.experimental_combo = ttk.Combobox(
            controls,
            textvariable=self.experimental_group_var,
            state="readonly",
            width=18,
        )
        self.experimental_combo.pack(side="left")

        run_frame = ttk.Frame(outer)
        run_frame.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        self.analysis_button = ttk.Button(
            run_frame,
            text="Run Circular W1 Analysis",
            command=self.on_run_analysis,
        )
        self.analysis_button.pack(side="left", padx=(0, 8))
        self.figure_button = ttk.Button(
            run_frame,
            text="Generate Circular W1 Figure",
            command=self.on_run_figure,
        )
        self.figure_button.pack(side="left", padx=(0, 8))
        self.open_folder_button = ttk.Button(
            run_frame,
            text="Open Output Folder",
            command=self.open_figure_output_folder,
            state="disabled",
        )
        self.open_folder_button.pack(side="left")

        ttk.Label(outer, text="Status / output log:").grid(
            row=7, column=0, sticky="w", pady=(2, 4)
        )
        log_frame = ttk.Frame(outer)
        log_frame.grid(row=8, column=0, sticky="nsew")
        outer.rowconfigure(8, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=14, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.log_text.yview,
        )
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self._refresh_group_choices()

    def append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def browse_cohort_folder(self) -> None:
        current = self.cohort_root
        selected = filedialog.askdirectory(
            title="Choose cohort folder",
            initialdir=str(current if current and current.is_dir() else Path.home()),
            mustexist=True,
        )
        if selected:
            self.cohort_path_var.set(selected)
            self.load_cohort_folder()

    def load_cohort_folder(self) -> None:
        self._analysis_output_dir = None
        self._figure_output_dir = None
        self.open_folder_button.configure(state="disabled")
        root = self.cohort_root
        if root is None:
            self.records = []
            self._render_records()
            self._refresh_group_choices()
            self.append_log("No cohort folder selected.")
            return
        try:
            self.records = scan_cohort(root)
        except (OSError, ValueError) as error:
            self.records = []
            self._render_records()
            self._refresh_group_choices()
            self.append_log(f"Could not scan cohort folder: {error}")
            return
        self._render_records()
        if not self.records:
            self.append_log(
                "No animals found. Check that the cohort structure is "
                "<cohort_root>/<group>/<animal>, with matching CBAS source files "
                "or an FRP_Phase_Output folder. Correct the folder organization "
                "if it differs; group labels are read from parent folder names."
            )
        else:
            self.append_log(f"Detected {len(self.records)} candidate animals in immediate group/animal folders.")
        self._refresh_group_choices()

    def _render_records(self) -> None:
        self.cohort_table.delete(*self.cohort_table.get_children())
        self._tree_records.clear()
        for index, record in enumerate(self.records):
            item_id = f"animal_{index}"
            self._tree_records[item_id] = record
            self.cohort_table.insert(
                "",
                "end",
                iid=item_id,
                values=(
                    CHECKED if bool(record["include"]) else UNCHECKED,
                    record["group"],
                    record["animal"],
                ),
            )

    def _on_table_click(self, event: tk.Event) -> str | None:
        if self._busy or self.cohort_table.identify_region(event.x, event.y) != "cell":
            return None
        if self.cohort_table.identify_column(event.x) != "#1":
            return None
        item_id = self.cohort_table.identify_row(event.y)
        if item_id not in self._tree_records:
            return None
        record = self._tree_records[item_id]
        record["include"] = not bool(record["include"])
        self.cohort_table.set(
            item_id,
            "include",
            CHECKED if bool(record["include"]) else UNCHECKED,
        )
        self._refresh_group_choices()
        return "break"

    def _included_records(self) -> list[dict[str, object]]:
        return [
            self._tree_records[item_id]
            for item_id in self.cohort_table.get_children()
            if bool(self._tree_records[item_id]["include"])
        ]

    def _refresh_group_choices(self) -> None:
        groups = list(
            dict.fromkeys(str(record["group"]) for record in self._included_records())
        )
        old_control = self.control_group_var.get()
        old_experimental = self.experimental_group_var.get()
        self.control_combo.configure(values=groups)
        self.experimental_combo.configure(values=groups)

        exactly_two_groups = len(groups) == 2
        if exactly_two_groups:
            if old_control not in groups or old_experimental not in groups or old_control == old_experimental:
                self.control_group_var.set(groups[0])
                self.experimental_group_var.set(groups[1])
        else:
            self.control_group_var.set("")
            self.experimental_group_var.set("")

        combo_state = "readonly" if exactly_two_groups and not self._busy else "disabled"
        self.control_combo.configure(state=combo_state)
        self.experimental_combo.configure(state=combo_state)
        self.figure_button.configure(
            state="normal" if exactly_two_groups and not self._busy else "disabled"
        )
        if len(groups) > 2 and not self._more_than_two_groups_logged:
            self.append_log(MORE_THAN_TWO_GROUPS_MESSAGE)
            self._more_than_two_groups_logged = True
        elif len(groups) <= 2:
            self._more_than_two_groups_logged = False

    def _write_check_report(
        self,
        reports: list[tuple[str, list[str]]],
        cohort_errors: list[str],
    ) -> bool:
        self.append_log("CHECK INPUTS")
        for error in cohort_errors:
            self.append_log(f"Cohort: {error}")
        passed = 0
        for label, issues in reports:
            if issues:
                self.append_log(f"{label}: " + "; ".join(issues))
            else:
                passed += 1
                self.append_log(f"{label}: OK")
        if reports:
            self.append_log(f"{passed} of {len(reports)} included animals passed structural checks.")
        elif not cohort_errors:
            self.append_log("No animals found to check.")
        return not cohort_errors and len(reports) >= 2 and passed == len(reports)

    def on_check_inputs(self) -> None:
        root = self.cohort_root
        reports, cohort_errors = check_inputs(root, self.records)
        self._write_check_report(reports, cohort_errors)

    def on_run_analysis(self) -> None:
        if self._busy:
            return
        root = self.cohort_root
        if root is None or not root.is_dir():
            self.append_log("Analysis not started: choose an existing cohort folder first.")
            return
        try:
            rows = build_metadata_rows(self.records)
        except ValueError as error:
            self.append_log(f"Analysis not started: {error}")
            return

        reports, cohort_errors = check_inputs(root, self.records)
        if not self._write_check_report(reports, cohort_errors):
            self.append_log("Analysis not started; fix the input issues above and check again.")
            return

        try:
            metadata_path = write_metadata(root, rows)
        except OSError as error:
            self.append_log(f"Could not write cohort metadata: {error}")
            return
        if not ANALYSIS_SCRIPT.is_file():
            self.append_log(f"Analysis backend script is missing: {ANALYSIS_SCRIPT}")
            return

        output_dir = root / OUTPUT_DIRNAME
        command = [
            sys.executable,
            str(ANALYSIS_SCRIPT),
            str(root),
            "--metadata",
            str(metadata_path),
        ]
        self._run_backend("analysis", command, output_dir)

    def on_run_figure(self) -> None:
        if self._busy:
            return
        root = self.cohort_root
        if root is None or not root.is_dir():
            self.append_log("Figure not started: choose an existing cohort folder first.")
            return
        try:
            rows = build_metadata_rows(self.records)
        except ValueError as error:
            self.append_log(f"Figure not started: {error}")
            return

        groups = list(dict.fromkeys(str(row["group"]) for row in rows))
        if len(groups) > 2:
            self.append_log(MORE_THAN_TWO_GROUPS_MESSAGE)
            return
        if len(groups) != 2:
            self.append_log("Figure not started: exactly two included groups are required.")
            return

        analysis_output = root / OUTPUT_DIRNAME
        metadata_path = analysis_output / METADATA_FILENAME
        if not analysis_output.is_dir():
            self.append_log(f"Figure not started: analysis output folder is missing: {analysis_output}")
            return
        if not metadata_path.is_file():
            self.append_log(
                "Figure not started: cohort_metadata.csv is missing. "
                "Run Circular W1 Analysis first."
            )
            return
        if not self._metadata_matches(metadata_path, rows):
            self.append_log(
                "Figure not started: included animals/groups differ from the saved "
                "cohort_metadata.csv. Run analysis again to refresh the outputs."
            )
            return

        control = self.control_group_var.get()
        experimental = self.experimental_group_var.get()
        if (
            not control
            or not experimental
            or control == experimental
            or control not in groups
            or experimental not in groups
        ):
            self.append_log("Figure not started: choose two distinct included groups.")
            return
        if not FIGURE_SCRIPT.is_file():
            self.append_log(f"Figure backend script is missing: {FIGURE_SCRIPT}")
            return

        figure_output = analysis_output / FIGURE_OUTPUT_DIRNAME
        command = [
            sys.executable,
            str(FIGURE_SCRIPT),
            "--input-dir",
            str(analysis_output),
            "--output-dir",
            str(figure_output),
            "--metadata",
            str(metadata_path),
            "--control-group",
            control,
            "--experimental-group",
            experimental,
        ]
        self._run_backend("figure", command, figure_output)

    @staticmethod
    def _metadata_matches(
        metadata_path: Path,
        expected_rows: list[dict[str, str | int]],
    ) -> bool:
        try:
            with metadata_path.open("r", newline="", encoding="utf-8-sig") as stream:
                saved_rows = list(csv.DictReader(stream))
        except (OSError, csv.Error):
            return False
        expected = [
            {
                "animal": str(row["animal"]),
                "group": str(row["group"]),
                "display_order": str(row["display_order"]),
            }
            for row in expected_rows
        ]
        return saved_rows == expected

    def _run_backend(
        self,
        kind: str,
        command: list[str],
        expected_output_dir: Path,
    ) -> None:
        self._process_kind = kind
        self._set_busy(True)
        self.append_log(f"Running {kind}: {subprocess.list2cmdline(command)}")
        worker = threading.Thread(
            target=self._backend_worker,
            args=(kind, command, expected_output_dir),
            daemon=True,
        )
        worker.start()

    def _backend_worker(
        self,
        kind: str,
        command: list[str],
        expected_output_dir: Path,
    ) -> None:
        try:
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
            )
            if process.stdout is not None:
                for line in process.stdout:
                    self._output_queue.put(("line", line.rstrip("\r\n")))
            return_code = process.wait()
            self._output_queue.put(
                ("finished", (kind, return_code, expected_output_dir))
            )
        except Exception:
            self._output_queue.put(("error", (kind, traceback.format_exc())))

    def _drain_output_queue(self) -> None:
        try:
            while True:
                kind, payload = self._output_queue.get_nowait()
                if kind == "line":
                    self.append_log(str(payload))
                elif kind == "finished":
                    process_kind, return_code, expected_output = payload
                    self._finish_backend(
                        str(process_kind),
                        int(return_code),
                        Path(expected_output),
                    )
                elif kind == "error":
                    process_kind, details = payload
                    self._set_busy(False)
                    self._process_kind = ""
                    self.append_log(f"Could not run {process_kind} backend:\n{details}")
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._drain_output_queue)

    def _finish_backend(
        self,
        kind: str,
        return_code: int,
        expected_output_dir: Path,
    ) -> None:
        self._set_busy(False)
        self._process_kind = ""
        if return_code != 0:
            self.append_log(f"{kind.title()} backend failed with return code {return_code}.")
            return
        if not expected_output_dir.is_dir():
            self.append_log(
                f"{kind.title()} backend returned success, but expected output folder "
                f"was not created: {expected_output_dir}"
            )
            return
        if kind == "analysis":
            self._analysis_output_dir = expected_output_dir
            self._figure_output_dir = expected_output_dir / FIGURE_OUTPUT_DIRNAME
            self.append_log("Circular W1 analysis completed successfully.")
            self.append_log(f"Results: {expected_output_dir}")
        else:
            self._figure_output_dir = expected_output_dir
            self.open_folder_button.configure(state="normal")
            self.append_log("Circular W1 figure generation completed successfully.")
            self.append_log(f"Figures: {expected_output_dir}")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for button in (self.browse_button, self.check_button, self.analysis_button):
            button.configure(state=state)
        self.path_entry.configure(state=state)
        if busy:
            self.open_folder_button.configure(state="disabled")
        self.cohort_table.state(["disabled"] if busy else ["!disabled"])
        self._refresh_group_choices()

    def open_figure_output_folder(self) -> None:
        folder = self._figure_output_dir
        if folder is None or not folder.is_dir():
            self.append_log("Figure output folder is not available yet.")
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as error:
            self.append_log(f"Could not open output folder: {error}")

    def _on_close(self) -> None:
        if self._busy:
            self.append_log("A backend is still running; wait for it to finish before closing.")
            return
        self.destroy()


def main() -> None:
    app = CircularW1Launcher()
    app.mainloop()


if __name__ == "__main__":
    main()
