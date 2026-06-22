import csv
import os
import zipfile
import xml.etree.ElementTree as ET


def add_spreadsheet_file(
    file_path,
    child_splitter,
    get_collection_by_language,
    detect_language,
    file_hash
):
    filename = os.path.basename(file_path)
    source = os.path.abspath(file_path)
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".csv":
        text = _read_csv(file_path)
        source_type = "csv"
    elif extension == ".xlsx":
        text = _read_xlsx(file_path)
        source_type = "xlsx"
    else:
        raise ValueError(f"Unsupported spreadsheet type: {extension}")

    chunks = child_splitter.split_text(text)
    batches = {}

    for i, chunk in enumerate(chunks):
        language = detect_language(chunk)
        target_col = get_collection_by_language(language)
        batch = batches.setdefault(
            language,
            {"collection": target_col, "documents": [], "ids": [], "metadatas": []}
        )
        batch["documents"].append(chunk)
        batch["ids"].append(f"{filename}_sheet_chunk_{i}")
        batch["metadatas"].append({
            "document": filename,
            "source": source,
            "file_hash": file_hash,
            "content_type": "text",
            "source_type": source_type,
            "language": language,
            "chunk_index": i
        })

    for batch in batches.values():
        batch["collection"].add(
            documents=batch["documents"],
            ids=batch["ids"],
            metadatas=batch["metadatas"]
        )


def _read_csv(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            rows.append(" | ".join(cell.strip() for cell in row if cell.strip()))
    return "\n".join(row for row in rows if row)


def _read_xlsx(file_path):
    with zipfile.ZipFile(file_path) as workbook:
        shared_strings = _read_shared_strings(workbook)
        sheet_names = _read_sheet_names(workbook)
        sheet_paths = sorted(
            name for name in workbook.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )

        parts = []
        for index, sheet_path in enumerate(sheet_paths):
            sheet_name = sheet_names[index] if index < len(sheet_names) else sheet_path
            rows = _read_xlsx_sheet(workbook, sheet_path, shared_strings)
            if rows:
                parts.append(f"Sheet: {sheet_name}\n" + "\n".join(rows))

        return "\n\n".join(parts)


def _read_shared_strings(workbook):
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []

    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings = []

    for item in root.findall("x:si", ns):
        text_parts = [
            node.text or ""
            for node in item.findall(".//x:t", ns)
        ]
        strings.append("".join(text_parts))

    return strings


def _read_sheet_names(workbook):
    if "xl/workbook.xml" not in workbook.namelist():
        return []

    root = ET.fromstring(workbook.read("xl/workbook.xml"))
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    return [
        sheet.attrib.get("name", "")
        for sheet in root.findall(".//x:sheet", ns)
    ]


def _read_xlsx_sheet(workbook, sheet_path, shared_strings):
    root = ET.fromstring(workbook.read(sheet_path))
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows = []

    for row in root.findall(".//x:row", ns):
        values = []
        for cell in row.findall("x:c", ns):
            value = _read_cell_value(cell, shared_strings, ns)
            if value:
                values.append(value)
        if values:
            rows.append(" | ".join(values))

    return rows


def _read_cell_value(cell, shared_strings, ns):
    value_node = cell.find("x:v", ns)
    if value_node is None or value_node.text is None:
        inline_node = cell.find(".//x:t", ns)
        return inline_node.text.strip() if inline_node is not None and inline_node.text else ""

    value = value_node.text.strip()
    if cell.attrib.get("t") == "s":
        index = int(value)
        return shared_strings[index] if index < len(shared_strings) else ""

    return value
