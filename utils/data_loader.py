import pandas as pd
from typing import List, Dict, Any


def load_csv(filepath: str) -> List[Dict[str, Any]]:
    """
    Loads a CSV file and returns a list of dicts.
    One dict per row — same pattern as Repo 2.
    dropna removes completely empty rows defensively.
    """
    df = pd.read_csv(filepath)
    df = df.dropna(how="all")
    return df.to_dict(orient="records")


def load_excel(filepath: str, sheet_name: str = "Sheet1") -> List[Dict[str, Any]]:
    """
    Loads an Excel sheet and returns a list of dicts.
    sheet_name parameter allows multi-sheet support.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df = df.dropna(how="all")
    return df.to_dict(orient="records")


def load_ground_truth(filepath: str, sheet_name: str = "ground_truth") -> Dict[str, List[str]]:
    """
    Loads ground truth facts from Excel for hallucination detection.
    Returns a dict where key is topic and value is list of facts.

    Expected columns: topic, fact
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df = df.dropna(how="all")

    ground_truth = {}
    for _, row in df.iterrows():
        topic = str(row["topic"]).strip()
        fact = str(row["fact"]).strip()
        if topic not in ground_truth:
            ground_truth[topic] = []
        ground_truth[topic].append(fact)

    return ground_truth