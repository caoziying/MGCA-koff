from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ColdStartStrategy(ABC):
    """Base class for cold-start entity splitting."""

    @abstractmethod
    def get_unique_entities(self, rows: list[tuple[str, str, float]]) -> list[str]:
        pass

    @abstractmethod
    def get_entity_from_row(self, row: tuple[str, str, float]) -> str:
        pass

    @abstractmethod
    def get_rows_by_entity(self, rows: list[tuple[str, str, float]], entity: str) -> list[int]:
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        pass


class DrugColdStartStrategy(ColdStartStrategy):
    """Drug cold start: split by SMILES."""

    def get_unique_entities(self, rows: list[tuple[str, str, float]]) -> list[str]:
        unique_smiles = set()
        for _, smiles, _ in rows:
            unique_smiles.add(smiles.strip())
        return list(unique_smiles)

    def get_entity_from_row(self, row: tuple[str, str, float]) -> str:
        return row[1].strip()

    def get_rows_by_entity(self, rows: list[tuple[str, str, float]], entity: str) -> list[int]:
        indices = []
        for i, (_, smiles, _) in enumerate(rows):
            if smiles.strip() == entity:
                indices.append(i)
        return indices

    def get_strategy_name(self) -> str:
        return "Drug Cold Start"


class ProteinColdStartStrategy(ColdStartStrategy):
    """Protein cold start: split by FASTA sequence."""

    def get_unique_entities(self, rows: list[tuple[str, str, float]]) -> list[str]:
        unique_fastas = set()
        for fasta, _, _ in rows:
            unique_fastas.add(fasta.strip())
        return list(unique_fastas)

    def get_entity_from_row(self, row: tuple[str, str, float]) -> str:
        return row[0].strip()

    def get_rows_by_entity(self, rows: list[tuple[str, str, float]], entity: str) -> list[int]:
        indices = []
        for i, (fasta, _, _) in enumerate(rows):
            if fasta.strip() == entity:
                indices.append(i)
        return indices

    def get_strategy_name(self) -> str:
        return "Protein Cold Start"


class DrugProteinPairColdStartStrategy(ColdStartStrategy):
    """Drug-protein pair cold start: split by FASTA and SMILES pair."""

    def get_unique_entities(self, rows: list[tuple[str, str, float]]) -> list[str]:
        unique_pairs = set()
        for fasta, smiles, _ in rows:
            pair = f"{fasta.strip()}|{smiles.strip()}"
            unique_pairs.add(pair)
        return list(unique_pairs)

    def get_entity_from_row(self, row: tuple[str, str, float]) -> str:
        return f"{row[0].strip()}|{row[1].strip()}"

    def get_rows_by_entity(self, rows: list[tuple[str, str, float]], entity: str) -> list[int]:
        indices = []
        for i, (fasta, smiles, _) in enumerate(rows):
            pair = f"{fasta.strip()}|{smiles.strip()}"
            if pair == entity:
                indices.append(i)
        return indices

    def get_strategy_name(self) -> str:
        return "Drug-Protein Pair Cold Start"


class ColdStartCrossValidator:
    """Cold-start cross-validator shared by all split strategies."""

    def __init__(self, strategy: ColdStartStrategy, n_splits: int = 5, seed: int = 43):
        self.strategy = strategy
        self.n_splits = n_splits
        self.seed = seed

    def make_cold_start_folds(self, rows: list[tuple[str, str, float]]) -> list[tuple[np.ndarray, np.ndarray]]:
        unique_entities = self.strategy.get_unique_entities(rows)
        n_entities = len(unique_entities)

        print(f"Found {n_entities} unique {self.strategy.get_strategy_name().lower()} entities")

        rng = np.random.RandomState(self.seed)
        entity_indices = np.arange(n_entities)
        rng.shuffle(entity_indices)

        entity_folds = np.array_split(entity_indices, self.n_splits)

        folds = []
        for i in range(self.n_splits):
            test_entities = [unique_entities[idx] for idx in entity_folds[i]]
            train_entities = [
                unique_entities[idx]
                for j in range(self.n_splits)
                for idx in entity_folds[j]
                if j != i
            ]

            test_indices = []
            train_indices = []

            for entity in test_entities:
                test_indices.extend(self.strategy.get_rows_by_entity(rows, entity))

            for entity in train_entities:
                train_indices.extend(self.strategy.get_rows_by_entity(rows, entity))

            folds.append((np.array(train_indices), np.array(test_indices)))

            print(f"  Fold {i + 1}: train {len(train_indices)} samples, test {len(test_indices)} samples")
            print(f"    train entities: {len(train_entities)}, test entities: {len(test_entities)}")

        return folds

    def validate_cold_start(
        self,
        folds: list[tuple[np.ndarray, np.ndarray]],
        rows: list[tuple[str, str, float]],
    ) -> dict[str, Any]:
        validation_results: dict[str, Any] = {
            "valid": True,
            "overlap_issues": [],
            "cold_start_issues": [],
            "statistics": {},
        }

        for fold_id, (train_idx, test_idx) in enumerate(folds):
            train_set = set(train_idx)
            test_set = set(test_idx)
            overlap = train_set.intersection(test_set)

            if overlap:
                validation_results["valid"] = False
                validation_results["overlap_issues"].append(f"Fold {fold_id}: {len(overlap)} overlapping indices")

            train_rows = [rows[i] for i in train_idx]
            test_rows = [rows[i] for i in test_idx]

            train_entities = set(self.strategy.get_entity_from_row(row) for row in train_rows)
            test_entities = set(self.strategy.get_entity_from_row(row) for row in test_rows)

            cold_start_entities = test_entities.intersection(train_entities)
            if cold_start_entities:
                validation_results["valid"] = False
                validation_results["cold_start_issues"].append(
                    f"Fold {fold_id}: {len(cold_start_entities)} entities appear in both train and test"
                )

            validation_results["statistics"][f"fold_{fold_id}"] = {
                "train_samples": len(train_idx),
                "test_samples": len(test_idx),
                "train_entities": len(train_entities),
                "test_entities": len(test_entities),
                "cold_start_entities": len(cold_start_entities),
            }

        return validation_results


def get_cold_start_strategy(mode: str) -> ColdStartStrategy:
    if mode == "drug":
        return DrugColdStartStrategy()
    if mode == "target":
        return ProteinColdStartStrategy()
    if mode == "pair":
        return DrugProteinPairColdStartStrategy()
    raise ValueError(f"Unknown cold-start mode: {mode}")
