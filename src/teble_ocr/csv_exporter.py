import csv
import numpy as np
from typing import List

from .preprocessor import Cell
from .ocr import RecognizedCharacter


def cos_sim(v1: np.array, v2: np.array):
    """コサイン類似度を計算する"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def export_csv(cells: List[Cell], results: List[RecognizedCharacter]):
    """Export data to CSV file."""
    for cell in cells:
        for result in results:
            # 認識結果がセルの中に含まれている時、セルに認識結果を追加
            if result.within_contour(cell.contour):
                cell.add_value(result)
        # セルに追加した認識結果をマージして、セルの値を決定する
        cell.merge_value()

    # セルをソートする
    cells.sort(key=lambda x: x.sort_key())

    row: List[str] = []
    row_standard_vector: np.array = np.array([1, 0])  # 同じ行とみなすための基準ベクトル. 初期値はx軸の正の向き
    column_standard_vector: np.array = np.array([0, 1])  # 改行の基準ベクトル. 初期値はy軸の正の向き
    with open("output.csv", mode="w") as f:
        writer = csv.writer(f)
        for i in range(len(cells)):

            # 最初のセルの場合、行に追加する
            if i == 0:
                row.append(cells[i].get_value())
                continue

            # 1つ前のセルからのベクトル
            v: float = cells[i - 1].vector_to(cells[i])
            # 1つ前のセルからのベクトルが、行の基準ベクトルの向きと近い場合、同じ行にあるとみなし、行に追加する
            if cos_sim(v, row_standard_vector) > cos_sim(v, column_standard_vector):
                row.append(cells[i].get_value())
                # 行の基準ベクトルを更新
                row_standard_vector = v
            # 1つ前のセルからのベクトルが、改行の基準ベクトルの向きと近い場合、改行して新しい行に追加する
            else:
                # 行を書き込む
                writer.writerow(row)
                # 新しい行に値を追加する
                row = [cells[i].get_value()]
                # 列の基準ベクトルを更新
                column_standard_vector = v

            # 最後のセルの場合、行に書き込んで終了する
            if i == len(cells) - 1:
                writer.writerow(row)
