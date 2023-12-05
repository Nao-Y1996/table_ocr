import sys
from typing import List

from teble_ocr.preprocessor import Cell, CellExtractor
from teble_ocr.ocr import RecognizedCharacter, Ocr
from teble_ocr.csv_exporter import export_csv

IMAGE_PATH = 'img/table.png'


def main():
    # 実行時の引数からデバッグモードを取得する
    is_debug = False
    if len(sys.argv) > 1:
        is_debug = sys.argv[1]

    # 画像からテーブルのセルを抽出する
    cells: List[Cell] = CellExtractor(IMAGE_PATH, debug=is_debug).extract()
    # OCRの実行
    results: List[RecognizedCharacter] = Ocr(IMAGE_PATH, debug=is_debug).execute()
    # CSVファイルに出力する
    export_csv(cells, results)


if __name__ == "__main__":
    main()
