from .ocr import RecognizedCharacter
from typing import List, Tuple
import cv2
import numpy as np


class Cell:
    """テーブルのセルを表すクラス

    Attributes
    ----------
    centroid : Tuple[int, int]
        セルの重心座標
    contour : np.ndarray
        セルの輪郭
    """

    def __init__(self, centroid: Tuple[int, int], contour: np.ndarray):
        self._recognized_list: List[RecognizedCharacter] = []
        self.centroid = centroid
        self.contour = contour
        self.__value = ""
        self.__merged = False

    def add_value(self, recognized: RecognizedCharacter):
        """このセルに文字認識結果を追加する.
        """
        self._recognized_list.append(recognized)

    def merge_value(self) -> None:
        """このセルに含まれる文字認識結果の一覧から文字を結合してセルの値を決定する.

        認識結果の重心のy座標が小さいものから順に結合される.
        y座標の差が10以下の場合、x座標が小さいものから順に結合される.
        """
        if self.__merged:
            return
        if len(self._recognized_list) == 0:
            self.__merged = True
            return
        self._recognized_list.sort(key=lambda x: x.sort_key())
        self.__value = "".join([x.string for x in self._recognized_list])
        self.__merged = True

    def get_value(self) -> str:
        """このセルの値を返す.

        Raises
        ------
        ValueError
            セルの値が決定されていない場合
        """
        if not self.__merged:
            raise ValueError("values in cell are not merged. call merge_value() first.")
        return self.__value

    def vector_to(self, another: "Cell") -> float:
        """このセルの重心から別のセルの重心へのベクトルを返す.

        Parameters
        ----------
        another : Cell
            別のセル
        """
        return np.array(another.centroid) - np.array(self.centroid)

    def sort_key(self):
        """ソートのためのキー関数

        y座標でソートする.
        ただし、y座標の差が20px未満の場合、x座標を優先してソートするため、y座標を20pxで割った値とx座標を組み合わせたタプルを返す.
        """
        return self.centroid[1] // 20, self.centroid[0]


class CellExtractor:

    def __init__(self, image_path: str, debug: bool = False):
        self.image_path = image_path
        self.debug = debug
        self.image = cv2.imread(self.image_path)

    def extract(self) -> List[Cell]:

        # BGR -> グレースケール
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # ノイズ除去
        smooth_img = cv2.blur(img_gray, (5, 5))
        if self.debug:
            cv2.imwrite('img/output/1_smooth.png', smooth_img)

        # エッジ抽出 (Canny)
        edges_img = cv2.Canny(smooth_img, 1, 100, apertureSize=3)
        if self.debug:
            cv2.imwrite('img/output/2_edges.png', edges_img)

        # 膨張処理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilates = cv2.dilate(edges_img, kernel)
        if self.debug:
            cv2.imwrite('img/output/4_dilates.png', dilates)

        # 2値化画像から輪郭抽出
        contours, hierarchy = cv2.findContours(dilates, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # 輪郭を描画
        if self.debug:
            contours_img = cv2.drawContours(dilates, contours, 10, (0, 255, 0), 1)
            cv2.imwrite('img/output/5_contours.png', contours_img)

        # 輪郭ごとの処理
        cells: List[Cell] = []
        for i, contour in enumerate(contours):
            # 輪郭の面積を求める
            area = cv2.contourArea(contour, True)
            if area < 3000:
                continue  # 面積が小さいものは除く

            # 輪郭の重心を求める
            m = cv2.moments(contour)
            centroid = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))

            # セルを作成
            cell: Cell = Cell(centroid, contour)
            cells.append(cell)

            # 画像に当該の枠線を追加
            if self.debug:
                # 輪郭を描画
                cv2.drawContours(self.image, contours, i, (0, 255, 0), 2)
                # 重心を画像に追加
                cv2.circle(self.image, centroid, radius=5, color=(0, 255, 0), thickness=5)
        if self.debug:
            cv2.imwrite('img/output/6_extracted.png', self.image)

        return cells
