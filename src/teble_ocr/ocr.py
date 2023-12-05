import cv2
import easyocr
import numpy as np
from typing import List, Tuple
from PIL import ImageFont, ImageDraw, Image


class RecognizedCharacter:
    """
    認識された文字列を表すクラス

    Attributes
    ----------
    string : str
        認識された文字列
    confidence : float
        認識された文字列の確信度
    """

    def __init__(self, top_left: Tuple[int, int], top_right: Tuple[int, int],
                 bottom_right: Tuple[int, int], bottom_left: Tuple[int, int],
                 string: str, confidence: float):
        """
        Parameters
        ----------
        top_left : Tuple[int, int]
            文字列の左上の座標
        top_right : Tuple[int, int]
            文字列の右上の座標
        bottom_right : Tuple[int, int]
            文字列の右下の座標
        bottom_left : Tuple[int, int]
            文字列の左下の座標
        string : str
            認識された文字列
        confidence : float
            認識された文字列の確信度
        """
        self.__top_left = top_left
        self.__top_right = top_right
        self.__bottom_right = bottom_right
        self.__bottom_left = bottom_left
        self.centroid = self.get_centroid()
        self.string = string
        self.confidence = confidence

    def get_centroid(self) -> Tuple[int, int]:
        """
        文字列の重心座標を取得する

        Returns
        -------
        Tuple[int, int]
            文字列の中心座標
        """
        return (self.__top_left[0] + self.__bottom_right[0]) // 2, (self.__top_left[1] + self.__bottom_right[1]) // 2

    def within_contour(self, contour: np.ndarray) -> bool:
        """
        認識された文字列が与えられた輪郭の中に含まれるかどうかを判定する

        Parameters
        ----------
        contour : List[List[int]]
            輪郭を表すリスト

        Returns
        -------
        bool
            この認識された文字列が与えられた輪郭の中に含まれるかどうか
        """
        return cv2.pointPolygonTest(contour, self.get_centroid(), False) >= 0

    def sort_key(self):
        """ソートのためのキー関数

        y座標でソートする.
        ただし、y座標の差が10px未満の場合、x座標を優先してソートするため、y座標を10pxで割った値とx座標を組み合わせたタプルを返す.
        """
        return round(self.centroid[1] / 10), self.centroid[0]


class Ocr:
    """画像から文字列を認識するクラス."""

    def __init__(self, image_path: str, debug: bool = False):
        """
        Parameters
        ----------
        :param image_path: 画像パス
        :param debug: デバッグモード
        """
        self.image_path = image_path
        self.debug = debug
        self.__reader = easyocr.Reader(['ja', 'en'])
        self.image = None
        if self.debug:
            self.image = cv2.imread(self.image_path)

    def __put_result(self, img: np.ndarray, text: str, point: tuple[int, int], size: int,
                     color: tuple[int, int, int]) -> np.ndarray:
        """
        画像中に日本語を記入する関数.
        """
        font = ImageFont.truetype('ヒラギノ角ゴシック W8.ttc', size=size)

        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text(point, text, fill=color, font=font)
        return np.array(img_pil)

    def execute(self) -> List[RecognizedCharacter]:
        img = None
        ocr_result = self.__reader.readtext(self.image_path)
        results: List[RecognizedCharacter] = []
        for element in ocr_result:
            rectangle: list = element[0]
            top_left: tuple = (round(rectangle[0][0]), round(rectangle[0][1]))
            top_right: tuple = (round(rectangle[1][0]), round(rectangle[1][1]))
            bottom_right: tuple = (round(rectangle[2][0]), round(rectangle[2][1]))
            bottom_left: tuple = (round(rectangle[3][0]), round(rectangle[3][1]))
            result: RecognizedCharacter = RecognizedCharacter(
                top_left, top_right, bottom_right, bottom_left, element[1], element[2])
            results.append(result)

            if self.debug:
                # 認識した文字を青で囲む
                cv2.rectangle(self.image, top_left, bottom_right, color=(255, 0, 0), thickness=5)
                # 認識した文字の中心を青で表示
                cv2.circle(self.image, result.get_centroid(), radius=5, color=(255, 0, 0), thickness=5)
                # 認識した文字と確信度を赤で表示
                self.image = self.__put_result(self.image, f"{result.string}", top_left, 30, (0, 0, 255))
        if self.debug:
            cv2.imwrite('img/output/ocr_result.png', self.image)

        return results
