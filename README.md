# Background Oriented Schlieren (BOS) 法の実験

[AirVisualizer](https://github.com/kambara/air-visualizer) を開発する前に書いた検証用のコードです。

撮影対象の背景に置く画像は[kambara/air-visualizer-background](https://github.com/kambara/air-visualizer-background) を使用。

## インストール

```
pipenv install --dev
```

## 実行

撮影した動画ファイルのパスを引数で渡して実行

```
pipenv shell
python main.py <path/to/input.mp4>
```

結果を動画ファイルに書き出す場合

```
python main.py --output <path/to/output.mp4> <path/to/input.mp4>
```

## 参考

- [Schlieren imaging without mirrors - Python and OpenCV, background oriented, synthetic - YouTube](https://www.youtube.com/watch?v=J0Ix4Wa3CJk)
- [OpenCV: ArUco Marker Detection](https://docs.opencv.org/3.4/d9/d6a/group__aruco.html)
