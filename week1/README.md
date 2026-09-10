# Quiz 1
## Numpy 程式實作原理：
依照 ITU-R BT.601 標準，加權公式：

$$\text{Gray} = 0.299 \times R + 0.587 \times G + 0.114 \times B$$

使用 np.dot 計算灰階數值，但後來發現圖片會有多餘紅點且在後續做 Absolute Difference Heatmap 時與 OpenCV 也會有許多紅點好像是內積算出來的數值不在 0-255 的區間導致整數溢位因此又加上了 np.clip 將數值限制在 0-255，最後成功在 Absolute Difference Heatmap 與 OpenCV 成果接近。

## OpenCV 程式實作原理：
OpenCV 以 BGR 順序讀取影像，`cv2.COLOR_BGR2GRAY` 底層為 C++ 實作。

## 比較執行速度與轉換結果：
=== 執行紀錄 ===
 1. NumPy    42.682 ms | 平均灰階: 132.78
 2. OpenCV   15.997 ms | 平均灰階: 132.78
 3. NumPy    40.661 ms | 平均灰階: 132.78
 4. NumPy    45.323 ms | 平均灰階: 132.78
 5. NumPy    39.808 ms | 平均灰階: 132.78
 6. NumPy    46.286 ms | 平均灰階: 132.78
 7. NumPy    44.538 ms | 平均灰階: 132.78
 8. NumPy    47.787 ms | 平均灰階: 132.78
 9. NumPy    48.700 ms | 平均灰階: 132.78
10. NumPy    46.242 ms | 平均灰階: 132.78
11. NumPy    45.748 ms | 平均灰階: 132.78
12. NumPy    45.572 ms | 平均灰階: 132.78
13. NumPy    39.482 ms | 平均灰階: 132.78
14. NumPy    40.964 ms | 平均灰階: 132.78
15. NumPy    43.633 ms | 平均灰階: 132.78
16. NumPy    43.027 ms | 平均灰階: 132.78
17. NumPy    43.774 ms | 平均灰階: 132.78
18. NumPy    48.844 ms | 平均灰階: 132.78
19. NumPy    59.252 ms | 平均灰階: 132.78
20. NumPy    51.619 ms | 平均灰階: 132.78
21. NumPy    45.440 ms | 平均灰階: 132.78
22. NumPy    58.604 ms | 平均灰階: 132.78
23. NumPy    71.351 ms | 平均灰階: 132.78
24. NumPy    46.345 ms | 平均灰階: 132.78
25. NumPy    43.660 ms | 平均灰階: 132.78
26. NumPy    49.006 ms | 平均灰階: 132.78
27. NumPy    47.789 ms | 平均灰階: 132.78
28. OpenCV    0.654 ms | 平均灰階: 132.78
29. OpenCV    1.087 ms | 平均灰階: 132.78
30. OpenCV    1.019 ms | 平均灰階: 132.78
31. OpenCV    0.746 ms | 平均灰階: 132.78
32. OpenCV    0.733 ms | 平均灰階: 132.78
33. OpenCV    0.493 ms | 平均灰階: 132.78
34. OpenCV    1.049 ms | 平均灰階: 132.78
35. OpenCV    0.900 ms | 平均灰階: 132.78
36. OpenCV    1.221 ms | 平均灰階: 132.78
37. OpenCV    0.903 ms | 平均灰階: 132.78
38. OpenCV    0.895 ms | 平均灰階: 132.78
39. OpenCV    0.883 ms | 平均灰階: 132.78
40. OpenCV    1.015 ms | 平均灰階: 132.78
41. OpenCV    1.060 ms | 平均灰階: 132.78
42. OpenCV    0.733 ms | 平均灰階: 132.78
43. OpenCV    1.049 ms | 平均灰階: 132.78
44. OpenCV    0.687 ms | 平均灰階: 132.78
45. OpenCV    0.748 ms | 平均灰階: 132.78
46. OpenCV    0.778 ms | 平均灰階: 132.78
47. OpenCV    0.505 ms | 平均灰階: 132.78
48. OpenCV    0.595 ms | 平均灰階: 132.78
49. OpenCV    0.544 ms | 平均灰階: 132.78
50. OpenCV    0.728 ms | 平均灰階: 132.78
51. OpenCV    0.725 ms | 平均灰階: 132.78
52. OpenCV    0.666 ms | 平均灰階: 132.78
53. OpenCV    0.600 ms | 平均灰階: 132.78
54. OpenCV    0.632 ms | 平均灰階: 132.78
55. OpenCV    0.790 ms | 平均灰階: 132.78

=== 各方法平均 ===
NumPy    47.159 ms (共 26 次)
OpenCV    1.325 ms (共 29 次)

最大差異: 6 | 平均差異: 0.0039

不管是 Numpy 還是 OpenCV，執行 30 次的平均灰階誤差非常小 0.0039，代表 Numpy 有成功復刻 OpenCV 效果，但由於 OpenCV 底層使用 C++ 與 SIMD 加速，所以執行速度一定比 Numpy 快。
<img src="data/quiz1_output.png" width="600" alt="灰階比較結果">