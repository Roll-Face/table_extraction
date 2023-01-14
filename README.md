## Architecture

1. Table detection: Using SOTA detectron2
2. Table Line: Using architecture Unet + rule base
3. OCR: Using SOTA easyocr

## Train

### Prepare dataset:

Data is private not public, you can learn on internet about tabular data, You can label data by labelme ([wkentaro/labelme: Image Polygonal Annotation with Python (polygon, rectangle, circle, line, point and image-level flag annotation). (github.com)](https://github.com/wkentaro/labelme))

`Refers datasets`:
  - https://www.icst.pku.edu.cn/cpdp/sjzy/index.htm
  - https://paperswithcode.com/dataset/icdar-2013
  - https://doc-analysis.github.io/tablebank-page/


### Training

`Config params: file base_config.yaml`

```python
bash sh scripts/train.sh 
```

## Demo Table Line

```python
bash sh scripts/infer_table_line.sh 
```

`Step 1: Table detection`

`Step 2: Table Line`

`Input:`

![1671954396105](image/README/1671954396105.png)

`Output:`

![1671954423770](image/README/1671954423770.png)

## Table OCR

`Step 1: Table detection `

`Step 2: Table line`

`Step 3: Crop image according line`

`Step 4: OCR`

`Step 5: Save file csv/excel`

```
sh scripts/infer_table_ocr.sh
```

`Input: ./datasets/demo_examples/demo2.png`

![1671956437092](image/README/1671956437092.png)

`Output: ./results/demo.csv`

![1671956507888](image/README/1671956507888.png)

## References
1. [nanonets-blog](https://nanonets.com/blog/table-extraction-deep-learning/#tablenet?&utm_source=nanonets.com/blog/&utm_medium=blog&utm_content=Table%20Detection,%20Information%20Extraction%20and%20Structuring%20using%20Deep%20Learning)
2. [table-detection-structure-recognition](https://github.com/abdoelsayed2016/table-detection-structure-recognition)
3. [table-transformer](https://github.com/microsoft/table-transformer)
4. [TableNet: Deep Learning Model for End-to-end Table Detection and Tabular Data Extraction from Scanned Document Images](https://www.researchgate.net/publication/337242893_TableNet_Deep_Learning_Model_for_End-to-end_Table_Detection_and_Tabular_Data_Extraction_from_Scanned_Document_Images)
