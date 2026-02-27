\# Baseline: YOLOv8n + OpenVINO для детекции человека на видео



Этот baseline оценивает качество и производительность детекции человека на трёх видео‑пресетах с разным освещением с помощью модели \*\*YOLOv8n\*\*, конвертированной в формат \*\*OpenVINO\*\* и запущенной на \*\*CPU\*\*.



\## Датасет и сценарии



Используются три видео из камеры наблюдения/смартфона:



\- `videos/backlight.mp4` — контровой свет (подсветка сзади, сложные условия для детекции силуэта).

\- `videos/fulllight.mp4` — равномерный комнатный свет без естественного освещения.

\- `videos/lowlight.mp4` — пониженная освещённость (сложный режим для детектора и трекинга).



Для каждого видео считаются:



\- производительность инференса (FPS по кадру);

\- факт присутствия человека в кадре (есть/нет);

\- статистика потерь цели (TTL) — длины серий подряд идущих кадров, где человек не обнаружен.



\## Модель и инференс



1\. Загрузка предобученной модели `YOLOv8n` из `ultralytics` и экспорт в OpenVINO:



```python

from ultralytics import YOLO

import openvino as ov



model = YOLO('yolov8n.pt')

model.export(format='openvino', imgsz=640)



core = ov.Core()

model\_path = 'yolov8n\_openvino\_model/yolov8n.xml'

compiled\_model = core.compile\_model(model\_path, 'CPU')

input\_layer = compiled\_model.input(0)

output\_layer = compiled\_model.output(0)



