# rp4objectdetection

## Vorgehen
Jedes Objekdetektionsmodell hält in einer jeweiligen 
measurement\_{modelname}.json die Ergebnisse
für Inferenzzeiten, Pre- und Postprocessing und detektieren Objekte 
(bounding boxes koordinaten, klassifizierung) für jedes Bild im datasets Ordner fest,
um später separat ausgewertet zu werden.

## Ordner eval
Im Ordner eval befindet sich ein skript namens map.py das alle
bestehenden measurement\_{modelname}.json Dateien im gesamten Projekt durchscannt
und für jedes modell entsprechend 'mean average precision' auswertet.
Diese Ergebnisse werden in eine 'mapResult.json' gespeichert.

Zusätzlich nimmt showResults.py sowohl alle measurement\_{modelname}.json
als auch die 'mapResult.json' entgegen und erstellt daraus matplotlib Diagramme.

## damo-yolo
Um für damoyolo die measurements Datei zu erstellen,
wurden die DAMO-YOLO Installationsanweisungen unter 
https://github.com/tinyvision/DAMO-YOLO befolgt
und anschließend unter dem tools Ordner die Datei demo.py kopiert und
abgeändert. Die neue Datei namens tb\_damoyolo\_detect.py nimmt die coco-2017 validation Bilder im übergeordneten datasets Ordner, lässt die Objektdektion darüber laufen
und erstellt eine measurement\_damoyolo.json.

## ncnn-cpp
Im ncnn-cpp Ordner wird die Objektdetektion für yolo-fastestv2, nanodet und yolov8n
unter dem c++ und dem ncnn framework getestet. Der Code entstammt von
- https://github.com/Qengineering/YoloV8-ncnn-Raspberry-Pi-4 
- https://github.com/Qengineering/YoloFastestV2-ncnn-Raspberry-Pi-4
- https://github.com/Qengineering/NanoDet-ncnn-Raspberry-Pi-4
und wurde zusammengeführt, ein wenig zusammengefasst und so modifiziert,
dass das kompilierte Programm die Bilder im datasets Ordner entgegen nimmt,
und eine "measurement\_ncnn\_cpp\_yoloFastestV2.json" eine 
"measurement\_ncnn\_cpp\_yolov8n.json" und eine "measurement\_ncnn\_cpp\_nanodet.json"
erstellt.

Zum Kompilieren ./compile.sh ausführen und mit ./a.out ausführen.

## picodet
Der Code für picodet kommt von
https://github.com/hpc203/picodet-onnxruntime
und die main.py wurde nach den obigen Anforderung modifiziert.
Standardmäßig wird das Modell picodet\_s\_319\_coco.onnx ausgewertet.

# yolov8

Die Installation benötigte leichte Änderungen in der requirements.txt für torch und
torchvision. Ansonste konnte die ultralytics gut mit pip installiert werden.

Im yolov8 Modell ist tb\_yolo\_detect.py für die Ergebnisse der Objekterkennung
zuständig und schreibt eine entsprechende measurement Datei.

In der benchmark.py wurden ursprünglich viele unterschiedliche modelle 
in einem array ptmodelnames angegeben und mit der ultralytics benchmark funktion
getestet. Die Ergebnisse sind in benchmark.log festgehalten.

