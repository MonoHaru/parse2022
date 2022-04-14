$cmd0 = "python main.py --T0 100 --epochs 3000 --resume 'C:\Users\bed1\src\cephalometric_landmark_detection\checkpoint\0213191615\model_best.pth'"
$host.UI.RawUI.WindowTitle = $cmd0
Invoke-Expression -Command $cmd0

$cmd0 = "python main.py --T0 150 --epochs 3000 --resume 'C:\Users\bed1\src\cephalometric_landmark_detection\checkpoint\0213191615\model_best.pth'"
$host.UI.RawUI.WindowTitle = $cmd0
Invoke-Expression -Command $cmd0

$cmd0 = "python main.py --T0 200 --epochs 3000 --resume 'C:\Users\bed1\src\cephalometric_landmark_detection\checkpoint\0213191615\model_best.pth'"
$host.UI.RawUI.WindowTitle = $cmd0
Invoke-Expression -Command $cmd0

$cmd0 = "python main.py --T0 250 --epochs 3000 --resume 'C:\Users\bed1\src\cephalometric_landmark_detection\checkpoint\0213191615\model_best.pth'"
$host.UI.RawUI.WindowTitle = $cmd0
Invoke-Expression -Command $cmd0