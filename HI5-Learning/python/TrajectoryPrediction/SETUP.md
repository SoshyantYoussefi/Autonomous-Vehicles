# Trajectory Prediction på Windows

## Verifierad installation på denna dator

Kursdatan har kopierats från `C:\Users\Soshy\Downloads\datasets\datasets` till
notebookens `datasets`-mapp: 8 925 scenfiler, 1 318 873 733 byte, 279 batcher.
`pip check`, alla tre modellernas provprediktioner och notebookens kodceller
(utom tidsmätningarnas repetitionsloopar) har godkänts med riktig data.
Utvärderingen testades på första batchen; ingen fullständig labbutvärdering har körts.

Miljön ligger i projektrotens `.venv-hi5` och använder Python 3.12 med CPU-versionen
av PyTorch 2.4.1. Använd inte den gamla `env`-miljön (Python 3.14) för denna labb.
Paket i `node_modules` är JavaScript-paket och ersätter inte Python-paketen.

## Starta

Kör från projektroten i PowerShell:

```powershell
.\start-trajectory-lab.ps1
```

Om PowerShell blockerar skriptet:

```powershell
powershell -ExecutionPolicy Bypass -File .\start-trajectory-lab.ps1
```

Välj **Python (HI5 Trajectory)** i Jupyter. I VS Code: välj kernel via
**Select Kernel → Python Environments → Enter interpreter path** och ange
`C:\Users\Soshy\Desktop\Autonomous-Vehicles\.venv-hi5\Scripts\python.exe`.
Kör sedan notebooken uppifrån. Första importen kan ta en stund.
För interaktiv zoom kan du använda `%matplotlib widget` i backend-cellen.

## Kursdata från Lisam

Hämta **Course documents/hi5_trajectorydata.zip** enligt exercise5.pdf.
Det kräver ditt kurskonto; rådata från SIND-projektet är inte samma förbehandlade data.
Om ZIP-filen ligger i Hämtade filer, kör från projektroten:

```powershell
.\.venv-hi5\Scripts\python.exe HI5-Learning\python\TrajectoryPrediction\import_course_data.py "$env:USERPROFILE\Downloads\hi5_trajectorydata.zip"
```

Importen hittar `sinD` även om ZIP-filen har en extra överordnad mapp.
Resultatet ska vara `HI5-Learning/python/TrajectoryPrediction/datasets/sinD/test/`.
Den skriver inte över befintliga data. Cirka 1,3 GB behövs efter uppackning.
För data på annan plats kan miljövariabeln `HI5_DATASET_PATH` sättas till mappen
som innehåller `sinD`. Alla tre förtränade modeller finns redan i projektet.

## Verifiera

```powershell
.\.venv-hi5\Scripts\python.exe -m pip check
.\.venv-hi5\Scripts\python.exe HI5-Learning\python\TrajectoryPrediction\verify_setup.py
```

Verifieringen testar importer, NumPy/PyTorch, grafoperationer och alla checkpoints.
När kursdata finns körs notebookens kodceller, figurer och utvärdering på första
batchen. Tidsmätningarnas repetitionsloopar hoppas över. Figurer sparas i
`setup-check`. Exitkod 2 betyder att paket/modeller fungerar men kursdata saknas.
Notebookens `full_eval=False` är kvar för snabb provkörning; välj `True` när du
själv ska göra hela utvärderingen för labben.

## Återskapa miljön

Med Python 3.12 installerad:

```powershell
.\setup-trajectory-lab.ps1
# Eller ange Python 3.12 explicit:
.\setup-trajectory-lab.ps1 -Python312 'C:\sökväg\python.exe'
```

Windows-paketen specificeras i `requirements-windows.txt`.
En exakt versionslista för den installerade miljön finns i `requirements-windows.lock.txt`.
Installation följer [PyTorchs CPU-instruktioner](https://pytorch.org/get-started/previous-versions/)
och använder matchande [PyG Windows-hjul för PyTorch 2.4](https://data.pyg.org/whl/torch-2.4.0+cpu.html).
Miljö, data och tillfälliga verifieringsfiler ignoreras av Git.
