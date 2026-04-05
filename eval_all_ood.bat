@echo off
REM Script to evaluate OOD detection on all datasets
REM Usage: eval_all_ood.bat <model_path>
REM Example: eval_all_ood.bat save\students\models\S~resnet8x4_T~resnet32x4_cifar100_pkt_r~1.0_a~1.0_b~1.0_0\resnet8x4_best.pth

setlocal enabledelayedexpansion

set MODEL_PATH=%1

if "%MODEL_PATH%"=="" (
    echo Usage: %0 ^<model_path^>
    echo Example: %0 save\students\models\S~resnet8x4_T~resnet32x4_cifar100_pkt_r~1.0_a~1.0_b~1.0_0\resnet8x4_best.pth
    exit /b 1
)

set MODEL_S=resnet8x4
set BATCH_SIZE=128
set NUM_WORKERS=2

REM Extract model directory and create output file path
for %%F in ("%MODEL_PATH%") do set MODEL_DIR=%%~dpF
set OUTPUT_FILE=%MODEL_DIR%ood_results\all_ood_results.txt

REM Create ood_results directory if it doesn't exist
if not exist "%MODEL_DIR%ood_results" mkdir "%MODEL_DIR%ood_results"

REM Extract best accuracy from test_best_metrics.json
set BEST_ACC=N/A
set METRICS_FILE=%MODEL_DIR%test_best_metrics.json
if exist "%METRICS_FILE%" (
    findstr /c:"test_acc" "%METRICS_FILE%" | findstr /v "top5" > "%MODEL_DIR%ood_results\temp_acc.txt"
    for /f "usebackq tokens=2 delims=:," %%a in ("%MODEL_DIR%ood_results\temp_acc.txt") do set BEST_ACC=%%a
    del "%MODEL_DIR%ood_results\temp_acc.txt"
)

REM Create header in output file
echo Model: %MODEL_S% > "%OUTPUT_FILE%"
echo Best Accuracy:!BEST_ACC! >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"
echo OOD Detection Results: >> "%OUTPUT_FILE%"
echo ======================================== >> "%OUTPUT_FILE%"

echo [1/7] Evaluating on CIFAR-10...
python eval_ood.py --model_path "%MODEL_PATH%" --model_s %MODEL_S% --ood_dataset cifar10 --batch_size %BATCH_SIZE% --num_workers %NUM_WORKERS% --quiet >> "%OUTPUT_FILE%" 2>&1

echo [2/7] Evaluating on Tiny-ImageNet-200...
python eval_ood.py --model_path "%MODEL_PATH%" --model_s %MODEL_S% --ood_dataset tiny-imagenet --batch_size %BATCH_SIZE% --num_workers %NUM_WORKERS% --quiet >> "%OUTPUT_FILE%" 2>&1

echo [3/7] Evaluating on Human Detection Dataset...
python eval_ood.py --model_path "%MODEL_PATH%" --model_s %MODEL_S% --ood_dataset human-detection --batch_size %BATCH_SIZE% --num_workers %NUM_WORKERS% --quiet >> "%OUTPUT_FILE%" 2>&1

echo [4/7] Evaluating on DTD...
python eval_ood.py --model_path "%MODEL_PATH%" --model_s %MODEL_S% --ood_dataset dtd --batch_size %BATCH_SIZE% --num_workers %NUM_WORKERS% --quiet >> "%OUTPUT_FILE%" 2>&1

echo [5/7] Evaluating on SVHN...
python eval_ood.py --model_path "%MODEL_PATH%" --model_s %MODEL_S% --ood_dataset svhn --batch_size %BATCH_SIZE% --num_workers %NUM_WORKERS% --quiet >> "%OUTPUT_FILE%" 2>&1

echo [6/7] Evaluating on Places...
python eval_ood.py --model_path "%MODEL_PATH%" --model_s %MODEL_S% --ood_dataset places --batch_size %BATCH_SIZE% --num_workers %NUM_WORKERS% --quiet >> "%OUTPUT_FILE%" 2>&1

echo [7/7] Evaluating on LSUN...
python eval_ood.py --model_path "%MODEL_PATH%" --model_s %MODEL_S% --ood_dataset lsun --batch_size %BATCH_SIZE% --num_workers %NUM_WORKERS% --quiet >> "%OUTPUT_FILE%" 2>&1

echo.
echo ================================================================================
echo All evaluations completed!
echo Results saved to: %OUTPUT_FILE%
echo ================================================================================

endlocal
