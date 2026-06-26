@ECHO OFF
SETLOCAL
ECHO Setting up environment and running Zillow analysis...

REM Change directory to the repository root
cd /d "%~dp0\..\.."

REM 1. Create virtual environment if it doesn't exist
IF NOT EXIST venv (
    ECHO Creating virtual environment...
    python -m venv venv
)

REM 2. Activate the virtual environment for this script session
CALL venv\Scripts\activate

REM 3. Install/upgrade dependencies into the virtual environment
ECHO Installing dependencies...
pip install -r requirements.txt
pip install --upgrade setuptools
pip install -e .

REM === CONFIGURATION ==============================

REM Los Angeles County:
SET TARGET_REGION=la_county
SET REGION_FILTER=--fips 6037

SET FILTERED_PROPERTIES_CSV=Data\zillow_data_for_%TARGET_REGION%.csv
SET ANALYSIS_READY_DATA=Data\%TARGET_REGION%_prepared.csv

SET SEEDS= 42
REM ================================================

ECHO.
CHOICE /C YN /M "Do you want to run data preparation (filtering)?"
IF ERRORLEVEL 2 GOTO SkipPrep

ECHO.
ECHO 1. Running data preparation for %TARGET_REGION%...
python prepare_zillow_input_data.py %REGION_FILTER% --output_csv %FILTERED_PROPERTIES_CSV%

:SkipPrep

ECHO.
CHOICE /C YN /M "Do you want to run spatial join (merging)?"
IF ERRORLEVEL 2 GOTO SkipJoin

ECHO.
ECHO 2. Running spatial join...
python spatial_join.py --input_csv %FILTERED_PROPERTIES_CSV% --output_csv %ANALYSIS_READY_DATA%

:SkipJoin
SET RUN_UNWEIGHTED=N
SET RUN_WEIGHTED=N

ECHO.

CHOICE /C YN /M "Do you want to run C4F analysis without weights?"
IF %ERRORLEVEL% EQU 1 SET RUN_UNWEIGHTED=Y

CHOICE /C YN /M "Do you want to run C4F analysis with preset weights?"
IF %ERRORLEVEL% EQU 1 SET RUN_WEIGHTED=Y

FOR %%S IN (%SEEDS%) DO (
    ECHO.

    IF "%RUN_UNWEIGHTED%"=="Y" (
        ECHO.
        ECHO Running main analysis without weights for seed %%S...
        python main.py ^
            --data_path %ANALYSIS_READY_DATA% ^
            --error_col logerror ^
            --error_type regression ^
            --sensitive_cols ADI_STATERNK ^
            --continuous_sensitive_cols ADI_STATERNK ^
            --regular_cols latitude,longitude ^
            --algorithm kmedoids ^
            --distance gower ^
            --n_min 1 --n_max 35 ^
            --min_datapoints 15 ^
            --seed %%S ^
            --experiment ^
            --save_full_data ^
            --include_conditions "REG+SEN+ERR, SEN+ERR"
    )

    IF "%RUN_WEIGHTED%"=="Y" (
        FOR %%W IN (1.0 2.0) DO (
            ECHO.
            ECHO Running main analysis with ADI_STATERNK weight %%W for seed %%S...
            python main.py ^
                --data_path %ANALYSIS_READY_DATA% ^
                --error_col logerror ^
                --error_type regression ^
                --sensitive_cols ADI_STATERNK ^
                --continuous_sensitive_cols ADI_STATERNK ^
                --regular_cols latitude,longitude ^
                --algorithm kmedoids ^
                --distance gower ^
                --n_min 1 --n_max 35 ^
                --min_datapoints 15 ^
                --seed %%S ^
                --experiment ^
                --save_full_data ^
                --include_conditions "REG+SEN+ERR, SEN+ERR" ^
                --feature_weights ADI_STATERNK:%%W
        )
    )
)

ECHO.
ECHO All selected analyses complete.
ENDLOCAL
PAUSE