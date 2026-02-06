@echo off
setlocal EnableDelayedExpansion

:: ================================================================
:: start.bat — Start the Multi-Modal RAG stack on Windows
::
:: Usage:
::   start.bat              Start all services
::   start.bat stop         Stop all services
::   start.bat status       Check service status
::   start.bat frontend     Start frontend only
::   start.bat backend      Start backend only
:: ================================================================

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

if "%~1"=="" goto :do_start
if /i "%~1"=="start" goto :do_start
if /i "%~1"=="stop" goto :do_stop
if /i "%~1"=="status" goto :do_status
if /i "%~1"=="frontend" goto :do_frontend
if /i "%~1"=="backend" goto :do_backend
echo Usage: start.bat [start^|stop^|status^|frontend^|backend]
exit /b 1

:: ── STATUS ──────────────────────────────────────────────────
:do_status
echo.
echo ========================================
echo   Multi-Modal RAG — Service Status
echo ========================================
echo.

:: Check Redis
call :check_port 6379 status_redis
if "!status_redis!"=="up" (
    echo   [OK]   Redis          :6379
) else (
    echo   [--]   Redis          :6379
)

:: Check Qdrant
call :check_port 6333 status_qdrant
if "!status_qdrant!"=="up" (
    echo   [OK]   Qdrant         :6333
) else (
    echo   [--]   Qdrant         :6333
)

:: Check FastAPI
call :check_port 8000 status_api
if "!status_api!"=="up" (
    echo   [OK]   FastAPI        :8000
) else (
    echo   [--]   FastAPI        :8000
)

:: Check Frontend
call :check_port 5173 status_fe
if "!status_fe!"=="up" (
    echo   [OK]   Frontend       :5173
) else (
    call :check_port 5174 status_fe2
    if "!status_fe2!"=="up" (
        echo   [OK]   Frontend       :5174
    ) else (
        echo   [--]   Frontend       :5173
    )
)

echo.
echo   Frontend    http://localhost:5173
echo   API         http://localhost:8000
echo   API Docs    http://localhost:8000/docs
echo   Qdrant      http://localhost:6333/dashboard
echo.
goto :eof

:: ── STOP ────────────────────────────────────────────────────
:do_stop
echo.
echo   Stopping services...
echo.

:: Kill frontend (node/vite on 5173 or 5174)
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5173 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /F /PID %%p >nul 2>&1
    echo   [OK]   Frontend stopped (PID %%p)
)
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5174 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /F /PID %%p >nul 2>&1
    echo   [OK]   Frontend stopped (PID %%p)
)

:: Kill FastAPI (uvicorn on 8000)
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /F /PID %%p >nul 2>&1
    echo   [OK]   FastAPI stopped (PID %%p)
)

:: Kill Qdrant (on 6333)
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":6333 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /F /PID %%p >nul 2>&1
    echo   [OK]   Qdrant stopped (PID %%p)
)

echo.
echo   All services stopped.
echo.
goto :eof

:: ── START FRONTEND ONLY ─────────────────────────────────────
:do_frontend
echo.
echo   Starting frontend...
cd /d "%PROJECT_DIR%\frontend"
call :check_port 5173 fe_check
if "!fe_check!"=="up" (
    echo   [OK]   Frontend already running on :5173
) else (
    start "MMRE-Frontend" cmd /c "npm run dev"
    echo   [OK]   Frontend starting on :5173
)
echo.
goto :eof

:: ── START BACKEND ONLY ──────────────────────────────────────
:do_backend
echo.
echo   Starting FastAPI backend...
cd /d "%PROJECT_DIR%"
call :check_port 8000 api_check
if "!api_check!"=="up" (
    echo   [OK]   FastAPI already running on :8000
) else (
    if exist ".venv\Scripts\python.exe" (
        start "MMRE-Backend" cmd /c ".venv\Scripts\python.exe -m uvicorn services.api_gateway.app:app --host 0.0.0.0 --port 8000 --log-level info"
    ) else if exist "..\.venv\Scripts\python.exe" (
        start "MMRE-Backend" cmd /c "..\.venv\Scripts\python.exe -m uvicorn services.api_gateway.app:app --host 0.0.0.0 --port 8000 --log-level info"
    ) else (
        start "MMRE-Backend" cmd /c "python -m uvicorn services.api_gateway.app:app --host 0.0.0.0 --port 8000 --log-level info"
    )
    echo   [OK]   FastAPI starting on :8000
    echo          Loading models (CLIP + MiniLM)... this takes ~15-30s
)
echo.
goto :eof

:: ── START ALL ───────────────────────────────────────────────
:do_start
echo.
echo ========================================
echo   Multi-Modal RAG — Starting Stack
echo ========================================
echo.

cd /d "%PROJECT_DIR%"

:: 1. Redis (optional — skip if not installed)
where redis-server >nul 2>&1
if !errorlevel! equ 0 (
    call :check_port 6379 redis_chk
    if "!redis_chk!"=="up" (
        echo   [OK]   Redis already running on :6379
    ) else (
        start "MMRE-Redis" /min cmd /c "redis-server"
        echo   [OK]   Redis starting on :6379
    )
) else (
    echo   [--]   Redis not installed (optional — skipping)
)

:: 2. Qdrant
call :check_port 6333 qdrant_chk
if "!qdrant_chk!"=="up" (
    echo   [OK]   Qdrant already running on :6333
) else (
    if exist "bin\qdrant.exe" (
        start "MMRE-Qdrant" /min cmd /c "bin\qdrant.exe --config-path qdrant_config.yaml"
        echo   [OK]   Qdrant starting on :6333
    ) else (
        echo   [--]   Qdrant binary not found at bin\qdrant.exe
        echo          Download from https://github.com/qdrant/qdrant/releases
    )
)

:: 3. FastAPI
call :check_port 8000 api_chk
if "!api_chk!"=="up" (
    echo   [OK]   FastAPI already running on :8000
) else (
    if exist ".venv\Scripts\python.exe" (
        start "MMRE-Backend" cmd /c ".venv\Scripts\python.exe -m uvicorn services.api_gateway.app:app --host 0.0.0.0 --port 8000 --log-level info"
    ) else if exist "..\.venv\Scripts\python.exe" (
        start "MMRE-Backend" cmd /c "..\.venv\Scripts\python.exe -m uvicorn services.api_gateway.app:app --host 0.0.0.0 --port 8000 --log-level info"
    ) else (
        start "MMRE-Backend" cmd /c "python -m uvicorn services.api_gateway.app:app --host 0.0.0.0 --port 8000 --log-level info"
    )
    echo   [OK]   FastAPI starting on :8000
    echo          Loading models... this takes ~15-30s
)

:: 4. Frontend
call :check_port 5173 fe_chk
if "!fe_chk!"=="up" (
    echo   [OK]   Frontend already running on :5173
) else (
    cd /d "%PROJECT_DIR%\frontend"
    start "MMRE-Frontend" cmd /c "npm run dev"
    cd /d "%PROJECT_DIR%"
    echo   [OK]   Frontend starting on :5173
)

echo.
echo ========================================
echo   All services starting!
echo ========================================
echo.
echo   Frontend    http://localhost:5173
echo   API         http://localhost:8000
echo   API Docs    http://localhost:8000/docs
echo   Qdrant      http://localhost:6333/dashboard
echo.
echo   Commands:
echo     start.bat status    — Check service health
echo     start.bat stop      — Stop everything
echo     start.bat frontend  — Start frontend only
echo     start.bat backend   — Start backend only
echo.
goto :eof

:: ── Helper: check if a port is listening ────────────────────
:check_port
set "_port=%~1"
set "%~2=down"
netstat -aon 2>nul | findstr ":%_port% " | findstr "LISTENING" >nul 2>&1
if !errorlevel! equ 0 set "%~2=up"
goto :eof
