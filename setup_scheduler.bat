@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ==========================================
echo   A股数据自动更新 - Windows任务计划设置
echo ==========================================
echo.

:: 获取脚本所在目录
set "SCRIPT_DIR=%~dp0"
set "PYTHON_SCRIPT=%SCRIPT_DIR%auto_update_stockdata.py"
set "TASK_NAME=StockDataAutoUpdate"

:: 检查Python是否安装
echo [1/5] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python已安装

:: 检查依赖包
echo.
echo [2/5] 检查依赖包...
pip show baostock >nul 2>&1
if errorlevel 1 (
    echo 正在安装 baostock...
    pip install baostock -q
)

pip show pandas >nul 2>&1
if errorlevel 1 (
    echo 正在安装 pandas...
    pip install pandas -q
)

pip show schedule >nul 2>&1
if errorlevel 1 (
    echo 正在安装 schedule...
    pip install schedule -q
)

pip show mootdx >nul 2>&1
if errorlevel 1 (
    echo 正在安装 mootdx (备用数据源)...
    pip install "mootdx[all]" -q
)
echo [OK] 依赖包已安装

:: 检查脚本文件
echo.
echo [3/5] 检查脚本文件...
if not exist "%PYTHON_SCRIPT%" (
    echo [错误] 找不到主脚本: %PYTHON_SCRIPT%
    pause
    exit /b 1
)
echo [OK] 主脚本存在

:: 读取配置文件获取更新时间
echo.
echo [4/5] 读取配置...
set "UPDATE_TIME=16:00"
if exist "%SCRIPT_DIR%config.json" (
    for /f "tokens=*" %%i in ('powershell -Command "(Get-Content '%SCRIPT_DIR%config.json' | ConvertFrom-Json).schedule.update_time"') do (
        set "UPDATE_TIME=%%i"
    )
)
echo [OK] 更新时间: %UPDATE_TIME%

:: 创建任务计划
echo.
echo [5/5] 创建Windows任务计划...
echo.

:: 删除已存在的任务（如果存在）
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

:: 创建新任务
:: 使用pythonw.exe运行（无窗口），或python.exe（有窗口）
schtasks /create /tn "%TASK_NAME%" /tr "python.exe \"%PYTHON_SCRIPT%\" --once" /sc daily /st %UPDATE_TIME% /ru SYSTEM /rl HIGHEST /f

if errorlevel 1 (
    echo [错误] 创建任务计划失败
    echo 尝试使用普通用户权限创建...
    schtasks /create /tn "%TASK_NAME%" /tr "python.exe \"%PYTHON_SCRIPT%\" --once" /sc daily /st %UPDATE_TIME% /f
)

echo.
echo ==========================================
echo   任务计划创建完成！
echo ==========================================
echo.
echo 任务名称: %TASK_NAME%
echo 执行时间: 每天 %UPDATE_TIME%
echo 执行脚本: %PYTHON_SCRIPT%
echo.
echo 可用命令:
echo   - 查看任务:   schtasks /query /tn "%TASK_NAME%"
echo   - 手动运行:   schtasks /run /tn "%TASK_NAME%"
echo   - 删除任务:   schtasks /delete /tn "%TASK_NAME%" /f
echo   - 立即测试:   python "%PYTHON_SCRIPT%" --once
echo.
echo ==========================================
echo.

:: 询问是否立即测试
set /p RUN_TEST="是否立即运行一次更新测试？(Y/N): "
if /i "%RUN_TEST%"=="Y" (
    echo.
    echo 开始测试运行...
    echo.
    python "%PYTHON_SCRIPT%" --once
    echo.
    echo 测试运行完成，请检查日志输出
)

pause
