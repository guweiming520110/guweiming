# ============================================================
# 注册每日自动运行任务：数字彩票分析系统
# 用法（管理员 PowerShell）:
#   powershell -ExecutionPolicy Bypass -File register-daily-run.ps1
#
# 说明:
#   - 每天 11:35 运行 run_all.py（11:35 时 kl8 当日已开奖，
#     dlt/ssq 若当日有开奖也已结束）
#   - 使用 tensorflow conda 环境 python（F:\anaconda3\envs\tensorflow）
#   - 删除任务: schtasks /Delete /TN "数字彩票每日运行" /F
# ============================================================

$ErrorActionPreference = 'Stop'

# ---- 配置（按需修改）----
$ProjectDir = 'G:\数字彩票分析系统\Chao-Ji-Da-Le-Tou-AIYu-Ce'
$PythonExe  = 'F:\anaconda3\envs\tensorflow\python.exe'
$TaskName   = '数字彩票每日运行'
$Time       = '11:35'

if (-not (Test-Path $PythonExe)) {
    # 回退到系统 python
    $PythonExe = (Get-Command python).Source
    Write-Host "⚠️ 未找到 tensorflow 环境，使用系统 python: $PythonExe"
}
if (-not (Test-Path (Join-Path $ProjectDir 'run_all.py'))) {
    Write-Host "❌ 找不到 $ProjectDir\run_all.py，请修正 ProjectDir"
    exit 1
}

# 构建命令: 用 cmd /c 包装，避免 PowerShell 与 python 参数解析冲突
$Cmd = "cd /d `"$ProjectDir`" && `"$PythonExe`" run_all.py"
$Action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c $Cmd"

# 触发器：每天 21:00
$Trigger = New-ScheduledTaskTrigger -Daily -At $Time

# 设置：失败重试 2 次（间隔 5 分钟）；允许空密码交互
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

# 以当前用户运行（需已登录），避免需要明文密码
Register-ScheduledTask -TaskName $TaskName -Action $Action `
    -Trigger $Trigger -Settings $Settings -Description '数字彩票AI分析系统每日自动运行(更新数据+验证+预测+推送)' `
    -Force

Write-Host ""
Write-Host "✅ 计划任务已注册: $TaskName"
Write-Host "   时间: 每天 $Time"
Write-Host "   命令: $Cmd"
Write-Host ""
Write-Host "可用命令:"
Write-Host "   立即运行:   schtasks /Run /TN `"$TaskName`""
Write-Host "   查看状态:   schtasks /Query /TN `"$TaskName`" /V"
Write-Host "   删除任务:   schtasks /Delete /TN `"$TaskName`" /F"
