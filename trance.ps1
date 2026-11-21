# Run the trance_lora CLI (PowerShell). Usage: .\trance.ps1 generate --version v2 --out out\demo.mid
$HERE = $PSScriptRoot
$env:PYTHONPATH = "$HERE\_bootstrap;$HERE\src"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path "$HERE\tmp" | Out-Null
$env:TMP = "$HERE\tmp"; $env:TEMP = "$HERE\tmp"
Push-Location $HERE
& "$HERE\.venv\Scripts\python.exe" -m trance_lora @args
$code = $LASTEXITCODE
Pop-Location
exit $code
