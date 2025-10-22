Write-Host "Activating GenAI environment..." -ForegroundColor Green
& ".\genai_env\Scripts\Activate.ps1"
Write-Host "GenAI environment activated!" -ForegroundColor Green
Write-Host "Available packages: genai, ipython, openai (v0.27.10), tiktoken (v0.3.3)" -ForegroundColor Yellow
